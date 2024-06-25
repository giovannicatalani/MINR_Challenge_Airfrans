from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
from utils.conditioning import shift_modulation

#Code adapted from https://github.com/LouisSerrano/coral

class LatentToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.
    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(
        self, latent_dim, num_modulations, dim_hidden, num_layers, activation=nn.SiLU
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.activation = activation

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, dim_hidden), self.activation()]
            if num_layers > 2:
                for i in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden),
                               self.activation()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)
    

class GaussianEncoding(nn.Module):
    def __init__(self, embedding_size, scale, dims=2, gaussian=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale

        if gaussian:
            bvals = torch.randn(embedding_size // 2, dims) * scale
        else:
            bvals = 2.0 ** torch.linspace(0, scale, embedding_size // 2) - 1

            if dims == 1:
                bvals = bvals[:, None]

            elif dims == 2:
                bvals = torch.stack([bvals, torch.zeros_like(bvals)], dim=-1)
                bvals = torch.cat([bvals, torch.roll(bvals, 1, -1)], dim=0)

            else:
                tmp = (dims - 1) * (torch.zeros_like(bvals),)
                bvals = torch.stack([bvals, *tmp], dim=-1)

                tmp = [torch.roll(bvals, i, -1) for i in range(1, dims)]
                bvals = torch.cat([bvals, *tmp], dim=0)

        avals = torch.ones((bvals.shape[0]))
        self.avals = nn.Parameter(avals, requires_grad=False)
        self.bvals = nn.Parameter(bvals, requires_grad=False)

    def forward(self, tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input.
        """

        return torch.cat(
            [
                self.avals * torch.sin((2.0 * np.pi * tensor) @ self.bvals.T),
                self.avals * torch.cos((2.0 * np.pi * tensor) @ self.bvals.T),
            ],
            dim=-1,
        )

    
class MultiScaleModulatedFourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        num_frequencies=8,
        latent_dim=128,
        width=256,
        depth=3,
        include_input=True,
        scales=[1,5],
        max_frequencies=32,
        base_frequency=1.25,
    ):
        
        super().__init__()
       
        self.include_input = include_input
        self.scales = scales

        self.embeddings = nn.ModuleList([GaussianEncoding(embedding_size=num_frequencies * 2, scale=scale, dims=input_dim) for scale in scales])
        embed_dim = num_frequencies * 2 
        embed_dim += input_dim if include_input else 0
        self.in_channels = [embed_dim] + [width] * (depth - 1)

        self.out_channels = [width] * (depth - 1) + [width]
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_channels[k], self.out_channels[k]) for k in range(depth)]
        )
        self.final_linear = nn.Linear(len(self.scales) * width, output_dim)
        self.depth = depth
        self.hidden_dim = width

        self.num_modulations = self.hidden_dim * (self.depth - 1)
        
        self.latent_to_modulation = LatentToModulation(self.latent_dim, self.num_modulations, dim_hidden=256, num_layers=1)

        self.conditioning = shift_modulation

    def modulated_forward(self, x, z):
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
       
        features = self.latent_to_modulation(z)
        positions = [embedding(x) for embedding in self.embeddings]


        if self.include_input:
            positions = [torch.cat([pos, x], axis=-1) for pos in positions]


        pre_outs = [self.conditioning(pos, features, self.layers[:-1], torch.relu) for pos in positions]
        outs = [self.layers[-1](pre_out) for pre_out in pre_outs]

        # Concatenate the outputs from each scale
        concatenated_out = torch.cat(outs, axis=-1)

        # A final linear layer to combine multi-scale outputs
        final_out = self.final_linear(concatenated_out)

        return final_out.view(*x_shape, final_out.shape[-1])


#Processor Model Architecture
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))  # initializing with 1.0 as it's a common choice
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    
class Block(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, process_last_dim=2):
        super(Block, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.batchnorm1 = nn.BatchNorm1d(hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.batchnorm2 = nn.BatchNorm1d(out_features) if hidden_features != out_features else None
        self.swish = Swish()

        # Dimension matching layer for the residual connection
        self.match_dimensions = nn.Linear(in_features, out_features) if in_features != out_features else None

        # Separate processing for the last two elements
        self.process_last = nn.Linear(process_last_dim, out_features)

    def forward(self, x):
        residual = x

        out = self.swish(self.batchnorm1(self.linear1(x)))
        out = self.swish(self.linear2(out))

        if self.batchnorm2:
            out = self.batchnorm2(out)

        # Apply dimension matching if necessary
        if self.match_dimensions is not None:
            residual = self.match_dimensions(residual)

        # Process last two elements
        last_elements = self.process_last(x[:, -2:])

        return out + residual + last_elements

class MLPWithSkipConnections(nn.Module):
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super(MLPWithSkipConnections, self).__init__()
        self.blocks = nn.ModuleList(
            [Block(input_dim if i == 0 else hidden_dim, hidden_dim, hidden_dim if i != num_blocks - 1 else output_dim, process_last_dim=2) for i in range(num_blocks)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x