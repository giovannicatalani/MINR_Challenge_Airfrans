from pathlib import Path
import numpy as np
import torch
from utils.models import  MultiScaleModulatedFourierFeatures, MLPWithSkipConnections
import os

#Code adapted from https://github.com/LouisSerrano/coral



def create_inr_instance(cfg, input_dim=1, output_dim=1, device="cuda"):
    device = torch.device(device)
    
    inr = MultiScaleModulatedFourierFeatures(
        input_dim=input_dim,
        output_dim=output_dim,
        num_frequencies=cfg.inr.num_frequencies,
        latent_dim=cfg.inr.latent_dim,
        width=cfg.inr.hidden_dim,
        depth=cfg.inr.depth,
        include_input=cfg.inr.include_input,
        scales=cfg.inr.scale,
        max_frequencies=cfg.inr.max_frequencies,
        base_frequency=cfg.inr.base_frequency,
    ).to(device)

    return inr

# Function to load a model
def load_inr(model_weights, cfg,input_dim, output_dim):
    # ... load the INR model from model_path using the configuration cfg
    # load inr weights
    
    inr_in = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim, device="cuda"
    )
    inr_in.load_state_dict(model_weights)
    inr_in.eval()

    return inr_in

def load_processor(cfg, input_dim, output_dim, model_save_path=None):
    # Check if the best model already exists
    if model_save_path is not None and os.path.exists(model_save_path):
        print("Loading the best model from:", model_save_path)
        model = MLPWithSkipConnections(cfg.model.depth, input_dim, cfg.model.width, output_dim).cuda()
        model.load_state_dict(torch.load(model_save_path))
    else:
        model = MLPWithSkipConnections(cfg.model.depth, input_dim, cfg.model.width, output_dim).cuda()
    
    return model
