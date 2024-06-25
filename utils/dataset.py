import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import copy


def process_dataset(dataset, training: bool, coef_norm = None):
    coord_x=dataset.data['x-position']
    coord_y=dataset.data['y-position']
    surf_bool=dataset.extra_data['surface']
    position = np.stack([coord_x,coord_y],axis=1)

    nodes_features,node_labels=dataset.extract_data()
    if training:
        print("Normalize train data")
        norm = True
        coef_norm = None
    else:
        print("Normalize not train data")
        norm = True
        coef_norm = coef_norm
        
    torchDataset=[]
    nb_nodes_in_simulations = dataset.get_simulations_sizes()
    start_index = 0
    for k,nb_nodes_in_simulation in enumerate(nb_nodes_in_simulations):
        end_index = start_index+nb_nodes_in_simulation
        simulation_positions = torch.tensor(position[start_index:end_index,:], dtype = torch.float) 
        simulation_features = torch.tensor(nodes_features[start_index:end_index,:], dtype = torch.float) 
        simulation_labels = torch.tensor(node_labels[start_index:end_index,:], dtype = torch.float) 
        simulation_surface = torch.tensor(surf_bool[start_index:end_index])
        
       

        sampleData=Data(pos=simulation_positions,
                        x=simulation_features, 
                        y=simulation_labels,
                        surf = simulation_surface.bool()) 
        torchDataset.append(sampleData)
        start_index += nb_nodes_in_simulation

        if norm and coef_norm is None:
            if k == 0:
                old_length = simulation_features.shape[0]
                mean_in = simulation_features.numpy().mean(axis = 0, dtype = np.double)
                mean_out = simulation_labels.numpy().mean(axis = 0, dtype = np.double)
            else:
                new_length = old_length + simulation_features.shape[0]
                mean_in += (simulation_features.numpy().sum(axis = 0, dtype = np.double) - simulation_features.shape[0]*mean_in)/new_length
                mean_out += (simulation_labels.numpy().sum(axis = 0, dtype = np.double) - simulation_features.shape[0]*mean_out)/new_length
                old_length = new_length 
    
    if norm and coef_norm is None:
        # Compute normalization
        mean_in = mean_in.astype(np.single)
        mean_out = mean_out.astype(np.single)
        # Umean = np.linalg.norm(data.x[:, 2:4], axis = 1).mean()     
        for k, data in enumerate(torchDataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double)/old_length
                std_out = ((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double)/old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_in)/new_length
                std_out += (((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_out)/new_length
                old_length = new_length
        
        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)
        std_out[3] *= 1
        
        #Dont normalize normals
        mean_in[5:7] = 1
        std_in[5:7] = 0

        # Normalize
        for data in torchDataset:
            data.x = (data.x - mean_in)/(std_in + 1e-8)
            data.y = (data.y - mean_out)/(std_out + 1e-8)
        
        coef_norm = (mean_in, std_in, mean_out, std_out)   
        #Train-Val Split  
        np.random.seed(42)  # Set the seed for deterministic behavior
        shuffled_indices = np.random.permutation(len(torchDataset))
        torchDataset = [torchDataset[i] for i in shuffled_indices]  # Shuffle the dataset deterministically

        val_size = int(len(torchDataset) * 0.9)
        train_size = int(len(torchDataset) * 0.9)
        torchDataset_train = torchDataset[:train_size]
        torchDataset_val = torchDataset[val_size:]

        torchDataset = (torchDataset_train, torchDataset_val, coef_norm)

    
    elif coef_norm is not None:
        # Normalize
        for data in torchDataset:
            data.x = (data.x - coef_norm[0])/(coef_norm[1] + 1e-8)
            data.y = (data.y - coef_norm[2])/(coef_norm[3] + 1e-8)
    

    return torchDataset

def process_dataset_field(original_dataset, target_field, num_points=None, global_coef_norm=None):
    """
    Apply subsampling and global normalization to a dataset.

    Args:
        dataset (list of Data): The dataset to process, each item is a PyTorch Geometric Data object.
        target_field (list of str): The target fields to be included in the output.
        num_points (int, optional): Number of points to subsample for each data object. If None, no subsampling is applied.
        global_coef_norm (tuple, optional): Tuple of global min and max for pos, x, and y (global_min_pos, global_max_pos, global_min_x, global_max_x, global_min_y, global_max_y). 
                                           If None, these values will be computed.

    Returns:
        list of Data: Processed dataset.
    """
    # Create a deep copy of the dataset to keep the original dataset unchanged
    dataset = copy.deepcopy(original_dataset)

    # Define field indices
    field_indices = {
        'implicit_distance': (4, 5),  # x[:, 4]
        'normals': (5, 7),            # x[:, 5:7]
        'U': (0, 2),                  # y[:, 0:2]
        'Ux':  (0, 1),                 # y[:, 0]
        'Uy': (1, 2),                 # y[:, 1]
        'p': (2, 3),                  # y[:, 2]
        'nut': (3, 4),                # y[:, 3]                           
        'all_outputs': (0, 4),         # y[:, 0:4]
        'p_surf': (2, 3)
    }

        # Initialize global min and max
    if global_coef_norm is None:
        global_min_pos = np.full(dataset[0].pos.shape[1], float('inf'))
        global_max_pos = np.full(dataset[0].pos.shape[1], float('-inf'))
        global_min_x = np.full(dataset[0].x.shape[1], float('inf'))
        global_max_x = np.full(dataset[0].x.shape[1], float('-inf'))
        global_min_y = np.full(dataset[0].y.shape[1], float('inf'))
        global_max_y = np.full(dataset[0].y.shape[1], float('-inf'))

        for data in dataset:
            global_min_pos = np.minimum(global_min_pos, data.pos.min(0)[0].numpy())
            global_max_pos = np.maximum(global_max_pos, data.pos.max(0)[0].numpy())
            global_min_x = np.minimum(global_min_x, data.x.min(0)[0].numpy())
            global_max_x = np.maximum(global_max_x, data.x.max(0)[0].numpy())
            global_min_y = np.minimum(global_min_y, data.y.min(0)[0].numpy())
            global_max_y = np.maximum(global_max_y, data.y.max(0)[0].numpy())


        global_coef_norm = global_min_pos, global_max_pos, global_min_x, global_max_x, global_min_y, global_max_y 

    else:
        global_min_pos, global_max_pos, global_min_x, global_max_x, global_min_y, global_max_y = global_coef_norm

    

    processed_dataset = []

    for data in dataset:       
        
        data.cond = data.x[0, 2:4].clone().unsqueeze(0).type(torch.float32) 

        #Normalize inputs to -1 and 1
        data.pos = 2 * (data.pos - global_min_pos) / (global_max_pos - global_min_pos) - 1
        data.sdf_norm = 2 * (data.x[:, 4:5] - global_min_x[4:5]) / (global_max_x[4:5] - global_min_x[4:5]) - 1
        data.normals_norm = 2 * (data.x[:, 5:7] - global_min_x[5:7]) / (global_max_x[5:7] - global_min_x[5:7]) - 1
        
        # Select target field data
        if target_field in field_indices:
            start_idx, end_idx = field_indices[target_field]

            if target_field == 'normals':
                # Select only surface points for normals
                idx = data.surf==True
                data.pos = data.pos[idx]
                data.x = data.x[idx]
                data.y = data.y[idx]
                data.surf = data.surf[idx]
                data.input = data.pos.type(torch.float32)
                data.output = data.normals_norm[idx].type(torch.float32)

            elif target_field == 'p_surf':
                # Select only surface points for normals
                idx = data.surf==True
                data.pos = data.pos[idx]
                data.x = data.x[idx]
                data.y = data.y[idx]
                data.surf = data.surf[idx]
                data.input = data.pos.type(torch.float32)
                data.output = data.y[:, start_idx:end_idx].type(torch.float32)

            elif target_field == 'implicit_distance':
                data.output = data.x[:, start_idx:end_idx].type(torch.float32)
                data.input = data.pos.type(torch.float32)
            else:
                data.output = data.y[:, start_idx:end_idx].type(torch.float32)
                data.input = torch.cat((data.pos.type(torch.float32),data.sdf_norm), dim=1).type(torch.float32)
        
        processed_dataset.append(data)
    
    if num_points is not None and num_points<len(processed_dataset[0].pos):
        # Subsample the entire dataset first
        processed_dataset = subsample_dataset(processed_dataset, num_points)


    return processed_dataset,global_coef_norm

def subsample_dataset(dataset, num_points):
    """
    Subsample each data object in the dataset to a fixed number of points.

    Args:
        dataset (list of Data): The dataset to process, each item is a PyTorch Geometric Data object.
        num_points (int): The number of points to subsample for each data object.

    Returns:
        list of Data: The subsampled dataset.
    """
    
    subsampled_dataset = []
    for data in dataset:
    
        idx = np.random.choice(data.pos.size(0), num_points, replace=False)
        subsampled_data = data.clone()  # Clone to avoid modifying the original data
        subsampled_data.pos = data.pos[idx]
        subsampled_data.x = data.x[idx]
        subsampled_data.y = data.y[idx]
        subsampled_data.input = data.input[idx]
        subsampled_data.output = data.output[idx]
        if hasattr(data, 'surf'):
            subsampled_data.surf = data.surf[idx]
        subsampled_dataset.append(subsampled_data)
        
    return subsampled_dataset



class Struct(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value
