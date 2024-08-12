import torch as th
import numpy as np
import torch
import torch.nn as nn



def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def select_batch_size(args, dataset):
    if 'TaxiNYC' in dataset:
        batch_size = int(args.batch_size_nyc * args.batch_ratio)
    if 'Crowd' in dataset or 'Flow' in dataset or 'Pop' in dataset:
        batch_size = int(args.batch_size_crowd * args.batch_ratio)
    if 'TaxiBJ' in dataset:
        batch_size = int(args.batch_size_taxibj * args.batch_ratio)
    if 'Graph' in dataset:
        batch_size = int(args.batch_size_graph_large * args.batch_ratio)
        if 'SH' in dataset:
            batch_size = int(batch_size*0.8)

    return batch_size

def select_patch_size(args, data):

    MIN, MID, MAX = [2,2,100]
    
    if 'TaxiBJ' in data or 'Flow' in data or 'TaxiNYC' in data or 'Crowd' in data or 'Pop' in data:
        patch_size = MIN

    if 'Graph' in data:
        patch_size = MAX
        if 'SH' in data:
            patch_size = MAX * 2

    return patch_size 


def tensor_to_adjacency_matrix(tensor):
    num_rows, num_cols = tensor.shape
    adjacency_matrix = torch.zeros((num_rows * num_cols, num_rows * num_cols), dtype=torch.int)

    # Define neighbors offset
    neighbors_offset = [(0, 1), (1, 0), (0, -1), (-1, 0),(0,0)]

    # Iterate through each element in the tensor
    for i in range(num_rows):
        for j in range(num_cols):
            current_index = i * num_cols + j

            # Check neighbors
            for offset_i, offset_j in neighbors_offset:
                new_i, new_j = i + offset_i, j + offset_j
                if 0 <= new_i < num_rows and 0 <= new_j < num_cols:
                    neighbor_index = new_i * num_cols + new_j
                    adjacency_matrix[current_index][neighbor_index] = 1

    return adjacency_matrix