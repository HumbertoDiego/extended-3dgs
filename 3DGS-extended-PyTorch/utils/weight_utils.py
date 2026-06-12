import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def curv3d(sorted_scales, max_weight, min_weight):
    scales2=sorted_scales * sorted_scales
    mins = scales2.min(dim=1).values
    weights_0_to_1 = 3 * mins / scales2.sum(dim=1) # curvatures3D
    # Scale to [min_weight, max_weight]
    weights = weights_0_to_1 * (max_weight - min_weight) + min_weight
    return weights

def curv2d(sorted_cov2d, max_weight, min_weight):
    eigenvalues = torch.linalg.eigvalsh(sorted_cov2d) # P
    weights_0_to_1 = 2 * eigenvalues.min(dim=1).values / eigenvalues.sum(dim=1) # curvatures2D
    # Scale to [min_weight, max_weight]
    weights = weights_0_to_1 * (max_weight - min_weight) + min_weight
    return weights

def diskindex(sorted_scales, max_weight, min_weight):
    """
    fs=1 if the gaussian assume a disk format
    fs=0 otherwise 
    """
    s, i = torch.sort(sorted_scales)
    weights_0_to_1 = (1.0/3.0)*((1-s[:,0]/s[:,1])+(1-s[:,0]/s[:,2])+(s[:,1]/s[:,2])) # fs
    # Scale to [min_weight, max_weight]
    weights = weights_0_to_1 * (max_weight - min_weight) + min_weight
    return weights

def smallsizeindex(sorted_scales, max_weight, min_weight):
    weights_0_to_1 = torch.exp(-1 * sorted_scales.norm(dim=1))
    # Scale to [min_weight, max_weight]
    weights = weights_0_to_1 * (max_weight - min_weight) + min_weight
    return weights

   
def get_gaussian_index_acc(in_frustum, size_ok, rect, raster_settings):
    # index of the gaussian in the tile
    index_acc = torch.tensor([], dtype=torch.int).to('cuda')
    TILE_SIZE = raster_settings.tile_size 
    for h in range(0, raster_settings.image_height, TILE_SIZE):
        for w in range(0, raster_settings.image_width, TILE_SIZE):
            # check if the rectangle penetrate the tile
            over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
            over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
            in_tile = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
            # 3D gaussian in the tile, in front of camera, and not too thin
            in_mask = in_tile & in_frustum & size_ok
            if not in_mask.sum() > 0:
                continue
            # _, index = torch.sort(depths[in_mask])
            # Get indices where mask is True
            index = torch.nonzero(in_mask, as_tuple=False).squeeze()
            index_acc = torch.cat((index_acc, index), dim=0).to('cuda')
    return index_acc
    
def reusageindex(in_frustum, size_ok, rect, index, raster_settings): # Added in 23-07-2025
    # Count occurrences of each unique index in index_acc
    max_weight, min_weight = raster_settings.max_weight, raster_settings.min_weight
    index_acc = get_gaussian_index_acc(in_frustum, size_ok, rect, raster_settings)
    weights = torch.tensor([(index_acc == val).sum().item() for val in index]).to('cuda')
    # Reescale
    min_weight = weights.min()
    max_weight = weights.max()
    # Avoid division by zero if all elements are the same
    if max_weight > min_weight:
        # Scale to [0, 1]
        weights_0_to_1 = (weights - min_weight) / (max_weight - min_weight)
        # Scale to [min_weight, max_weight]
        weights = weights_0_to_1 * (max_weight - min_weight) + min_weight
    else:
        weights = torch.ones_like(weights)
    return weights