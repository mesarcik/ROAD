import numpy as np
import torch
import cv2
from tqdm import tqdm
import copy

def remove_singles(data: list, *args) -> list:
    """
        Removes all single channel observations from dataset
        
        Parameters
        ----------
        data: list of baselines 
        args*: optional argument of lists that will modified according to data

        Returns
        -------
        data: list of baselines with single channels removed

    """
    _data = copy.deepcopy(data)
    _args = []
    indx = [i for i, d in enumerate(data) if d.shape[1] !=1]
    _data = [_data[i] for i in indx]

    if args:
        for a in args:
            for i in indx:
                _args.append(a[i])

        return _data,_args
    return _data

def reshape(data: list, dim: list, verbose:bool=False) -> np.array:
    """
        Resamples data to be of size dim
        
        Parameters
        ----------
        data: list of baselines 
        verbose: prints tqdm output

        Returns
        -------
        _data: np.array of baselines with the same dimensions

    """
    _data = []
    for d in tqdm(data,disable=not verbose):
        _temp = np.zeros((dim[0], dim[1],4))
        for p in range(4):
            _temp[...,0] = cv2.resize(d[...,p], dim)
        _data.append(_temp)

    return np.array(_data)
        
def patch(data: torch.Tensor, patch_size:int, verbose:bool=False) -> torch.Tensor:
    """
        Makes (N,C,h,w) shaped tensor into (N*(h/size)*(w/size),C,h/size, w/size)
        Note: this only works for square patches sizes
        
        Parameters
        ----------
        data: Tensor in form (N,C,h,w)
        patch_size: square patch size
        verbose: prints tqdm output

        Returns
        -------
        patches: Tensor of patches reshaped to (N*(h/size)*(w/size),C,h/size, w/size)

    """
    unfold = data.permute(0,2,3,1).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = unfold.contiguous().view(-1, data.shape[1], patch_size, patch_size)
    return patches

def unpatch(patches: torch.Tensor, patch_size:int, data_shape:tuple, verbose:bool=False) -> torch.Tensor:
    """
        Used to convert patches dataset into original dimenions
        
        Parameters
        ----------
        patches: Tensor in form (N,C,patch_size, patch_size)
        patch_size: square patch size
        data_shape: tuple of original data.shape
        verbose: prints tqdm output

        Returns
        -------
        data : Tensor of reconstructed patches

    """
    assert len(data_shape) == 4, "Data shape must be in form (N,C,...)"

    n_patches = data_shape[-2]//patch_size # only for square patches 
    N_orig = patches.shape[0]//n_patches**2
    unfold_shape = (N_orig, n_patches, n_patches, data_shape[1], patch_size, patch_size)
    
    data = patches.view(unfold_shape)
    data = data.permute(0, 3, 1, 4, 2, 5).contiguous()
    data = data.view(data_shape)

    return data


