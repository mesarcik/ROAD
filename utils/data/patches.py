import numpy as np
import torch
from utils import args 


def combine(x:torch.tensor, dim_begin:int, dim_end:int)->torch.tensor:
    """
        Joins together axis from dim_begin to dim_end

        Parameters
        ----------
        x: tensor to be joined
        dim_begin: start dimensions 
        dim_end: end dimensions 

        Returns
        -------
        combined tensor

    """
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)

def reconstruct_distances(distances:np.array, args:args):
    """
        Reconstruct distance vector to original dimensions when using patches

        Parameters
        ----------
        distances (np.array): Vector of per neighbour distances
        args (Namespace): cmd_args 

        Returns
        -------
        dists (np.array): reconstructed patches if necessary

    """

    dists = np.mean(distances, axis = tuple(range(1,distances.ndim)))
    dists = np.array([[d]*args.patch_size**2 for d in dists]).reshape(len(dists), 
                                                                args.patch_size, 
                                                                args.patch_size)

    dists_recon = reconstruct(np.expand_dims(dists,axis=1),args)
    return dists_recon

def reconstruct(data, args:args, verbose:bool=False) -> torch.tensor:
    """
        Used to convert patches dataset into original dimenions
        Fixed to 4 channels with original shape of 256
        
        Parameters
        ----------
        data: data in patches to be unpatched
        args: utils.args
        verbose: prints tqdm output

        Returns
        -------
        data : tensor of reconstructed patches

    """
    assert (type(data) == np.ndarray or type(data) == torch.Tensor), "Data must be either numpy array or torch Tensor"
    assert len(data.shape) == 4, "Data shape must be in form (N,...,C)"

    if type(data) == np.ndarray:
        data = torch.from_numpy(data)

    n_patches = 256//args.patch_size # only for square patches 
    N_orig = data.shape[0]//n_patches**2
    unfold_shape = (N_orig, n_patches, n_patches, data.shape[1], args.patch_size, args.patch_size)
    
    _data = data.view(unfold_shape)
    _data = _data.permute(0, 3, 1, 4, 2, 5).contiguous()
    _data = _data.view([N_orig, data.shape[1], 256, 256])

    return _data
