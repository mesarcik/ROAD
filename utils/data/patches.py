import numpy as np
import torch
from utils import args 

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

    dists_recon = reconstruct(np.expand_dims(dists,axis=-1),args)
    return dists_recon

def reconstruct(data:np.array, args:args, verbose:bool=False) -> torch.tensor:
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
    assert len(data.shape) == 4, "Data shape must be in form (N,...,C)"
    data = torch.from_numpy(data)


    n_patches = 256//args.patch_size # only for square patches 
    N_orig = data.shape[0]//n_patches**2
    unfold_shape = (N_orig, n_patches, n_patches, 4, args.patch_size, args.patch_size)
    
    data = data.view(unfold_shape)
    data = data.permute(0, 3, 1, 4, 2, 5).contiguous()
    data = data.view([N_orig, 4, 256, 256])

    return data
