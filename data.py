import numpy as np
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
    for a in args:
        for i in indx:
            _args.append(a[i])

    return _data,_args

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
        
