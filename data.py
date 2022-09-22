import numpy as np
import torch
import cv2
from tqdm import tqdm
import copy
import _pickle as cPickle

import torch 
from torch.utils.data import Dataset, DataLoader


class LOFARDataset(Dataset):
    def __init__(self, dataset_dir, patch_size, sourceTransform):
        self.dataset_dir = dataset_dir 
        self.patch_size = patch_size

        with open(self.dataset_dir, 'rb') as f:
            self.data, self.labels, self.source, self.ids, self.frequency_band = cPickle.load(f,encoding='latin')
        
        self.data, self.labels = self.remove_singles( self.data, self.labels)
        # TODO: when reshaping we are destroying the frequency band information, when this is necessary I need to change the code
        self.data = self.reshape(self.data, (256,256))

        self.data = np.clip(self.data, 0, np.percentile(self.data, 99))
        self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        self.data = torch.from_numpy(self.data).permute(0,3,1,2)
        self.data = self.patch(self.data, self.patch_size)


        self.sourceTransform = sourceTransform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datum = self.data[idx,...]
        label = str('-'.join(self.labels[idx]))

        if self.sourceTransform:
            datum = self.sourceTransform(datum)

        return datum, label



    def remove_singles(self, data: list, *args) -> list:
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

    def reshape(self, data: list, dim: list, verbose:bool=False) -> np.array:
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
            
    def patch(self, data: torch.Tensor, patch_size:int, verbose:bool=False) -> torch.Tensor:
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

    def unpatch(self, patches: torch.Tensor, patch_size:int, data_shape:tuple, verbose:bool=False) -> torch.Tensor:
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


