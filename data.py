import numpy as np
import torch
import cv2
from tqdm import tqdm
import copy
import _pickle as cPickle

import torch 
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


class LOFARDataset(Dataset):
    def __init__(self, dataset_dir, patch_size, sourceTransform=None, limit =5000):# set default types
        self.dataset_dir = dataset_dir 
        self.patch_size = patch_size

        with open(self.dataset_dir, 'rb') as f:
            self.data, self.labels, self.source, self.ids, self.frequency_band = cPickle.load(f,encoding='latin')
        
        self.data, self.labels = self.remove_singles( self.data, self.labels)
        # TODO: when reshaping we are destroying the frequency band information, when this is necessary I need to change the code
        self.data = self.reshape(self.data, (256,256))

        ###########################################################################
        mask = np.random.randint(low=0, high= 12285, size=limit) 
        self.labels = np.array(self.labels, dtype='object')
        self.data, self.labels = self.data[mask], self.labels[mask]
        ###########################################################################

        indxs = np.random.randint(low=0, high= len(self.data), size=16)
        self.data = self.normalise(self.data)
        #self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        self.plot_spectra(self.data, indxs, '/tmp/sample')
        self.data = torch.from_numpy(self.data).permute(0,3,1,2)
        self.data = self.patch(self.data, self.patch_size)


        self.sourceTransform = sourceTransform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datum = self.data[idx,...]
        label = ''#str('-'.join(self.labels[idx]))

        if self.sourceTransform:
            datum = self.sourceTransform(datum)

        return datum, label

    def normalise(self, data:np.array)->np.array:
        """
            perpendicular polarisation normalisation

        """
        _data = np.zeros(data.shape)
        for i, spec in enumerate(data):
            for pol in range(4):
                _min, _max = np.percentile(spec[...,pol], [5,95])
                temp = np.clip(spec[...,pol],_min, _max)
                temp  = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
                _data[i,...,pol] = temp
        _data = np.nan_to_num(_data, 0)
        return _data

    def plot_spectra(self, data: np.array, indxs: np.array, loc:str)->None:
        sq = int(np.sqrt(len(indxs)))
        fig, axs = plt.subplots(sq, sq, figsize=(10,10))
        _c = 0
        for i in range(sq):
            for j in range(sq):
                axs[i,j].imshow(data[indxs[_c],...,0], aspect='auto',interpolation='nearest', vmin =0, vmax=1)
                _c+=1
        plt.savefig(loc, dpi=300)

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
                temp_args = []
                for i in indx:
                    temp_args.append(a[i])
                _args.append(temp_args)
            return _data,*_args
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
        _data = np.zeros((len(data), dim[0], dim[1], 4))
        for i,d in enumerate(tqdm(data,disable=not verbose)):
            for p in range(4):
                _data[i,...,p] = cv2.resize(d[...,p], dim)
        return _data
            
    def patch(self, data: torch.tensor, patch_size:int, verbose:bool=False) -> torch.tensor:
        """
            Makes (N,C,h,w) shaped tensor into (N*(h/size)*(w/size),C,h/size, w/size)
            Note: this only works for square patches sizes
            
            Parameters
            ----------
            data: tensor in form (N,C,h,w)
            patch_size: square patch size
            verbose: prints tqdm output

            Returns
            -------
            patches: tensor of patches reshaped to (N*(h/size)*(w/size),C,h/size, w/size)

        """
        unfold = data.permute(0,2,3,1).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = unfold.contiguous().view(-1, data.shape[1], patch_size, patch_size)
        return patches

    def unpatch(self, patches: torch.tensor, patch_size:int, data_shape:tuple, verbose:bool=False) -> torch.tensor:
        """
            Used to convert patches dataset into original dimenions
            
            Parameters
            ----------
            patches: tensor in form (N,C,patch_size, patch_size)
            patch_size: square patch size
            data_shape: tuple of original data.shape
            verbose: prints tqdm output

            Returns
            -------
            data : tensor of reconstructed patches

        """
        assert len(data_shape) == 4, "Data shape must be in form (N,C,...)"

        n_patches = data_shape[-2]//patch_size # only for square patches 
        N_orig = patches.shape[0]//n_patches**2
        unfold_shape = (N_orig, n_patches, n_patches, data_shape[1], patch_size, patch_size)
        
        data = patches.view(unfold_shape)
        data = data.permute(0, 3, 1, 4, 2, 5).contiguous()
        data = data.view(data_shape)

        return data


