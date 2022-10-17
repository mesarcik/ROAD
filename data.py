import numpy as np
import torch
import cv2
from tqdm import tqdm
import copy
import _pickle as cPickle

import torch 
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from utils import args

def get_data(args, anomaly:str=None):
    with open(args.data_path, 'rb') as f:
        data, labels, source, ids, frequency_band = cPickle.load(f,encoding='latin')

    labels = [('-').join(l) for l in labels]

    train_inds = [i for i,l in enumerate(labels) if l =='']
    if anomaly is None:
        test_inds = [i for i,l in enumerate(labels) if args.anomaly_class == l ]
    else:                                                                         
        test_inds = [i for i,l in enumerate(labels) if anomaly == l ]              


    train_data  = [data[i] for i in  train_inds]
    train_labels = [labels[i] for i in train_inds]
    train_frequency = [frequency_band[i] for i in train_inds]

    test_data = [data[i] for i in test_inds]
    test_labels = [labels[i] for i in test_inds]
    test_frequency = [frequency_band[i] for i in test_inds]


    # add equal amounts of training data to the test data
    _ext = len(test_data)
    test_data.extend(train_data[-_ext:])
    test_labels.extend(train_labels[-_ext:])
    test_frequency.extend(train_frequency[-_ext:])

    train_data = train_data[:-_ext]
    train_labels = train_labels[:-_ext]
    train_frequency = train_frequency[:-_ext]

    if args.limit != 'None':
        mask = np.random.randint(low=0, high= len(train_data), size=int(args.limit)) 
        train_data = [train_data[m] for m in mask] 
        train_labels = [train_labels[m] for m in mask]
        train_frequency = [train_frequency[m] for m in mask] 

    train_dataset = LOFARDataset(train_data,
            train_labels,
            train_frequency,
            args)

    test_dataset = LOFARDataset(test_data,
            test_labels,
            test_frequency,
            args)
    return train_dataset, test_dataset
                        

class LOFARDataset(Dataset):
    def __init__(self, data:list, labels:list, frequency_band:list, args:args, sourceTransform=None):# set default types
        self.args = args

        self.data, self.labels, self.frequency_band = self.remove_singles(data, labels, frequency_band)

        self.data, self.frequency_band = self.reshape((256,256)) 

        _t = self.data.shape
        self.original_shape = [_t[0], _t[3], _t[2], _t[1]] # change to be [N,C,...]

        self.data = self.normalise(self.data)
        self.plot_spectra(self.data, '/tmp/sample')
        self.data = torch.from_numpy(self.data).permute(0,3,1,2)
        self.data = self.patch(self.data)

        self.frequency_band = torch.from_numpy(self.frequency_band).permute(0,3,1,2)
        self.frequency_band = self.patch(self.frequency_band)[:,0,0,[0,-1]]#use start, end frequencies per patch
        self.frequency_band = (self.frequency_band- torch.min(self.frequency_band)) / (torch.max(self.frequency_band) - torch.min(self.frequency_band))

        self.sourceTransform = sourceTransform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datum = self.data[idx,...]
        label = ''#str('-'.join(self.labels[idx]))
        frequency = self.frequency_band[idx,...]

        if self.sourceTransform:
            datum = self.sourceTransform(datum)

        return datum, label, frequency


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

    def plot_spectra(self, data: np.array, loc:str)->None:
        indxs = np.random.randint(low=0, high= len(self.data), size=16)
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

    def reshape(self, dim: list, verbose:bool=False) -> np.array:
        """
            Resamples data and frequency_band to be of size dim
            
            Parameters
            ----------
            verbose: prints tqdm output

            Returns
            -------
            _data: np.array of baselines with the same dimensions
            _freq: np.array of baselines with the same dimensions

        """
        _data = np.zeros((len(self.data), dim[0], dim[1], 4))
        _freq = np.zeros((len(self.frequency_band), dim[0], dim[1], 1))

        for i,d in enumerate(tqdm(self.data,disable=not verbose)):
            temp_freq = np.vstack([self.frequency_band[i]]*len(self.frequency_band[i]))
            _freq[i,...,0] = cv2.resize(temp_freq, dim)

            for p in range(4):
                _data[i,...,p] = cv2.resize(d[...,p], dim)

        return _data, _freq
            
    def patch(self, _input:torch.tensor, verbose:bool=False) -> torch.tensor:
        """
            Makes (N,C,h,w) shaped tensor into (N*(h/size)*(w/size),C,h/size, w/size)
            Note: this only works for square patches sizes
            
            Parameters
            ----------
            verbose: prints tqdm output

            Returns
            -------
            patches: tensor of patches reshaped to (N*(h/size)*(w/size),C,h/size, w/size)

        """
        unfold = _input.data.permute(0,2,3,1).unfold(1, 
                 self.args.patch_size, 
                 self.args.patch_size).unfold(2, 
                        self.args.patch_size, 
                        self.args.patch_size)
        patches = unfold.contiguous().view(-1, 
                _input.shape[1], 
                self.args.patch_size, 
                self.args.patch_size)
        return patches

