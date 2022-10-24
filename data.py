import numpy as np
import torch
from tqdm import tqdm
import torch 
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import h5py 

from utils import args
from utils.data.defaults import *


def get_data(args, anomaly:str=None):
    hf = h5py.File(args.data_path,'r')

    #TODO Fix limit
    #if args.limit != 'None':
    #    mask = np.random.randint(low=0, high= len(train_data), size=int(args.limit)) 
    #    train_data = [train_data[m] for m in mask] 
    #    train_labels = [train_labels[m] for m in mask]
    #    train_frequency = [train_frequency[m] for m in mask] 

    (train_data, val_data, 
    train_labels, val_labels, 
    train_frequency_band, val_frequency_band,
    train_source, val_source) = train_test_split(hf['train_data/data'][:], 
                                                 hf['train_data/labels'][:].astype(str), 
                                                 hf['train_data/frequency_band'][:], 
                                                 hf['train_data/source'][:].astype(str), 
                                                 test_size=0.05, 
                                                 random_state=args.seed)

    train_dataset = LOFARDataset(train_data, 
                                 train_labels, 
                                 train_frequency_band, 
                                 train_source,
                                 args)

    val_dataset =   LOFARDataset(val_data, 
                                 val_labels, 
                                 val_frequency_band, 
                                 val_source,
                                 args)

    if anomaly is None: anomaly = args.anomaly_class
    test_dataset = LOFARDataset(_join(hf, anomaly, 'data'),
                                _join(hf, anomaly, 'labels').astype(str),
                                _join(hf, anomaly, 'frequency_band'),
                                _join(hf, anomaly, 'source').astype(str),
                                args)

    return train_dataset, val_dataset, test_dataset
                        
def _join(hf:h5py.File, anomaly:str, field:str)->np.array:
    """
        Joins together the normal and anomalous testing data
        
        Parameters
        ----------
        hf: h5py dataset 
        anomaly: anomalous class
        field: the field that is meant to be concatenated along 

        Returns
        -------
        data: concatenated array

    """
    data = np.concatenate([hf['test_data/{}'.format(field)][:], 
                           hf['anomaly_data/{}/{}'.format(anomaly,field)]],axis=0)
                        
    return data 

class LOFARDataset(Dataset):
    def __init__(self, 
            data:np.array, 
            labels:np.array, 
            frequency_band:np.array, 
            source:np.array, 
            args:args, 
            sourceTransform=None):# set default types

        self.args = args
        self.labels = labels
        self.stations = self.encode_stations(source)
        self.stations = torch.from_numpy(np.repeat(self.stations, int(256/args.patch_size)**2, axis=0))

        _t = data.shape
        self.original_shape = [_t[0], _t[3], _t[2], _t[1]] # change to be [N,C,...]

        self.data = self.normalise(data)
        self.plot_spectra(self.data, '/tmp/sample')
        self.data = torch.from_numpy(self.data).permute(0,3,1,2)
        self.data = self.patch(self.data)

        self.frequency_band = torch.from_numpy(frequency_band).permute(0,3,1,2)
        self.frequency_band = self.patch(self.frequency_band)[:,0,0,[0,-1]]#use start, end frequencies per patch
        self.frequency_band = torch.from_numpy(self.encode_frequencies(self.frequency_band.numpy()))
        #self.frequency_band = ((self.frequency_band- torch.min(self.frequency_band)) / 
        #                        (torch.max(self.frequency_band) - torch.min(self.frequency_band)))


        self.sourceTransform = sourceTransform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datum = self.data[idx,...]
        label = ''#str('-'.join(self.labels[idx]))
        frequency = self.frequency_band[idx,...]
        station = self.stations[idx]

        if self.sourceTransform:
            datum = self.sourceTransform(datum)

        return datum, label, frequency, station

    def encode_frequencies(self, frequency_band:np.array)->np.array:
        """
            Extracts stations from source feild of dataset and encodes 
            
            Parameters
            ----------
            sources:  np.array of sources for each baseline

            Returns
            -------
            encoded_stations: station names encoded between 0-1

        """
        encoded_frequencies = []

        for f in frequency_band:
            _temp = '-'.join((str(f[0]),str(f[1])))
            encoded_frequencies.append(np.where(np.array(default_frequency_bands) == _temp)[0][0])

        return np.array(encoded_frequencies)

    def encode_stations(self, sources:list)->np.array:
        """
            Extracts stations from source feild of dataset and encodes 
            
            Parameters
            ----------
            sources:  np.array of sources for each baseline

            Returns
            -------
            encoded_stations: station names encoded between 0-1

        """
        stations = np.array([s.split('_')[2] for s in sources])
        #_u = np.unique(stations)
        #mapping  = np.linspace(0, 1, len(_u))
        encoded_stations = np.array([np.where(np.array(default_stations) == s)[0][0] for s in stations]) 
        #encoded_stations = mapping[indexes]
        return encoded_stations

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
