import numpy as np
import torch
from tqdm import tqdm
import torch 
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import h5py 

from utils import args
from utils.data.defaults import *
from utils.data.patches import reconstruct


def get_data(args, transform=None, anomaly:str=None):
    hf = h5py.File(args.data_path,'r')

    (train_data, val_data, 
    train_labels, val_labels, 
    train_frequency_band, val_frequency_band,
    train_source, val_source) = train_test_split(_temp_join_test(hf, 0,'data'), #hf['train_data/data'][:],#
                                                 _temp_join_test(hf, 0, 'labels').astype(str), #hf['train_data/labels'][:].astype(str),#
                                                 _temp_join_test(hf, 0, 'frequency_band'), #hf['train_data/frequency_band'][:],#
                                                 _temp_join_test(hf, 0, 'source').astype(str), #hf['train_data/source'][:].astype(str),#
                                                 test_size=0.05, 
                                                 random_state=args.seed)

    train_dataset = LOFARDataset(train_data, 
                                 train_labels, 
                                 train_frequency_band, 
                                 train_source,
                                 args,
                                 transform=transform,
                                 roll=True)

    val_dataset =   LOFARDataset(val_data, 
                                 val_labels, 
                                 val_frequency_band, 
                                 val_source,
                                 args,
                                 transform=None)

    if anomaly is None: anomaly = args.anomaly_class
    test_dataset = LOFARDataset(_join(hf, anomaly, 'data'),
                                _join(hf, anomaly, 'labels').astype(str),
                                _join(hf, anomaly, 'frequency_band'),
                                _join(hf, anomaly, 'source').astype(str),
                                args,
                                transform=None)

    return train_dataset, val_dataset, test_dataset


def _temp_join_test(hf:h5py.File,amount:float ,field:str):
    """
        temp function to evaluate how much data we need to add

    """

    _test = hf['test_data/{}'.format(field)][:]
    _indx  = np.random.default_rng(42).choice(len(_test), size=int(len(_test)*amount)) 
    data = np.concatenate([_test[_indx], 
                         hf['train_data/{}'.format(field)][:]],axis=0)
    return data

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
    #indx = np.arange(len(data))
    #indx = np.array([i for i in indx if i not in excluded_sources[anomaly]])
                        
    return data#[indx]

class LOFARDataset(Dataset):
    def __init__(self, 
            data:np.array, 
            labels:np.array, 
            frequency_band:np.array, 
            source:np.array, 
            args:args, 
            transform=None,
            roll=False):# set default types

        if roll:
            _data, _frequency_band = self.circular_shift(data, 
                                                       frequency_band, 
                                                       args.patch_size//2)
            data = np.concatenate([data, _data],axis=0)
            frequency_band= np.concatenate([frequency_band, _frequency_band],axis=0)
            labels = np.concatenate([labels, labels],axis=0)
            source = np.concatenate([source, source],axis=0)
            

        self.args = args
        self.source = source
        self.n_patches = int(256/args.patch_size)

        self.labels= np.repeat(labels, self.n_patches**2, axis=0)

        self.stations = self.encode_stations(source)
        self.stations = torch.from_numpy(np.repeat(self.stations, self.n_patches**2, axis=0))

        # Need to flatten all polarisations
        data = self.normalise(data)

        _t = data.shape
        self.original_shape = [_t[0], _t[3], _t[2], _t[1]] # change to be [N,C,...]

        self.data = torch.from_numpy(data).permute(0,3,1,2)
        self.data = self.patch(self.data)

        (self.context_labels, 
                self.context_images_pivot, 
                self.context_images_neighbour) = self.context_prediction(10, args)

        self.jittered_data = self.jitter(args)

        self.frequency_band = torch.from_numpy(frequency_band).permute(0,3,1,2)
        self.frequency_band = self.patch(self.frequency_band)[:,0,0,[0,-1]]#use start, end frequencies per patch
        self.frequency_band = torch.from_numpy(self.encode_frequencies(self.frequency_band.numpy()))

        self.transform=transform 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datum = self.data[idx,...]
        label = self.labels[idx]
        frequency = self.frequency_band[idx] 
        station = self.stations[idx]
        context_label = self.context_labels[idx]
        context_image_pivot = self.context_images_pivot[idx]
        context_image_neighbour = self.context_images_neighbour[idx]
        jittered_datum = self.jittered_data[idx]

        if self.transform:
            datum = self.transform(datum)

        return datum, label, frequency, station, context_label, context_image_pivot, context_image_neighbour, jittered_datum

    def circular_shift(self,
                       data:np.array, 
                       frequency_band:np.array, 
                       max_roll:int)->(np.array, np.array):
        """
            Circular shift to each spectrogram by random amount 
            
            Parameters
            ----------
            data:  spectrogram data
            frequency_band:  frequency_band info
            max_roll: the maximum shift amount 

            Returns
            -------
            data: circular shifted data
            freq: circular shifted frequency information 

        """

        _data =  np.zeros(data.shape)
        _frequency_band = np.zeros(frequency_band.shape)

        for i in range(len(data)):
            r = np.random.randint(-max_roll,max_roll)
            _data[i,:] = np.roll(data[i,:], r, axis =1)
            _data[i,:] = np.roll(_data[i,:], r, axis =0)
            _frequency_band[i,:] = np.roll(frequency_band[i,:], r, axis =1)
            _frequency_band[i,:] = np.roll(_frequency_band[i,:], r, axis =0)
        
        return _data, _frequency_band

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
        #encoded_frequencies = []
        #for f in frequency_band:
        #    _temp = '-'.join((str(f[0]),str(f[1])))
        #    encoded_frequencies.append(_temp)
        #print(np.unique(encoded_frequencies))

        encoded_frequencies = []
        for f in frequency_band:
            _temp = '-'.join((str(f[0]),str(f[1])))
            encoded_frequencies.append(0)#np.where(np.array(default_frequency_bands[self.args.patch_size]) == _temp)[0][0])

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
            for pol in range(data.shape[-1]):
                #_min, _max = np.percentile(spec[...,pol], [self.args.amount,100-self.args.amount])
                temp = spec[...,pol]#np.clip(spec[...,pol],_min, _max)
                temp = np.log(temp)
                temp  = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
                _data[i,...,pol] = temp
        _data = np.nan_to_num(_data, 0)
        return _data

            
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

    def context_prediction(self, N:int, args:args) -> (torch.tensor, torch.tensor, torch.tensor):
        """
            Arranges a context prediction dataset
            
            Parameters
            ----------
            N: Number of context samples to choose per image
            args: cmd arugments

            Returns
            -------
            patches: tensor of patches reshaped to (N*(h/size)*(w/size),C,h/size, w/size)

        """

        context_labels = np.ones([self.data.shape[0]],dtype='int')
        context_images_pivot  = np.ones([self.data.shape[0],
                                          self.data.shape[1],
                                          args.patch_size, 
                                          args.patch_size],dtype='float32')
        context_images_neighbour = np.zeros([self.data.shape[0],
                                             self.data.shape[1],
                                             args.patch_size, 
                                             args.patch_size],dtype='float32')
        _indx = 0
        _locations = [-self.n_patches-1, -self.n_patches, -self.n_patches+1, -1, +1, +self.n_patches-1, +self.n_patches, +self.n_patches+1]

        for _image_indx in range(self.n_patches**2, self.data.shape[0]+1,self.n_patches**2):
            temp_patches = self.data[_image_indx-self.n_patches**2:_image_indx,...] #selects 1 image in patch form has dimensioons (64, 4, 32, 32)

            for _patch_index in range(self.n_patches**2):
                if _patch_index < self.n_patches:
                    # TOPRIGHT 
                    if _patch_index % self.n_patches == self.n_patches-1:
                        #options are 3,5,6
                        context_labels[_indx] =np.random.choice([3,5,6])
                        #[-1               ,       X        ]
                        #[+self.n_patches-1, +self.n_patches]

                    # TOPLEFT 
                    elif _patch_index % self.n_patches == 0:
                        #options are 4,6,7
                        context_labels[_indx] = np.random.choice([4,6,7]) 
                        #[      X        ,                 1]
                        #[+self.n_patches, +self.n_patches+1]

                    else: #TOPMIDDLE
                        #options are 3,4,5,6,7
                        context_labels[_indx] = np.random.choice([3,4,5,6,7])
                        #[-1               ,       X        ,                 1]
                        #[+self.n_patches-1, +self.n_patches, +self.n_patches+1]

                elif _patch_index >= self.n_patches**2 -self.n_patches:
                    # BOTTOMRIGHT 
                    if _patch_index % self.n_patches == self.n_patches-1:
                        #options are 0,1,3
                        context_labels[_indx] =  np.random.choice([0,1,3])
                        #[-self.n_patches-1, -self.n_patches]
                        #[-1               ,       X        ]

                    # BOTTOMLEFT 
                    elif _patch_index % self.n_patches == 0:
                        #options are 1,2,4
                        context_labels[_indx] =  np.random.choice([1,2,4])
                        #[-self.n_patches, -self.n_patches+1]
                        #[      X        ,                 1]

                    else: #BOTTOMMIDDLE
                        #options are 0,1,2,3,4
                        context_labels[_indx] = np.random.choice([0,1,2,3,4])
                        #[-self.n_patches-1, -self.n_patches, -self.n_patches+1]
                        #[-1               ,       X        ,                 1]

                elif _patch_index % self.n_patches == self.n_patches-1: #RIGHT
                    #options are 0,1,3,5,6
                    context_labels[_indx] = np.random.choice([0,1,3,5,6])
                    #[-self.n_patches-1, -self.n_patches]
                    #[-1               ,       X        ]
                    #[+self.n_patches-1, +self.n_patches]

                elif _patch_index % self.n_patches == 0: #LEFT
                    #options are 1,2,4,6,7
                    context_labels[_indx] = np.random.choice([1,2,4,6,7])
                    #[ -self.n_patches, -self.n_patches+1]
                    #[       X        ,                 1]
                    #[ +self.n_patches, +self.n_patches+1]
                
                else: #MIDDEL
                    #options are 0,1,2,3,4,5,6,7
                    context_labels[_indx] = np.random.choice([0,1,2,3,4,5,6,7])
                    #[-self.n_patches-1, -self.n_patches, -self.n_patches+1]
                    #[-1               ,       X        ,                 1]
                    #[+self.n_patches-1, +self.n_patches, +self.n_patches+1]
                context_images_pivot[_indx,:] = temp_patches[_patch_index]   
                context_images_neighbour[_indx,:] = temp_patches[_patch_index + _locations[context_labels[_indx]]]
                _indx +=1
        context_labels = torch.from_numpy(context_labels)
        context_images_pivot = torch.from_numpy(context_images_pivot)
        context_images_neighbour = torch.from_numpy(context_images_neighbour)

        return context_labels, context_images_pivot, context_images_neighbour

    def jitter(self, args:args) -> torch.tensor:
        """
            Creates a jittered pair for training  
            
            Parameters
            ----------
            args: cmd arugments

            Returns
            -------
            jittered_data: tensor of patches 

        """

        jittered_data = np.ones([self.data.shape[0],
                                 self.data.shape[1],
                                 args.patch_size, 
                                 args.patch_size],dtype='float32')
        _indx = 0

        for _image_indx in range(self.n_patches**2, self.data.shape[0]+1,self.n_patches**2):
            temp_patches = self.data[_image_indx-self.n_patches**2:_image_indx,...] #selects 1 image in patch form has dimensioons (64, 4, 32, 32)
            temp_data = reconstruct(temp_patches, args) # reconstruct to  shape (1,4,256,256)

            _max = 1#args.amount

            for _patch_index in range(self.n_patches**2):
                x_coor = args.patch_size*(_patch_index%self.n_patches)
                y_coor = args.patch_size*(_patch_index//self.n_patches)

                if _patch_index < self.n_patches:
                    # TOPRIGHT 
                    if _patch_index % self.n_patches == self.n_patches-1:
                        # x-jitter can only be negative 
                        x_coor = x_coor + np.random.randint(-_max, 0)
                        # y-jitter can only be positive
                        y_coor = y_coor + np.random.randint(0, _max)

                    # TOPLEFT 
                    elif _patch_index % self.n_patches == 0:
                        # x-jitter can only be positive 
                        x_coor = x_coor + np.random.randint(0, _max)
                        # y-jitter can only be positive
                        y_coor = y_coor + np.random.randint(0, _max)

                    else: #TOPMIDDLE
                        # x-jitter can be positive or negative 
                        x_coor = x_coor + np.random.randint(-_max, _max)
                        # y-jitter can only be positive
                        y_coor = y_coor + np.random.randint(0, _max)

                elif _patch_index >= self.n_patches**2 -self.n_patches:
                    # BOTTOMRIGHT 
                    if _patch_index % self.n_patches == self.n_patches-1:
                        # x-jitter can only be negative 
                        x_coor = x_coor + np.random.randint(-_max,0)
                        # y-jitter can only be negative 
                        y_coor = y_coor + np.random.randint( -_max,0)

                    # BOTTOMLEFT 
                    elif _patch_index % self.n_patches == 0:
                        # x-jitter can only be positive 
                        x_coor = x_coor + np.random.randint(0,_max)
                        # y-jitter can only be negative 
                        y_coor = y_coor + np.random.randint( -_max,0)

                    else: #BOTTOMMIDDLE
                        # x-jitter can be positive or negative 
                        x_coor = x_coor + np.random.randint(-_max,_max)
                        # y-jitter can only be negative 
                        y_coor = y_coor + np.random.randint( -_max,0)


                elif _patch_index % self.n_patches == self.n_patches-1: #RIGHT
                    # x-jitter can only be negative 
                    x_coor = x_coor + np.random.randint(-_max,0)
                    # y-jitter can be positive or negative 
                    y_coor = y_coor + np.random.randint( -_max,_max)

                elif _patch_index % self.n_patches == 0: #LEFT
                    # x-jitter can only be positive
                    x_coor = x_coor + np.random.randint(0,_max)
                    # y-jitter can be positive or negative 
                    y_coor = y_coor + np.random.randint( -_max,_max)
                
                else: #MIDDEL
                    # x-jitter can be positive or negative
                    x_coor = x_coor + np.random.randint(-_max,_max)
                    # y-jitter can be positive or negative 
                    y_coor = y_coor + np.random.randint( -_max,_max)

                jittered_data[_indx,:] = temp_data[0,
                                                   :,
                                                   y_coor:y_coor+args.patch_size,
                                                   x_coor:x_coor+args.patch_size]
                _indx +=1

        return torch.from_numpy(jittered_data)
