import numpy as np
import torch
from tqdm import tqdm
import torch 
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import h5py 

from utils import args
from utils.data import defaults
from utils.data.patches import reconstruct

def get_finetune_data(args, transform=None):
    hf = h5py.File(args.data_path,'r')

    (test_data, train_data,
    test_labels, train_labels,
    test_frequency_band, train_frequency_band,
    test_source, train_source) = train_test_split(_join(hf, 'data', args),
                                                  _join(hf, 'labels', args).astype(str),
                                                  _join(hf, 'frequency_band', args),
                                                  _join(hf, 'source', args).astype(str),
                                                  test_size=0.3,
                                                  random_state=args.seed)
    train_dataset = LOFARDataset(train_data,
                                 train_labels,
                                 train_frequency_band,
                                 train_source,
                                 args,
                                 transform=transform,
                                 roll=True,
                                 fine_tune=True)

    test_dataset =   LOFARDataset(test_data,
                                 test_labels,
                                 test_frequency_band,
                                 test_source,
                                 args,
                                 transform=None,
                                 fine_tune=True,
                                 test=True)

    return train_dataset, test_dataset

def get_data(args, transform=None):
    hf = h5py.File(args.data_path,'r')

    (train_data, val_data, 
    train_labels, val_labels, 
    train_frequency_band, val_frequency_band,
    train_source, val_source) = train_test_split(hf['train_data/data'][:],
                                                 hf['train_data/labels'][:].astype(str),
                                                 hf['train_data/frequency_band'][:],
                                                 hf['train_data/source'][:].astype(str),
                                                 test_size=0.05, 
                                                 random_state=args.seed)
    mask = np.random.choice(np.arange(len(train_labels)),
                             int(len(train_labels)*1.0))#args.percentage_data))
    train_dataset = LOFARDataset(train_data[mask], 
                                 train_labels[mask], 
                                 train_frequency_band[mask], 
                                 train_source[mask],
                                 args,
                                 transform=transform,
                                 roll=True)

    val_dataset =   LOFARDataset(val_data, 
                                 val_labels, 
                                 val_frequency_band, 
                                 val_source,
                                 args,
                                 transform=None)

    test_dataset = LOFARDataset(_join(hf, 'data', args),
                                _join(hf, 'labels', args).astype(str),
                                _join(hf, 'frequency_band', args),
                                _join(hf, 'source', args).astype(str),
                                args,
                                transform=None,
                                test=True)

    return train_dataset, val_dataset, test_dataset

def _join(hf:h5py.File, field:str,args)->np.array:
    """
        Joins together the normal and anomalous testing data
        
        Parameters
        ----------
        hf: h5py dataset 
        field: the field that is meant to be concatenated along 

        Returns
        -------
        data: concatenated array

    """
    data = hf['test_data/{}'.format(field)][:]
    for a in defaults.anomalies:
        if a != 'all': 
            _data = hf['anomaly_data/{}/{}'.format(a,field)][:]
            if args.percentage_data > len(_data):
                a = len(_data)
            else:
                a= int(args.percentage_data)
            mask = np.random.choice(np.arange(len(_data)),a)
            data = np.concatenate([data,
                                   _data],axis=0)
    return data

class LOFARDataset(Dataset):
    def __init__(self, 
            data:np.array, 
            labels:np.array, 
            frequency_band:np.array, 
            source:np.array, 
            args:args, 
            transform=None,
            roll=False,
            fine_tune=False,
            test=False):# set default types

        self.test = test

        if roll:
            _data, _frequency_band = self.circular_shift(data, 
                                                       frequency_band, 
                                                       args.patch_size//2)
            data = np.concatenate([data, _data],axis=0)
            frequency_band= np.concatenate([frequency_band, _frequency_band],axis=0)
            labels = np.concatenate([labels, labels],axis=0)
            source = np.concatenate([source, source],axis=0)
            

        self.args = args
        self.anomaly_mask = []
        self.n_patches = int(defaults.SIZE[0]/args.patch_size)
        self._source = np.repeat(source, self.n_patches**2, axis=0)

        if fine_tune: 
            self._labels = self.encode_labels(labels)
            self._labels= np.repeat(self._labels, self.n_patches**2, axis=0)
            self._labels = torch.from_numpy(self._labels)
        else:
            self._labels= np.repeat(labels, self.n_patches**2, axis=0)


        data = self.normalise(data)
        self._data = torch.from_numpy(data).permute(0,3,1,2)
        self._data = self.patch(self._data)

        self._frequency_band = torch.from_numpy(frequency_band).permute(0,3,1,2)
        self._frequency_band = self.patch(self._frequency_band)[:,0,0,[0,-1]]#use start, end frequencies per patch
        self._frequency_band = torch.from_numpy(self.encode_frequencies(self._frequency_band.numpy()))

        (self._context_labels, 
         self._context_images_neighbour,
         self._context_frequency_neighbour) = self.context_prediction(args)

        self.transform=transform 
        self.set_anomaly_mask('all')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datum = self.data[idx,...]
        label = self.labels[idx]
        frequency = self.frequency_band[idx] 
        context_label = self.context_labels[idx]
        context_image_neighbour = self.context_images_neighbour[idx]
        context_frequency_neighbour = self.context_frequency_neighbour[idx]

        if self.transform:
            datum = self.transform(datum)

        return datum, label, frequency, context_label, context_image_neighbour, context_frequency_neighbour

    def set_percentage_contam(self, anomaly:str)->np.array:
        """
            Sets the contamination per anomaly class
            
            Parameters
            ----------
            anomaly: anomaly class for mask 

            Returns
            -------
            None 
        """
        assert anomaly in defaults.anomalies or anomaly =='all', "Anomaly not found"

        if self.test:
            _normal_samples = [i for i,l in enumerate(self._labels) if l == '' ]
            mask = np.zeros(self._labels.shape, dtype='bool')
            mask[_normal_samples] = True

            if anomaly == 'all':
                _sum = len(_normal_samples)
                for a in defaults.anomalies:
                    if a == 'all': continue
                    temp_indx = [i for i,l in enumerate(self._labels) if l==a]
                    temp_mask = np.random.choice(temp_indx, int(len(_normal_samples)*defaults.percentage_comtamination[a]), replace=False)
                    print(temp_mask)
                    print(temp_mask.dtype)
                    mask[temp_mask] = True 
            else:
                temp_indx = [i for i,l in enumerate(self._labels) if anomaly in l]

                temp_mask = np.random.choice(temp_indx,
                                        int(len(_normal_samples)*
                                            defaults.percentage_comtamination[anomaly]),
                                        replace=False)
                mask[temp_mask] = True
        else:
            mask = [True]*len(self._data)

        return mask

    def set_anomaly_mask(self, anomaly:str):
        """
            Sets the mask for the dataloader to load only specific classes 
            
            Parameters
            ----------
            anomaly: anomaly class for mask 

            Returns
            -------
            None 
        """

        assert anomaly in defaults.anomalies or anomaly =='all', "Anomaly not found"
        if anomaly == 'all':
            self.anomaly_mask = [True]*len(self._data)
        else:
            self.anomaly_mask = [((anomaly in l) | (l == '')) for l in self._labels]

        #self.anomaly_mask = self.set_percentage_contam(anomaly)

        self.data = self._data[self.anomaly_mask]
        self.labels = self._labels[self.anomaly_mask]
        self.frequency_band = self._frequency_band[self.anomaly_mask]
        self.source = self._source[self.anomaly_mask]
        self.context_labels = self._context_labels[self.anomaly_mask]
        self.context_images_neighbour = self._context_images_neighbour[self.anomaly_mask]
        self.context_frequency_neighbour= self._context_frequency_neighbour[self.anomaly_mask]

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

    def encode_labels(sef, labels):
        _labels = []
        for label in labels:
            if label =='':
                _labels.append(len(defaults.anomalies))
            else:
                _labels.append([i for i,a in enumerate(defaults.anomalies) if a in label][0])
        return _labels

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
            encoded_frequencies.append(0)#np.where(np.array(defaults.frequency_bands[self.args.patch_size]) == _temp)[0][0])

        return np.array(encoded_frequencies)

    def normalise(self, data:np.array)->np.array:
        """
            perpendicular polarisation normalisation

        """
        _data = np.zeros(data.shape)
        for i, spec in enumerate(data):
            for pol in range(data.shape[-1]):
                _min, _max = np.percentile(spec[...,pol], [self.args.amount,100-self.args.amount])
                temp = np.clip(spec[...,pol],_min, _max)
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

    def context_prediction(self, args:args) -> (torch.tensor, torch.tensor, torch.tensor):
        """
            Arranges a context prediction dataset
            
            Parameters
            ----------
            args: cmd arugments

            Returns
            -------
            patches: tensor of patches reshaped to ((h/size)*(w/size),C,h/size, w/size)

        """

        context_labels = np.ones([self._data.shape[0]],dtype='int')
        context_images_neighbour = np.zeros([self._data.shape[0],
                                             self._data.shape[1],
                                             args.patch_size, 
                                             args.patch_size],dtype='float32')
        context_frequency_neighbour = np.zeros([self._frequency_band.shape[0]],dtype='int')
        _indx = 0
        _locations = [-self.n_patches-1, -self.n_patches, -self.n_patches+1, -1, +1, +self.n_patches-1, +self.n_patches, +self.n_patches+1]

        for _image_indx in range(self.n_patches**2, self._data.shape[0]+1,self.n_patches**2):
            temp_patches = self._data[_image_indx-self.n_patches**2:_image_indx,...] #selects 1 image in patch form has dimensioons (64, 4, 32, 32)
            temp_freq = {0:0, 1:1, 2:2, 
                         3:0,      4:2, 
                         5:0, 6:1, 7:2}

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
                context_images_neighbour[_indx,:] = temp_patches[_patch_index + _locations[context_labels[_indx]]]
                context_frequency_neighbour[_indx] = temp_freq[context_labels[_indx]]
                _indx +=1
        context_labels = torch.from_numpy(context_labels)
        context_images_neighbour = torch.from_numpy(context_images_neighbour)
        context_frequency_neighbour= torch.from_numpy(context_frequency_neighbour)

        return context_labels, context_images_neighbour, context_frequency_neighbour

