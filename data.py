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

def get_data(args, remove=None, transform=None):
    """
        Constructs datasets and loaders for training, validation and testing
        Test data for supervised and unsupervised must be the same

    """
    hf = h5py.File(args.data_path,'r')
    test_indexes, train_indexes = train_test_split(np.arange(len(_join(hf, 'labels').astype(str))),
                                                   test_size=0.5,
                                                   random_state=args.seed)

    _data = _join(hf, 'data')[train_indexes]
    _labels = _join(hf, 'labels').astype(str)[train_indexes]
    _frequency_band = _join(hf, 'frequency_band')[train_indexes]
    _source = _join(hf, 'source').astype(str)[train_indexes]

    (train_data, val_data, 
    train_labels, val_labels, 
    train_frequency_band, val_frequency_band,
    train_source, val_source) = train_test_split(_data,
                                                 _labels,
                                                 _frequency_band, 
                                                 _source,
                                                 test_size=0.02, 
                                                 random_state=args.seed)

    supervised_train_dataset = LOFARDataset(train_data, 
                                 train_labels, 
                                 train_frequency_band, 
                                 train_source,
                                 args,
                                 test=False,
                                 transform=transform,
                                 remove=remove,
                                 roll=False,
                                 supervised=True)

    supervised_val_dataset =   LOFARDataset(val_data, 
                                 val_labels, 
                                 val_frequency_band, 
                                 val_source,
                                 args,
                                 test=False,
                                 transform=None,
                                 remove=remove,
                                 supervised=True)

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
                                 args,
                                 test=False,
                                 transform=transform,
                                 roll=False,
                                 remove=None,
                                 supervised=False)

    val_dataset =   LOFARDataset(val_data, 
                                 val_labels, 
                                 val_frequency_band, 
                                 val_source,
                                 args,
                                 test=False,
                                 transform=None,
                                 remove=None,
                                 supervised=False)

    test_dataset = LOFARDataset(_join(hf, 'data')[test_indexes],
                                _join(hf, 'labels').astype(str)[test_indexes],
                                _join(hf, 'frequency_band')[test_indexes],
                                _join(hf, 'source').astype(str)[test_indexes],
                                args,
                                test=True,
                                transform=None,
                                remove=None,
                                supervised=False)
                                

    return (train_dataset, 
            val_dataset, 
            test_dataset, 
            supervised_train_dataset, 
            supervised_val_dataset)

def _join(hf:h5py.File, field:str, compound:bool=False)->np.array:
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
            labels = hf['anomaly_data/{}/labels'.format(a)][:].astype(str)
            if not compound: 
                mask = [l == a for l in labels]
            else: mask = [True for l in labels]
            _data = hf['anomaly_data/{}/{}'.format(a,field)][:][mask]
            data = np.concatenate([data,_data],axis=0)
    return data

class LOFARDataset(Dataset):
    def __init__(self, 
            data:np.array, 
            labels:np.array, 
            frequency_band:np.array, 
            source:np.array, 
            args:args, 
            test:bool,
            transform=None,
            remove=None,
            roll=False,
            supervised=False):# set default types

        self.supervised = supervised
        self.test = test
        self.test_seed=42
        self.remove=remove

        if roll:
            _data, _frequency_band = self.circular_shift(data, 
                                                       frequency_band, 
                                                       args.patch_size//2)
            data = np.concatenate([data, _data],axis=0)
            frequency_band= np.concatenate([frequency_band, _frequency_band],axis=0)
            labels = np.concatenate([labels, labels],axis=0)
            source = np.concatenate([source, source],axis=0)
            
        if remove is not None:
            mask = labels!=remove
            labels = labels[mask]
            source = source[mask]
            data = data[mask]
            frequency_band = frequency_band[mask]

        self.args = args
        self.anomaly_mask = []
        self.original_anomaly_mask = []
        self.n_patches = int(defaults.SIZE[0]/args.patch_size)
        
        self._labels = torch.from_numpy(self.encode_labels(labels))
        self._source = source
        self._data = torch.from_numpy(self.normalise(data)).permute(0,3,1,2)
        self._frequency_band = torch.from_numpy(frequency_band).permute(0,3,1,2)

        self.transform=transform 
        self.set_anomaly_mask(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.supervised:
            datum = self.data[idx]
            label = self.labels[idx]
            source = self.source[idx]
            return datum, label, source

        else:
            label = np.repeat(self.labels[idx], self.n_patches**2, axis=0)
            datum =  self.patch(self.data[idx:idx+1,...])
            frequency = np.random.random(len(label))#self.patch(self._frequency_band[idx])[0,0,[0,-1]]#TODO fix frequncy encoding

            (context_label, 
             context_image_neighbour,
             context_frequency_neighbour) = self.context_prediction(datum)

            if self.transform:
                datum = self.transform(datum)

            return datum, label, frequency, context_label, context_image_neighbour, context_frequency_neighbour


    def set_supervision(self, supervised:bool)->None:
        """
            sets supervision flag
        """
        self.supervised = supervised

    def remove_sources(self, remove):
        """
            removes data corresponding source
        """
        _, _, indxs = np.intersect1d(remove, self._source, assume_unique=True, return_indices=True)
        mask = [i not in indxs for i in range(len(self._source))]
        self._data = self._data[mask]
        self._labels = self._labels[mask]
        self._frequency_band = self._frequency_band[mask]
        self._source = self._source[mask]
        self.set_anomaly_mask(-1)
        


    def set_seed(self, seed:int)->None:
        """
            sets test data seed 
        """
        self.test_seed = seed 

    def set_anomaly_mask(self, anomaly:int):
        """
            Sets the mask for the dataloader to load only specific classes 
            
            Parameters
            ----------
            anomaly: anomaly class for mask, -1 for all 

            Returns
            -------
            None 
        """

        assert anomaly in np.arange(len(defaults.anomalies)) or anomaly == -1, "Anomaly not found"

        #if self.test:
        #    subsample_mask = self.subsample(self._labels)
        #else:
        subsample_mask = [True]*len(self._data)

        self.data = self._data[subsample_mask]
        self.labels = self._labels[subsample_mask]
        self.frequency_band = self._frequency_band[subsample_mask]
        self.source = self._source[subsample_mask]


        if anomaly == -1:
            self.anomaly_mask = [True]*len(self.data)
        else:
            self.anomaly_mask = [((anomaly == l) | (l ==len(defaults.anomalies))) for l in self.labels]

        self.data = self.data[self.anomaly_mask]
        self.labels = self.labels[self.anomaly_mask]
        self.frequency_band = self.frequency_band[self.anomaly_mask]
        self.source = self.source[self.anomaly_mask]

    def subsample(self, labels:np.array)->np.array:
        """
            Subsamples dataset to enforce percentage_comtamination
            
            Parameters
            ----------
            labels: numpy array containing labels 
            seed: random seed for sampling 

            Returns
            -------
            mask
        """

        np.random.seed(self.test_seed)
        mask = np.array([],dtype=int)
        _len_ = len(labels[labels==len(defaults.anomalies)])
        for i, a in enumerate(defaults.percentage_comtamination):
            _amount = int(_len_*defaults.percentage_comtamination[a])
            _indices = [j for j, x in enumerate(labels) if x == i]
            _indices = np.random.choice(_indices, _amount, replace=False)
            mask = np.concatenate([mask, _indices],axis=0)
        mask = np.concatenate([mask, [i for i, x in enumerate(labels) if x == len(defaults.anomalies)]],axis=0)

        return mask.astype(int)

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
            r = max_roll#np.random.randint(-max_roll,max_roll)
            _data[i,:] = np.roll(data[i,:], r, axis =1)
            _data[i,:] = np.roll(_data[i,:], r, axis =0)
            _frequency_band[i,:] = np.roll(frequency_band[i,:], r, axis =1)
            _frequency_band[i,:] = np.roll(_frequency_band[i,:], r, axis =0)
        
        return _data, _frequency_band

    def encode_labels(self, labels:np.array)->np.array:
        out = []
        for label in labels:
            if label =='':
                out.append(len(defaults.anomalies))
            else:
                out.append([i for i,a in enumerate(defaults.anomalies) if a in label][0])
        return np.array(out)

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

    def context_prediction(self, data:torch.tensor) -> (torch.tensor, torch.tensor, torch.tensor):
        """
            Arranges a context prediction dataset
            
            Parameters
            ----------
            data: indexed patched data

            Returns
            -------
            patches: tensor of patches reshaped to ((h/size)*(w/size),C,h/size, w/size)

        """

        context_labels = np.ones([data.shape[0]],dtype='int')
        context_images_neighbour = np.zeros([data.shape[0],
                                             data.shape[1],
                                             self.args.patch_size, 
                                             self.args.patch_size],dtype='float32')
        context_frequency_neighbour = np.zeros([data.shape[0]],dtype='int')
        _indx = 0
        _locations = [-self.n_patches-1, -self.n_patches, -self.n_patches+1, -1, +1, +self.n_patches-1, +self.n_patches, +self.n_patches+1]

        for _image_indx in range(self.n_patches**2, data.shape[0]+1,self.n_patches**2):
            temp_patches = data[_image_indx-self.n_patches**2:_image_indx,...] #selects 1 image in patch form has dimensioons (64, 4, 32, 32)
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

