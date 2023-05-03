import _pickle as cPickle
import numpy as np
import h5py
import random 
from tqdm import tqdm
from utils.data.defaults import anomalies, SIZE 
import cv2
import copy
from datetime import datetime

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
    
    if args:
        for a in args:
            temp_args = []
            for i in indx:
                temp_args.append(a[i])
            _args.append(temp_args)
        return _data,*_args
    return _data

def reshape(data: list, frequency_band:list, dim: list, verbose:bool=False) -> np.array:
    """
        Resamples data and frequency_band to be of size dim
        
        Parameters
        ----------
        data: list of data
        frequency_band: list of frequency per channel
        verbose: prints tqdm output

        Returns
        -------
        _data: np.array of baselines with the same dimensions
        _freq: np.array of baselines with the same dimensions

    """
    _data = np.zeros((len(data), dim[0], dim[1], 4))
    _freq = np.zeros((len(frequency_band), dim[0], dim[1], 1))

    for i,d in enumerate(tqdm(data,disable=not verbose)):
        temp_freq = np.vstack([frequency_band[i]]*len(frequency_band[i]))
        _freq[i,...,0] = cv2.resize(temp_freq, dim)

        for p in range(4):
            _data[i,...,p] = cv2.resize(d[...,p], dim)

    return _data, _freq

def create_dataset()->None:
    """
        Reads in pickle generates properly formatted hdf5
        
        Parameters
        ----------

        Returns
        -------

    """
    hf = h5py.File('/data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_{}.h5'.format(datetime.now().strftime('%d-%m-%y')), 'w')
    train_group = hf.create_group('train_data')
    test_group = hf.create_group('test_data')
    anomalies_group = hf.create_group('anomaly_data')

    with open('/data/mmesarcik/LOFAR/LOFAR_AD/LOFAR_AD_dataset_05-04-23.pkl', 'rb') as f:
        data, labels, source, ids, frequency_band, flags = cPickle.load(f,encoding='latin')
  
    data, labels, source, ids, frequency_band  = remove_singles(data, 
                                                                labels, 
                                                                source,
                                                                ids,
                                                                frequency_band)


    data, frequency_band = reshape(data, frequency_band, SIZE) 

    labels = [('-').join(l) for l in labels]

    # separate test_data and train_data
    # train_data has label of ''

    train_inds = [i for i,l in enumerate(labels) if l =='']

    train_data = np.array([data[i] for i in  train_inds],dtype='float32')
    train_labels = np.array([labels[i] for i in train_inds],dtype='S100')
    train_source = np.array([source[i] for i in train_inds],dtype='S100')
    train_ids = np.array([ids[i] for i in train_inds],dtype='S100')
    train_frequency = np.array([frequency_band[i] for i in train_inds], dtype='float32')

    # Randomly sample non-anomalous data for train/test split
    reserved_samples = 1000
    mask = np.zeros(len(train_inds))
    mask[:reserved_samples] = 1
    np.random.shuffle(mask)
    train_mask,test_mask = mask==0, mask==1

    train_group.create_dataset('data',data=train_data[train_mask], compression='gzip', compression_opts=2)
    train_group.create_dataset('labels',data=train_labels[train_mask], compression='gzip', compression_opts=2)
    train_group.create_dataset('source',data=train_source[train_mask], compression='gzip', compression_opts=2)
    train_group.create_dataset('ids',data=train_ids[train_mask], compression='gzip', compression_opts=2)
    train_group.create_dataset('frequency_band',data=train_frequency[train_mask], compression='gzip', compression_opts=2)

    test_group.create_dataset('data',data=np.array(train_data[test_mask], dtype='float32'), compression='gzip', compression_opts=2)
    test_group.create_dataset('labels',data=np.array(train_labels[test_mask],dtype='S100'), compression='gzip', compression_opts=2)
    test_group.create_dataset('source',data=np.array(train_source[test_mask],dtype='S100'), compression='gzip', compression_opts=2)
    test_group.create_dataset('ids',data=np.array(train_ids[test_mask],dtype='S100'), compression='gzip', compression_opts=2)
    test_group.create_dataset('frequency_band',data=np.array(train_frequency[test_mask],dtype='float32'), compression='gzip', compression_opts=2)
    
    for anomaly in tqdm(anomalies):
        anomaly_group = anomalies_group.create_group(anomaly)
        # do some sanitation and remove poorly labelled classes remove unknown classes, scintillation, high_noise_elements
        test_inds = [i for i,l in enumerate(labels) if ((anomaly in l) and 
                                                        ('unknown' not in l) and 
                                                        ('scintillation' not in l) and 
                                                        ('real_high_noise' not in l) and 
                                                        (('ionosphere' not in l) or ('_ionosphere' in l)) and 
                                                        #('electric_fence' not in l) and 
                                                        ('high_noise_elements' not in l))]              
        print(anomaly, len(test_inds))

        test_data = np.array([data[i] for i in  test_inds],dtype='float32')
        test_labels = np.array([labels[i] for i in test_inds],dtype='S100')
        test_source = np.array([source[i] for i in test_inds],dtype='S100')
        test_ids = np.array([ids[i] for i in test_inds],dtype='S100')
        test_frequency = np.array([frequency_band[i] for i in test_inds],dtype='float32')

        anomaly_group.create_dataset('data',data=test_data, compression='gzip', compression_opts=2)
        anomaly_group.create_dataset('labels',data=test_labels, compression='gzip', compression_opts=2)
        anomaly_group.create_dataset('source',data=test_source, compression='gzip', compression_opts=2)
        anomaly_group.create_dataset('ids',data=test_ids, compression='gzip', compression_opts=2)
        anomaly_group.create_dataset('frequency_band',data=test_frequency, compression='gzip', compression_opts=2)

    hf.close()

if __name__ == '__main__':
    create_dataset()
