import aoflagger as aof
import numpy as np 
from tqdm import tqdm

def flag_data(data:np.array, starting_threshold:int=20)->np.array:
    """
        Applies AOFlagger to visibilities

        Parameters
        ----------
        data (np.array) spectrograms

        Returns
        -------
        np.array, np.array
    """
    mask = np.empty(data[...,0].shape, dtype=np.bool)

    aoflagger = aof.AOFlagger()
    strategy = aoflagger.load_strategy_file('utils/flagging/stratergies/lofar-default-{}.lua'.format(starting_threshold))

    # LOAD data into AOFlagger structure
    for indx in range(len(data)):
        _data = aoflagger.make_image_set(data.shape[1], data.shape[2], 4)
        for pol in range(4):
            _data.set_image_buffer(pol, data[indx,...,pol]) # Real values 

        flags = strategy.run(_data)
        flag_mask = flags.get_buffer()
        mask[indx,...] = flag_mask
    
    return mask.astype('bool')

