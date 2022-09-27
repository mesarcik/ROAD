import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os

def imscatter(z:np.array, inputs:np.array, path:str, epoch:int, ax=None, zoom=1)->None:
    """
        code adapted from: https://gist.github.com/feeblefruits/20e7f98a4c6a47075c8bfce7c06749c2
    """
    fig,ax = plt.subplots(1,1,figsize=(10,10));
    for x, y, image in zip(z[:,0], z[:,1], inputs[:,0,...]):
        im = OffsetImage(image, zoom=zoom)
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    _dir = '{}/training_outputs'.format(path)
    if not os.path.exists(_dir):
        os.makedirs(_dir)

    plt.savefig('{}/epoch_embedding_{}'.format(_dir,epoch),dpi=98)
    plt.close('all')

def io(n_plots:int,inputs:np.array, reconstructions:np.array,path:str, epoch:int)->None:
    """
        Plots input and reconstructions
    """
    fig,axs = plt.subplots(n_plots,2,figsize=(5,10));
    for i in range(n_plots):
        r = np.random.randint(len(inputs))
        axs[i,0].imshow(inputs[r,0,...], aspect='auto', interpolation='nearest')#, vmin=0, vmax=1)
        axs[i,1].imshow(reconstructions[r,0,...], aspect='auto', interpolation='nearest')#, vmin=0, vmax=1)
   
    _dir = '{}/training_outputs'.format(path)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    plt.savefig('{}/epoch_output_{}'.format(_dir,epoch),dpi=98)
    plt.close(fig)

def loss_curve(path:str, **kwargs)->None:
    plt.title("Train-Validation Accuracy")
    for _name, _loss in kwargs.items():
        plt.plot(_loss, label=_name)

    plt.legend()
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.savefig('{}/loss'.format(path),dpi=98)
