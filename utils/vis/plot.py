import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os
from tqdm import tqdm

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

def io(n_plots:int,
       inputs:np.array, 
       reconstructions:np.array,
       path:str, 
       epoch:int, 
       neighbours:int=0, 
       anomaly:str='')->None:
    """
        Plots input and reconstructions
    """
    fig,axs = plt.subplots(n_plots,2,figsize=(5,10));
    for i in range(n_plots):
        r = np.random.randint(len(inputs))
        axs[i,0].imshow(inputs[r,0,...], aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        axs[i,1].imshow(reconstructions[r,0,...], aspect='auto', interpolation='nearest', vmin=0, vmax=1)
   
    _dir = '{}/training_outputs'.format(path)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    plt.savefig('{}/{}_epoch_output_{}_{}'.format(_dir, anomaly, epoch, neighbours),dpi=98)
    plt.close(fig)


def nln_io(n_plots:int,
        x_test:np.array, 
        neighbours:np.array,
        labels:np.array,
        D:np.array,
        path:str, 
        epoch:int, 
        anomaly:str='')->None:
    """
        Plots input and neighbours
    """
    _dir = '{}/training_outputs/{}'.format(path,anomaly)
    if not os.path.exists(_dir):
        os.makedirs(_dir)

    for _ in tqdm(range(10)):
        i = np.random.randint(len(x_test))

        while anomaly not in labels[i]:
            i = np.random.randint(len(x_test))

        fig,axs = plt.subplots(2,neighbours.shape[0]+1,figsize=(10,5));
        axs[0][0].imshow(x_test[i,0,...], aspect='auto', interpolation='nearest', vmin=0,vmax=1)
        axs[0][0].set_title("Input", fontsize=5)
        axs[0][0].axis('off')
        axs[1][0].axis('off')
        for n in range(neighbours.shape[0]):
            axs[0][n+1].imshow(neighbours[n,i,0,...], aspect='auto', interpolation='nearest', vmin=0,vmax=1)
            axs[0][n+1].set_title("{}".format(labels[i]), fontsize=5)
            axs[0][n+1].axis('off')
            axs[1][n+1].imshow(D[n,i,...], aspect='auto', interpolation='nearest', vmin=0,vmax=1)
            axs[1][n+1].set_title("{}".format(round(np.mean(D[n,i,...]),3)), fontsize=5)
            axs[1][n+1].axis('off')

        plt.savefig('{}/{}_{}'.format(_dir, anomaly, i),dpi=96)
        plt.close('all')


def loss_curve(path:str,epoch:int, **kwargs)->None:
    plt.title("Train-Validation Accuracy")
    for _name, _loss in kwargs.items():
        plt.plot(np.arange(epoch),_loss, label=_name)

    plt.legend()
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.savefig('{}/loss'.format(path),dpi=99)
    plt.close('all')
