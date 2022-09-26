import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

from data import LOFARDataset
from models import VAE
from sklearn.manifold import TSNE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 2**10
PS=64
LD=100

# define dataset/dataloader
#test_transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.RandomCrop((128,128)),
#                                 transforms.Normalize(DATA_MEANS, DATA_STD)
#                                 ])

# For training, we add some augmentation. Networks are too powerful and would overfit.
#train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                      #transforms.RandomRotation([0, 180]),
#                                      transforms.RandomCrop((128,128)),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(DATA_MEANS, DATA_STD)
#                                     ])

def imscatter(x, y, image, ax=None, zoom=1):
    """
        code adapted from: https://gist.github.com/feeblefruits/20e7f98a4c6a47075c8bfce7c06749c2
    """
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

# make dataset
#TODO 
train_dataset = LOFARDataset('/home/mmesarcik/data/LOFAR/compressed/organised/LOFAR_AD_dataset_21-09-22.pkl',
                             PS,
                             )
#valid_dataset = HeraDataset('/home/mmesarcik/HERA_ResNet/HERA_val.pkl',test_transform) #test transforms are applied
#test_dataset =  HeraDataset('/home/mmesarcik/HERA_ResNet/HERA_test.pkl',test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
#valid_dataloader = DataLoader(valid_dataset, batch_size=BS, shuffle=True)
#test_dataloader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

# define model 
vae = VAE(in_channels=4, latent_dim=LD,patch_size=PS, hidden_dims=[32,64,128,256]).to(device)
vae.to(device)

optimizer = torch.optim.Adam(vae.parameters(),lr=0.001)#, lr=0.0001, momentum=0.9)

n_epochs = 100

valid_loss_min = np.Inf
train_loss = []
train_acc = []
total_step = len(train_dataloader)
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    #print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_dataloader):
        data_ = data_.float().to(device)
        optimizer.zero_grad()

        [_decoded, _input, _mu, _log_var] = vae(data_)
        z = vae.reparameterize(_mu, _log_var)
        loss = vae.loss_function(_decoded, _input, _mu, _log_var)
        loss['loss'].backward()
        optimizer.step()

        running_loss += loss['Reconstruction_Loss'].item()
        if (batch_idx) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch, n_epochs, batch_idx, total_step, loss['Reconstruction_Loss'].item()))
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}')
    batch_loss = 0

    if True:
        valid_loss_min = batch_loss
        torch.save(vae.state_dict(), 'outputs/vae.pt')
        print('Improvement-Detected, save-model')
        vae.eval()
        mu, log_var = vae.encode(data_)
        Z = vae.reparameterize(mu, log_var).cpu().detach().numpy()
        z = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(Z)
        
        imgs = data_.cpu().detach().numpy()
        xhats = _decoded.cpu().detach().numpy()
        fig,axs = plt.subplots(10,2,figsize=(5,10));
        for i in range(10):
            r = np.random.randint(len(imgs))
            axs[i,0].imshow(imgs[r,0,...], aspect='auto', interpolation='nearest')#, vmin=0, vmax=1)
            axs[i,1].imshow(xhats[r,0,...], aspect='auto', interpolation='nearest')#, vmin=0, vmax=1)
        plt.savefig('outputs/embeddings/epoch_output_{}'.format(epoch),dpi=98)
        plt.close(fig)

        fig,ax = plt.subplots(1,1,figsize=(10,10));
        for x, y, image_path in zip(z[:,0], z[:,1], imgs[:,0,...]):
            imscatter(x, y, image_path, zoom=0.7, ax=ax)
        plt.savefig('outputs/embeddings/epoch_embedding_{}'.format(epoch),dpi=98)
        plt.close('all')
    vae.train()

    plt.title("Train-Validation Accuracy")
    plt.plot(train_loss, label='train')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.savefig('/tmp/temp',dpi=300)
