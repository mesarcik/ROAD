import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os

class Decoder(nn.Module):
    def __init__(self, out_channels:int, patch_size:int, latent_dim:int):
        super().__init__()

        self.out_channels = out_channels
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.n_layers =  3#int(np.log2(self.patch_size**2//self.patch_size))
        #self.hidden_dims  = [self.patch_size*2**i for i in range(self.n_layers)]
        self.hidden_dims  = [2**(2+i) for i in range(self.n_layers)]
        modules  = []

        self.intermediate = self.patch_size//2**self.n_layers
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[0]*(self.intermediate**2))
        #self.decoder_input = nn.Linear(self.latent_dim, self.patch_size)

        
        for i in range(self.n_layers - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        modules.append(nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.BatchNorm2d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(self.hidden_dims[-1], out_channels= self.out_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

        self.loss_fn = nn.CrossEntropyLoss()

    def save(self, name):
        fpath = self.fpath_from_name(name)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self, name):
        return f'ckpts/{name}/position_classifier.pkl'

    def loss_function(self,
                      pred:torch.tensor,
                      labels:torch.tensor) -> dict:
        loss = self.loss_fn(pred, labels)

        return {'loss':loss}

    def forward(self, input:torch.tensor):

        x = self.decoder_input(input)
        x = x.view(-1, self.hidden_dims[0], self.intermediate, self.intermediate)
        x = self.decoder(x)
        return x
