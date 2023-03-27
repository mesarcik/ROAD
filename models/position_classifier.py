import torch
from torch import nn
import torch.nn.functional as F

import math

class PositionClassifier(nn.Module):
    def __init__(self, 
            latent_dim:int,
            out_dims:int):

        super().__init__()

        self.latent_dim = latent_dim 
        self.out_dims = out_dims

        modules  = []
        modules.append(nn.Linear(self.latent_dim*2, 64))
        modules.append(nn.LeakyReLU(0.1))

        modules.append(nn.Linear(64, self.out_dims))
        modules.append(nn.BatchNorm1d(num_features=self.out_dims))

        self.model = nn.Sequential(*modules)
        self.loss_fn = nn.CrossEntropyLoss()

    def save(self, args):
        fpath = self.fpath_from_name(args)
        torch.save(self.state_dict(), fpath)

    def load(self, args):
        fpath = self.fpath_from_name(args)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self,args)->str:
        return f'outputs/models/{args.model_name}/position_classifier_{args.ood}_{args.seed}_{args.pretrain}.pkl'

    def forward(self, 
                z_0: torch.tensor,
                z_1: torch.tensor,
                **kwargs):
        z = torch.cat([z_0, z_1], axis=1)
        c = self.model(z)
        return c

    def loss_function(self,
                      pred:torch.tensor,
                      labels:torch.tensor) -> torch.tensor:
        loss = self.loss_fn(pred, labels)

        return loss

