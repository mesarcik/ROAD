import torch
from torchvision import models
from torch import nn
import os

class ClassificationHead(nn.Module):
    def __init__(self, 
            out_dims: int, 
            latent_dim: int, 
            **kwargs) -> None:
        super(ClassificationHead, self).__init__()
        self.out_dims =  out_dims
        self.latent_dim = latent_dim

        # classifier  
        modules  = []
        modules.append(nn.Linear(16*self.latent_dim, 8)) # 16 because 64x64 patches
        modules.append(nn.LeakyReLU())

        modules.append(nn.Linear(8, out_dims))
        modules.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*modules)

        self.loss_fn = nn.CrossEntropyLoss()

    def save(self, 
            model:str,
            name:str, 
            ood_class:int, 
            seed:int, 
            pretrain:bool):
        fpath = self.fpath_from_name(model, 
                name, 
                ood_class, 
                seed, 
                pretrain)
        torch.save(self.state_dict(), fpath)

    def load(self, 
            model:str,
            name:str, 
            ood_class:int, 
            seed:int, 
            pretrain:bool):
        fpath = self.fpath_from_name(model,
                name, 
                seed, 
                ood_class, 
                pretrain)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self, 
            model:str,
            name:str, 
            seed:int, 
            ood_class:int, 
            pretrain:bool)->str:
        return f'outputs/{model}/{name}/classification_head_{ood_class}_{seed}_{pretrain}.pkl'

    def forward(self,
            _input: torch.tensor)->tensor:
        c = self.classifier(_input)
        return c

    def loss_function(self,
                      c:torch.tensor, 
                      labels:torch.tensor, 
                      **kwargs) -> dict:
        """
        Computes the BCE loss function
        """
        return {"loss": self.loss_fn(c, labels)}
