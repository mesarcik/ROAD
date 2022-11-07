import torch
from torch import nn
import torch.nn.functional as F

import math

class PositionClassifier(nn.Module):
    def __init__(self, in_dims:int, out_dims:int, class_num=8):
        super().__init__()

        self.in_dims = in_dims
        modules  = []
        modules.append(nn.Linear(in_dims, 128))
        modules.append(nn.LeakyReLU(0.1))

        modules.append(nn.Linear(128, 64))
        modules.append(nn.LeakyReLU(0.1))

        modules.append(nn.Linear(64, out_dims))
        modules.append(nn.BatchNorm1d(num_features=out_dims))

        self.model = nn.Sequential(*modules)
        self.loss_fn = nn.CrossEntropyLoss()

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
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
        z = self.model(input)
        return z
