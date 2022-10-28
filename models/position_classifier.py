import torch
from torch import nn
import torch.nn.functional as F

import math

class NormalizedLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(NormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        with torch.no_grad():
            w = self.weight / self.weight.data.norm(keepdim=True, dim=0)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class PositionClassifier(nn.Module):
    def __init__(self, in_dims:int, out_dims:int, class_num=8):
        super().__init__()

        self.in_dims = in_dims
        modules  = []
        modules.append(nn.Linear(in_dims, 128))
        modules.append(nn.LeakyReLU(0.1))

        modules.append(nn.Linear(128, 128))
        modules.append(nn.LeakyReLU(0.1))

        modules.append(NormalizedLinear(128, out_dims))

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

    def forward(self, h1, h2):
        h1 = h1.view(-1, self.in_dims)
        h2 = h2.view(-1, self.in_dims)

        h = h1 - h2
        h = self.model(h)

        return h
