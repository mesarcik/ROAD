import torch
from torchvision import models
from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self,out_dims: int, hidden_dims: list, **kwargs) -> None:
        super(ClassificationHead, self).__init__()
        self.out_dims =  out_dims
        self.hidden_dims = hidden_dims 

        # classifier  
        modules  = []

        for i, h in enumerate(self.hidden_dims):
            if i >= len(self.hidden_dims)-1: break
            modules.append(nn.Linear(h, self.hidden_dims[i+1]))
            modules.append(nn.LeakyReLU(0.1))
            modules.append(nn.BatchNorm1d(num_features=self.hidden_dims[i+1]))

        modules.append(nn.Linear(self.hidden_dims[-1], 
                                 self.out_dims))

        self.classifier = nn.Sequential(*modules)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, 
                input: torch.tensor,
                **kwargs):
        c = self.classifier(input)
        return c

    def loss_function(self,
                      c:torch.tensor, 
                      labels:torch.tensor, 
                      **kwargs) -> dict:
        """
        Computes the BCE loss function
        """
        return {"loss": self.loss_fn(c, labels)}

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self, name):
        return f'ckpts/{name}/position_classifier.pkl'
