import torch
from torchvision import models
from torch import nn

class BackBone(nn.Module):
    def __init__(self, 
            in_channels: int,
            out_dims: int, 
            model_type:str='resnet50',
            **kwargs) -> None:
        super(BackBone, self).__init__()

        assert (model_type == 'resnet50' or
                model_type == 'resnet18'), 'Backbone not defined'

        self.in_channels = in_channels
        self.out_dims = out_dims
        self.model_type = model_type

        if model_type == 'resnet50':
            self.resnet = models.resnet50(weights=None)
            self.resnet.conv1 = nn.Conv2d(self.in_channels, 64,  #increase the number of channels to channels
                                     kernel_size=(7, 7), 
                                     stride=(2, 2), 
                                     padding=(3, 3), 
                                     bias=False)
            self.resnet.fc = nn.Linear(2048,  self.out_dims)

        self.loss_fn = nn.CrossEntropyLoss()

    def save(self, args):
        fpath = self.fpath_from_name(args)
        torch.save(self.state_dict(), fpath)

    def load(self, args):
        fpath = self.fpath_from_name(args)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self,args)->str:
        return f'outputs/{args.model}/{args.model_name}/backbone_{args.ood}_{args.seed}_{args.pretrain}.pkl'

    def forward(self, 
                z: torch.tensor,
                **kwargs):
        c = self.resnet(z)
        return c

    def embed(self, 
            input:torch.tensor) -> torch.tensor:
        modules=list(self.resnet.children())[:-1]
        model=nn.Sequential(*modules)
        return model(input)[...,0,0] # remove the last two 1,1 dims

    def loss_function(self,
                      c:torch.tensor, 
                      labels:torch.tensor, 
                      **kwargs) -> torch.tensor:
        return self.loss_fn(c, labels)

