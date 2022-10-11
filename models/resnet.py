import torch
from torchvision import models
from torch import nn

class ResNet(nn.Module):
    def __init__(self,in_channels: int, dim: int, **kwargs) -> None:
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.dim = dim

        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(512, self.dim)
        self.resnet.conv1 = nn.Conv2d(self.in_channels, 64,  #increase the number of channels to channels
                                 kernel_size=(7, 7), 
                                 stride=(2, 2), 
                                 padding=(3, 3), 
                                 bias=False)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input: torch.tensor, **kwargs):
        return self.resnet(input)

    def embed(self, input:torch.tensor):
        modules=list(self.resnet.children())[:-1]
        model=nn.Sequential(*modules)
        return model(input)[...,0,0] # remove the last two 1,1 dims

    def loss_function(self,z:torch.tensor, 
                      labels:torch.tensor, 
                      **kwargs) -> dict:
        """
        Computes the BCE loss function
        """
        return {"loss": self.loss_fn(z, labels)}
