import torch
from torchvision import models
from torch import nn

class ResNet(nn.Module):
    def __init__(self,in_channels: int, out_dims: int, **kwargs) -> None:
        super(ResNet, self).__init__()

        self.in_channels = in_channels

        # embedding 
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(self.in_channels, 64,  #increase the number of channels to channels
                                 kernel_size=(7, 7), 
                                 stride=(2, 2), 
                                 padding=(3, 3), 
                                 bias=False)
        self.resnet =torch.nn.Sequential(*(list(resnet.children())[:-1]))

        # classifier  
        modules  = []
        modules.append(nn.Linear(1024, 128)) # 1024 because each input is 512
        modules.append(nn.LeakyReLU(0.1))

        modules.append(nn.Linear(128, 64))
        modules.append(nn.LeakyReLU(0.1))

        modules.append(nn.Linear(64, out_dims))
        modules.append(nn.BatchNorm1d(num_features=out_dims))

        self.classifier = nn.Sequential(*modules)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, 
                input_0: torch.tensor,
                input_1: torch.tensor,
                **kwargs):
        x_0 = self.resnet(input_0)[...,0,0]
        x_1 = self.resnet(input_1)[...,0,0]
        z = torch.cat([x_0, x_1], axis=1)
        c = self.classifier(z)

        return c

    def embed(self, input:torch.tensor) -> torch.tensor:
        return self.resnet(input)[...,0,0] # remove the last two 1,1 dims

    def loss_function(self,
                      c:torch.tensor, 
                      labels:torch.tensor, 
                      **kwargs) -> dict:
        """
        Computes the BCE loss function
        """
        return {"loss": self.loss_fn(c, labels)}

