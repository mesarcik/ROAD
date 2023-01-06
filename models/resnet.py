import torch
from torchvision import models
from torch import nn

class ResNet(nn.Module):
    def __init__(self,in_channels: int, out_dims: int, latent_dim:int,  **kwargs) -> None:
        super(ResNet, self).__init__()

        self.in_channels = in_channels

        # embedding 
        self.resnet = models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(self.in_channels, 64,  #increase the number of channels to channels
                                 kernel_size=(7, 7), 
                                 stride=(2, 2), 
                                 padding=(3, 3), 
                                 bias=False)
        self.resnet.fc = nn.Linear(2048, latent_dim)
        #self.resnet =torch.nn.Sequential(*(list(resnet.children())[:-1]))

        # classifier  
        modules  = []
        modules.append(nn.Linear(2*latent_dim, 64)) # 256 because each input is 128 
        modules.append(nn.LeakyReLU(0.1))

        modules.append(nn.Linear(64, out_dims))
        modules.append(nn.BatchNorm1d(num_features=out_dims))

        self.classifier = nn.Sequential(*modules)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, 
                x_0: torch.tensor,
                x_1: torch.tensor,
                **kwargs):
        z = torch.cat([x_0, x_1], axis=1)
        c = self.classifier(z)

        return c

    def embed(self, input:torch.tensor) -> torch.tensor:
        return self.resnet(input) # remove the last two 1,1 dims

    def loss_function(self,
                      c:torch.tensor, 
                      labels:torch.tensor, 
                      **kwargs) -> dict:
        """
        Computes the BCE loss function
        """
        return {"loss": self.loss_fn(c, labels)}

