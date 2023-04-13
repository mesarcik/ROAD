import torch
from torchvision import models
from vit_pytorch import ViT,SimpleViT
from torch import nn

class BackBone(nn.Module):
    def __init__(self, 
            in_channels: int,
            out_dims: int, 
            model_type:str='resnet50',
            supervision=False, 
            **kwargs) -> None:
        super(BackBone, self).__init__()

        assert (model_type == 'resnet18' or
                model_type == 'resnet50' or 
                model_type == 'resnet101' or 
                model_type == 'resnet152' or 
                model_type == 'vit' or
                model_type == 'convnext'), 'Backbone not defined'

        self.in_channels = in_channels
        self.out_dims = out_dims
        self.model_type = model_type

        if model_type == 'resnet152':
            self.model = models.resnet152(weights=None)
            self.model.conv1 = nn.Conv2d(self.in_channels, 64,  #increase the number of channels to channels
                                     kernel_size=(7, 7), 
                                     stride=(2, 2), 
                                     padding=(3, 3), 
                                     bias=False)
            #TODO
            self.model.fc = nn.Linear(2048,  self.out_dims)

        elif model_type == 'resnet101':
            self.model = models.resnet101(weights=None)
            self.model.conv1 = nn.Conv2d(self.in_channels, 64,  #increase the number of channels to channels
                                     kernel_size=(7, 7), 
                                     stride=(2, 2), 
                                     padding=(3, 3), 
                                     bias=False)
            #TODO
            self.model.fc = nn.Linear(2048,  self.out_dims)

        elif model_type == 'resnet50':
            self.model = models.resnet50(weights=None)
            self.model.conv1 = nn.Conv2d(self.in_channels, 64,  #increase the number of channels to channels
                                     kernel_size=(7, 7), 
                                     stride=(2, 2), 
                                     padding=(3, 3), 
                                     bias=False)
            self.model.fc = nn.Linear(2048,  self.out_dims)

        elif model_type == 'resnet18':
            self.model = models.resnet18(weights=None)
            self.model.conv1 = nn.Conv2d(self.in_channels, 64,  #increase the number of channels to channels
                                     kernel_size=(7, 7), 
                                     stride=(2, 2), 
                                     padding=(3, 3), 
                                     bias=False)
            self.model.fc = nn.Linear(512,  self.out_dims)

        elif model_type == 'vit':
            if supervision: 
                image_size=256
                patch_size=32
            else: 
                image_size=64
                patch_size=8
            self.model = SimpleViT(
                            image_size = image_size,
                            channels=self.in_channels,
                            patch_size = patch_size,
                            num_classes = self.out_dims,
                            dim = 512,
                            depth = 4,
                            heads = 16,
                            mlp_dim = 1024)

        elif model_type == 'convnext':
            self.model = models.convnext_tiny(weights=None)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 96,  #increase the number of channels to channels
                                     kernel_size=(4, 4), 
                                     stride=(4, 4))
            self.model.classifier[-1]  = nn.Linear(768, self.out_dims)

        self.loss_fn = nn.CrossEntropyLoss()

    def save(self, args, training, ft):
        fpath = self.fpath_from_name(args, training, ft)
        torch.save(self.state_dict(), fpath)

    def load(self, args, training, ft):
        fpath = self.fpath_from_name(args, training,  ft)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self, args, training, ft)->str:
        return f'outputs/models/{args.model_name}/backbone_{training}_{args.ood}_{args.seed}_{args.pretrain}_{ft}.pkl'

    def forward(self, 
                z: torch.tensor,
                **kwargs):
        c = self.model(z)
        return c

    def embed(self, 
            input:torch.tensor) -> torch.tensor:
        modules=list(self.model.children())[:-1]
        model=nn.Sequential(*modules)
        return model(input)[...,0,0] # remove the last two 1,1 dims

    def loss_function(self,
                      c:torch.tensor, 
                      labels:torch.tensor, 
                      **kwargs) -> torch.tensor:
        return self.loss_fn(c, labels)

