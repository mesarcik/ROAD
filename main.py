import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from utils.vis import imscatter
from utils.args import args
from utils.data.defaults import default_stations, default_frequency_bands
import os

from data import get_data
from models import VAE, ResNet
from train import train_vae, train_resnet
from eval import eval_vae, eval_resnet


def main():
    print(args.model_name)
    train_dataset, val_dataset, test_dataset = get_data(args)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    if args.model == 'vae':
        vae = VAE(in_channels=4,
                  latent_dim=args.latent_dim,
                  patch_size=args.patch_size,
                  hidden_dims=args.hidden_dims)
        #vae = train_vae(train_dataloader, vae, args)
        # TODO Add eval/train state
        vae.load_state_dict(torch.load('outputs/vae/lightning/big-khaki-jackal-of-inquire/vae.pt'))
        eval_vae(vae, train_dataloader, args, error='nln')

    elif args.model == 'resnet':
        resnet = ResNet(dim=len(default_frequency_bands), in_channels=4)
        resnet = train_resnet(train_dataloader, val_dataset, resnet, args)
        #resnet.load_state_dict(torch.load('outputs/resnet/lightning/notorious-ancient-lori-of-kindness/resnet.pt'))
        eval_resnet(resnet, train_dataloader, args, error='nln')


if __name__ == '__main__':
    main()
