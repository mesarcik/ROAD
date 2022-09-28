import torch 
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from utils.vis import imscatter 
from utils.args import args
import os

from data import get_data 
from models import VAE
from train import train_vae


def main():
    train_dataset, test_dataset = get_data(args)

    train_dataloader = DataLoader(train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True)

    test_dataloader = DataLoader(test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False)

    vae = VAE(in_channels=4, 
            latent_dim=args.latent_dim,
            patch_size=args.patch_size, 
            hidden_dims=args.hidden_dims)

    vae = train_vae(train_dataloader, vae, args)

if __name__ == '__main__':
    main()