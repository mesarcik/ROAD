import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from utils.vis import imscatter
from utils.args import args
from utils.data.defaults import default_stations, default_frequency_bands, SIZE
import os

from data import get_data, get_finetune_data
from models import VAE, ResNet, PositionClassifier, Decoder, ClassificationHead
from train import train_vae, train_resnet, train_position_classifier
from eval import eval_vae, eval_resnet, eval_finetune


def main():
    print(args.model_name)
    transform = None#transforms.Compose([transforms.RandomResizedCrop(size=(args.patch_size, args.patch_size),
                #                                                 scale=(0.5, 1.0)),  
                #                    transforms.RandomHorizontalFlip(p=0.5),
                #                    transforms.RandomVerticalFlip(p=0.5),
                #                    ])
    train_dataset, val_dataset, test_dataset = get_data(args,transform=transform)

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
        vae = train_vae(train_dataloader, vae, args)
        if args.load_model:
            vae.load_state_dict(torch.load('outputs/vae/{}/vae.pt'.format(args.model_name)))
        eval_vae(vae, train_dataloader, test_dataloader, args, error='nln')

    elif args.model == 'resnet':
        resnet = ResNet(dim=test_dataset.n_patches**2, in_channels=4)
        resnet = train_resnet(train_dataloader, val_dataset, resnet, args)
        if args.load_model:
            resnet.load_state_dict(torch.load('outputs/resnet/{}/resnet.pt'.format(args.model_name)))
        eval_resnet(resnet, train_dataloader, test_dataloader, args, error='nln')

    elif args.model == 'position_classifier':
        resnet = ResNet(out_dims=8,in_channels=4, latent_dim=args.latent_dim)
        classifier = PositionClassifier(in_dims=128, out_dims=1)#len(default_frequency_bands[args.patch_size]))
        if args.load_model:
            resnet.load_state_dict(torch.load('outputs/position_classifier/{}/resnet.pt'.format(args.model_name)))
            classifier.load_state_dict(torch.load('outputs/position_classifier/{}/classifier.pt'.format(args.model_name)))
        #resnet = train_position_classifier(train_dataloader, val_dataset, resnet, classifier, args)
        #eval_resnet(resnet, train_dataloader, test_dataloader, args, error='nln')

        if args.fine_tune:
            train_dataset, test_dataset = get_finetune_data(args,transform=transform)
            classification_head = ClassificationHead(out_dims=len(anomalies)+1,
                                                     hidden_dims=[args.latent_dim*(SIZE[0]//(args.patch_size))**2,
                                                     int(0.25*args.latent_dim*(SIZE[0]//(args.patch_size))**2),
                                                     int(0.0625*args.latent_dim*(SIZE[0]//(args.patch_size))**2)])
            classification_head = fine_tune(train_dataloader, test_dataloader, resnet, classification_head, args)
            eval_finetune(resnet, classification_head, test_dataloader, args, args.epochs)



if __name__ == '__main__':
    main()
