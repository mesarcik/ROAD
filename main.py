import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from utils.vis import imscatter
from utils.args import args
from utils.data import defaults
import os
import gc

from data import get_data
from models import VAE, BackBone, PositionClassifier, Decoder, ClassificationHead
from train import train_vae, train_supervised, train_ssl
from eval import eval_vae, eval_resnet
from utils import plot_results


def main():
    print(args.model_name)
    (train_dataset, 
            val_dataset, 
            test_dataset, 
            supervised_train_dataset, 
            supervised_val_dataset) = get_data(args)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    supervised_train_dataloader = DataLoader(supervised_train_dataset,
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
        if args.load_model:
            vae.load_state_dict(torch.load('outputs/vae/{}/vae.pt'.format(args.model_name)))
        else:
            vae = train_vae(train_dataloader, vae, args)
        eval_vae(vae, train_dataloader, test_dataloader, args, error='nln')

    elif args.model == 'supervised':
        backbone = BackBone(in_channels=4, 
                out_dims=len(defaults.anomalies), 
                model_type='resnet50')
        if args.load_model:
            backbone.load(args)
        else:
            backbone = train_supervised(supervised_train_dataloader, 
                    supervised_val_dataset, 
                    backbone,
                    args)

        eval(backbone, 
            supervised_train_dataloader, 
            test_dataloader, 
            args, 
            error='nln')

    elif args.model == 'position_classifier':
        backbone = BackBone(in_channels=4,
                out_dims=args.latent_dim, 
                model_type=args.backbone)
        position_classifier = PositionClassifier(in_dims=args.latent_dim, 
                out_dims=8)
        decoder = Decoder(out_channels=4,
                         patch_size=args.patch_size,
                         latent_dim=args.latent_dim,
                         n_layers=5)

        if args.load_model:
            resnet.load(args)
            position_classifier.load(args)
            decoder.load(args)
        else:
            backbone, position_classifier, decoder = train_ssl(train_dataloader, 
                    val_dataset, 
                    backbone, 
                    position_classifier, 
                    decoder,
                    args)

        for i in range(10):
            test_dataloader.dataset.set_seed(np.random.randint(100))
            eval_resnet(resnet, train_dataloader, test_dataloader, args, error='nln')
        plot_results(args.output_path,
                    f'outputs/{args.model}/{args.model_name}',  
                    args.model_name,
                    np.max(args.neighbours))



if __name__ == '__main__':
    main()
