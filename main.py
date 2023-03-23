import torch
from torch.utils.data import DataLoader

import numpy as np
from utils.args import args
from utils.data import defaults

from data import get_data
from models import VAE, BackBone, PositionClassifier, Decoder, ClassificationHead
from train import train_vae, train_supervised, train_ssl
from fine_tune import fine_tune
from eval import eval_vae, eval_supervised, eval_classification_head, eval_knn
from utils import plot_results


print(args.model_name)
(train_dataset,
        val_dataset,
        test_dataset,
        supervised_train_dataset,
        supervised_val_dataset)= get_data(args)

train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

supervised_train_dataloader = DataLoader(supervised_train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

test_dataloader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)
test_dataloader.dataset.set_seed(np.random.randint(100))

if args.model == 'vae':
    vae = VAE(in_channels=4,
            latent_dim=args.latent_dim,
            patch_size=args.patch_size,
            hidden_dims=args.hidden_dims)
    if args.load_model:
        vae.load_state_dict(torch.load(f'outputs/vae/{args.model_name}/vae.pt'))
    else:
        vae = train_vae(train_dataloader, vae, args)
    #eval_vae(vae, train_dataloader, test_dataloader, args, error='nln')

elif args.model in ('supervised', 'all'):
    backbone = BackBone(in_channels=4,
            out_dims=len(defaults.anomalies)+1,
            model_type='resnet50')
    if args.load_model:
        backbone.load(args)
    else:
        backbone = train_supervised(supervised_train_dataloader,
                supervised_val_dataset,
                backbone,
                args)
    pred, thr = eval_supervised(backbone, test_dataloader, args)

elif args.model in ('ssl', 'all'):
    backbone = BackBone(in_channels=4,
            out_dims=args.latent_dim,
            model_type=args.backbone)
    position_classifier = PositionClassifier(latent_dim=args.latent_dim,
            out_dims=8)
    decoder = Decoder(out_channels=4,
                     patch_size=args.patch_size,
                     latent_dim=args.latent_dim,
                     n_layers=5)
    classification_head =  ClassificationHead(out_dims=1,
                                                latent_dim= args.latent_dim)
    if args.load_model:
        backbone.load(args)
        position_classifier.load(args)
        decoder.load(args)
        classification_head.load(args)
    else:
        backbone, position_classifier, decoder = train_ssl(train_dataloader,
                val_dataset,
                backbone,
                position_classifier,
                decoder,
                args)
        backbone_ft, classification_head = fine_tune(supervised_train_dataloader,
                                        val_dataset,
                                        backbone,
                                        classification_head,
                                        args)
    pred, thr = eval_classification_head(backbone_ft, classification_head, test_dataloader, args)
    pred, thr = eval_knn(backbone, decoder, test_dataloader, train_dataloader, args)

    if args.model  == 'all':
        pred, thr = eval_supervised(backbone, test_dataloader, args,  pred, thr)


    ##plot_results(args.output_path,
    ##            f'outputs/{args.model}/{args.model_name}',  
    ##            args.model_name,
    ##            np.max(args.neighbours))
