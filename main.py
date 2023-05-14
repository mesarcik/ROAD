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
if args.model == 'vae':
    vae = VAE(in_channels=4,
            latent_dim=args.latent_dim,
            patch_size=args.patch_size,
            hidden_dims=args.hidden_dims)
    if args.load_model:
        vae.load(args)
    else:
        vae = train_vae(train_dataloader, vae, args)
    pred, thr = eval_knn(vae, None, test_dataloader, train_dataloader, args)

elif args.model in ('supervised', 'all', 'random_init'):
    supervised_backbone = BackBone(in_channels=4,
            out_dims=len(defaults.anomalies)+1,
            model_type=args.backbone,
            supervision=True)
    if args.load_model:
        supervised_backbone.load(args,'supervised', False)
    else:
        supervised_backbone = train_supervised(supervised_train_dataloader,
                supervised_val_dataset,
                supervised_backbone,
                args)
    #pred, thr = eval_supervised(supervised_backbone, test_dataloader, args)

if args.model in ('ssl', 'all', 'random_init'):
    ssl_backbone = BackBone(in_channels=4,
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
        ssl_backbone.load(args, 'ssl', True)
        position_classifier.load(args)
        decoder.load(args)
        classification_head.load(args)
    else:
        if args.model != 'random_init':
            ssl_backbone, position_classifier, decoder = train_ssl(train_dataloader,
                    val_dataset,
                    ssl_backbone,
                    position_classifier,
                    decoder,
                    args)

        #pred_knn, thr_knn = eval_knn(ssl_backbone, decoder, test_dataloader, train_dataloader, args)

        ssl_backbone, classification_head = fine_tune(supervised_train_dataloader,
                                        supervised_val_dataset,
                                        test_dataloader,
                                        ssl_backbone,
                                        classification_head,
                                        args)

    #pred_ft, thr_ft = eval_classification_head(ssl_backbone, classification_head, test_dataloader, args)

if args.model  in ('all', 'random_init'):
    for i in range(10):
        test_dataloader.dataset.set_seed(np.random.randint(1000))
        pred, thr = eval_supervised(supervised_backbone, test_dataloader, args)
        pred_ft, thr_ft = eval_classification_head(ssl_backbone, classification_head, test_dataloader, args)
        pred, thr = eval_supervised(supervised_backbone, test_dataloader, args,  pred_ft, thr_ft)
    
