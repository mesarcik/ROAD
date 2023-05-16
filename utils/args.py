import argparse
import os
#from utils.data import sizes 
from coolname import generate_slug as new_name
import torch
import random
import numpy as np

parser = argparse.ArgumentParser(description='Radio Astronomy Anomaly Detection (RAAD)')

parser.add_argument('-model',metavar='-m', type=str, default='vae', choices={'dknn', 'vae','random_init', 'supervised', 'ssl', 'all'}, help = 'Model to train and evaluate')
parser.add_argument('-limit',metavar='-l', type=str, default='None', help = 'Limit on the number of samples in training data ')
parser.add_argument('-ood',metavar='-oo', type=int,  default=-1, help = 'The label of the anomalous class')
parser.add_argument('-epochs', metavar='-e', type=int, default=100, help = 'The number of epochs for training')
parser.add_argument('-latent_dim', metavar='-ld', type=int, default=2, help = 'The latent dimension size of the AE based models')
parser.add_argument('-neighbours', metavar='-n', nargs='+',type=int, default=[2,5,10], help = 'The maximum number of neighbours for latent anomaly detection')
parser.add_argument('-data_path', metavar='-dp', type=str, help = 'Path to training data')
parser.add_argument('-model_path', metavar='-mp', type=str, help = 'Path to model data')
parser.add_argument('-output_path', metavar='-op', default='outputs/results_LOFAR_AD.csv', type=str, help = 'Output path')
parser.add_argument('-seed', metavar='-s', type=int, default=42, help = 'The random seed')
parser.add_argument('-verbose', metavar='-v', type=str, help = 'Verbose output')
parser.add_argument('-patch_size', metavar='-ps', type=int, default=32, help = 'Dimension of patches')
parser.add_argument('-batch_size', metavar='-bs', type=int, default=2**11, help = 'Batch size')
parser.add_argument('-learning_rate', metavar='-lr', type=float, default=1e-3, help = 'Learning rate')
parser.add_argument('-hidden_dims', metavar='-hd', nargs='+',type=int, default=[32,64,128,256], help = 'Hidden dims for VAE')
parser.add_argument('-percentage_data', metavar='-pd', type=float, default=0.5, help = 'Percentage training data')
parser.add_argument('-amount', metavar='-am', type=float, default=0.1, help = 'Amount of test data')
parser.add_argument('-model_name', metavar='-mn', type=str, default=None, help = 'Model name for loading')
parser.add_argument('-fine_tune', metavar='-ft', type=bool, default=True, help = 'Use pretrained models for finetuning')
parser.add_argument('-kernel_size', metavar='-ks', type=int, default=3,  choices={3,5}, help = 'Kernel size of context prediction')
parser.add_argument('-backbone',metavar='-bb', type=str, default='resnet18', choices={'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152','convnext','vit'}, help = 'Model to train and evaluate')
parser.add_argument('-pretrain', metavar='-pt', type=bool, default=True,  help = 'Random intialisation or pretraining')
parser.add_argument('-resize_amount', metavar='-ra', type=float, default=0.05,  help = 'The random crop and resize percentage')


args = parser.parse_args()

if args.model_name is None: 
    args.model_name = new_name()
    args.load_model = False
else: 
    try:
        with open(f'{args.model_path}/outputs/models/{args.model_name}/model.config', 'r') as f:
            # Load configuration file values
             d = {}
             for line in f:
                if 'cuda:' in line: continue #bug with previous version writing the device
                (key, val) = line.strip().replace(' ', '').split(':')
                d[key] = val
        args.load_model = True
        args.seed = int(d['seed'])
        args.backbone = d['backbone']
        args.ood = int(d['ood'])
        args.percentage_data = float(d['percentage_data'])
    except FileNotFoundError:
        raise FileNotFoundError(errno.ENOENT, 
                                os.strerror(errno.ENOENT), 
                                f'{args.model_path}/outputs/models/{args.model_name}/model.config')


args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

