import argparse
import os
#from utils.data import sizes 
from coolname import generate_slug as new_name
import torch
import random
import numpy as np

parser = argparse.ArgumentParser(description='Radio Astronomy Anomaly Detection (RAAD)')

parser.add_argument('-model',metavar='-m', type=str, default='vae', choices={'vae', 'resnet', 'position_classifier'}, help = 'Model to train and evaluate')
parser.add_argument('-limit',metavar='-l', type=str, default='None', help = 'Limit on the number of samples in training data ')
parser.add_argument('-anomaly_class',metavar='-a', type=str,  default='lightning', help = 'The label of the anomalous class')
parser.add_argument('-epochs', metavar='-e', type=int, default=100, help = 'The number of epochs for training')
parser.add_argument('-latent_dim', metavar='-ld', type=int, default=2, help = 'The latent dimension size of the AE based models')
parser.add_argument('-neighbours', metavar='-n', nargs='+',type=int, default=[2,5,10], help = 'The maximum number of neighbours for latent anomaly detection')
parser.add_argument('-data_path', metavar='-dp', type=str, help = 'Path to training data')
parser.add_argument('-output_path', metavar='-op', default='outputs/results_LOFAR_AD.csv', type=str, help = 'Output path')
parser.add_argument('-seed', metavar='-s', type=int, default=42, help = 'The random seed')
parser.add_argument('-verbose', metavar='-v', type=str, help = 'Verbose output')
parser.add_argument('-patch_size', metavar='-ps', type=int, default=32, help = 'Dimension of patches')
parser.add_argument('-batch_size', metavar='-bs', type=int, default=2**11, help = 'Batch size')
parser.add_argument('-learning_rate', metavar='-lr', type=float, default=1e-3, help = 'Learning rate')
parser.add_argument('-hidden_dims', metavar='-hd', nargs='+',type=int, default=[32,64,128,256], help = 'Hidden dims for VAE')
parser.add_argument('-percentage_anomalies', metavar='-pa', type=float, default=0.1, help = 'Percentage of anomalies')
parser.add_argument('-amount', metavar='-am', type=float, default=0.1, help = 'Amount of test data')

args = parser.parse_args()

args.model_name = new_name()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

