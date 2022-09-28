import argparse
import os
#from utils.data import sizes 
from coolname import generate_slug as new_name
from torch import device as _device
from torch import cuda as _cuda

parser = argparse.ArgumentParser(description='Radio Astronomy Anomaly Detection (RAAD)')

parser.add_argument('-model',metavar='-m', type=str, default='VAE', choices={'VAE'}, help = 'Model to train and evaluate')
parser.add_argument('-limit',metavar='-l', type=str, default='None', help = 'Limit on the number of samples in training data ')
parser.add_argument('-anomaly_class',metavar='-a', type=str,  default='lightning', help = 'The label of the anomalous class')
parser.add_argument('-epochs', metavar='-e', type=int, default=100, help = 'The number of epochs for training')
parser.add_argument('-latent_dim', metavar='-ld', type=int, default=2, help = 'The latent dimension size of the AE based models')
parser.add_argument('-neighbours', metavar='-n', type=int, default=10, help = 'The maximum number of neighbours for latent anomaly detection')
parser.add_argument('-data_path', metavar='-dp', type=str, default='/home/mmesarcik/data/LOFAR/compressed/organised/LOFAR_AD_dataset_21-09-22.pkl', help = 'Path to training data')
parser.add_argument('-seed', metavar='-s', type=int, help = 'The random seed')
parser.add_argument('-verbose', metavar='-v', type=str, help = 'Verbose output')
parser.add_argument('-patch_size', metavar='-ps', type=int, default=32, help = 'Dimension of patches')
parser.add_argument('-batch_size', metavar='-bs', type=int, default=2**10, help = 'Batch size')
parser.add_argument('-learning_rate', metavar='-lr', type=float, default=1e-3, help = 'Learning rate')
parser.add_argument('-hidden_dims', metavar='-hd', nargs='+',type=int, default=[32,64,128,256], help = 'Hidden dims for VAE')

args = parser.parse_args()

args.model_name = new_name()
args.device = _device("cuda:0" if _cuda.is_available() else "cpu")