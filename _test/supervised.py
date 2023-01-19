import torch
from torch import nn
import torchvision 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils.data import defaults
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import models
import h5py 
import os
import data

from utils.vis import loss_curve
from eval import  compute_metrics, nln, integrate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = '/data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_14-01-23.h5'
epochs=50
batch_size=256

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def set_amount(self, amount):
        self.amount = amount

args = Namespace(device = device,
              latent_dim = 128, 
              anomaly_class ='all',
              model='position_classifier',
              amount=6,
              model_name='private-cheetah-of-spectacular-influence',
              limit='None',
              data_path =data_path,
              patch_size=64,
              seed=42,
              batch_size=2**10,
              neighbours=[5])
# get model 
class ResNet(nn.Module):
    def __init__(self,in_channels: int, out_dims: int,  **kwargs) -> None:
        super(ResNet, self).__init__()

        self.in_channels = in_channels

        # embedding 
        self.resnet = torchvision.models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(self.in_channels, 64,  #increase the number of channels to channels
                                 kernel_size=(7, 7), 
                                 stride=(2, 2), 
                                 padding=(3, 3), 
                                 bias=False)
        self.resnet.fc = nn.Linear(2048, out_dims)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, 
                x: torch.tensor,
                **kwargs):
        c = self.resnet(x)
        return c

    def embed(self, input:torch.tensor) -> torch.tensor:
        return self.resnet(input) # remove the last two 1,1 dims

    def loss_function(self,
                      c:torch.tensor, 
                      labels:torch.tensor, 
                      **kwargs) -> dict:
        """
        Computes the BCE loss function
        """
        return {"loss": self.loss_fn(c, labels)}

# get data 
class LOFARDataset(Dataset):
    def __init__(self,data:np.array,labels:np.array):# set default types

        self.labels = self.encode_labels(labels)
        self.labels = torch.from_numpy(self.labels)

        data = self.normalise(data)
        self.data = torch.from_numpy(data).permute(0,3,1,2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datum = self.data[idx,...]
        label = self.labels[idx]

        return datum, label


    def encode_labels(sef, labels):
        _labels = []
        for label in labels:
            if label =='':
                _labels.append(len(defaults.anomalies))
            else:
                _labels.append([i for i,a in enumerate(defaults.anomalies) if a in label][0])
        return np.array(_labels)

    def normalise(self, data:np.array)->np.array:
        """
            perpendicular polarisation normalisation

        """
        _data = np.zeros(data.shape)
        for i, spec in enumerate(data):
            for pol in range(data.shape[-1]):
                _min, _max = np.percentile(spec[...,pol], [1,97])
                temp = np.clip(spec[...,pol],_min, _max)
                temp = np.log(temp)
                temp  = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
                _data[i,...,pol] = temp
        _data = np.nan_to_num(_data, 0)
        return _data
            
def _join(hf:h5py.File, field:str)->np.array:
    """
        Joins together the normal and anomalous testing data
        
        Parameters
        ----------
        hf: h5py dataset 
        field: the field that is meant to be concatenated along 

        Returns
        -------
        data: concatenated array

    """
    #test_data = hf['test_data/{}'.format(field)][:]
    #train_data = hf['train_data/{}'.format(field)][:]
    #data = np.concatenate([test_data, train_data], axis=0)
    data = hf['test_data/{}'.format(field)][:]
    for a in defaults.anomalies:
        if a != 'all': 
            _data = hf['anomaly_data/{}/{}'.format(a,field)][:]
            data = np.concatenate([_data, data],axis=0)
    return data

def train(train_dataloader: DataLoader, 
          val_dataset: LOFARDataset,
          resnet:ResNet):

    model_path = '_test/outputs/resnet'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    resnet.to(device)
    optimizer = torch.optim.Adam(resnet.parameters(),lr=1e-3) 
    train_loss, validation_accuracies = [], []
    total_step = len(train_dataloader)

    for epoch in range(1, epochs+1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss, running_acc = 0.0, 0.0
            for _data, _target  in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                _target = _target.type(torch.LongTensor).to(device)
                _data = _data.type(torch.FloatTensor).to(device)

                optimizer.zero_grad()

                z = resnet(_data)
                loss = resnet.loss_function(z, _target)['loss']
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _data = val_dataset.data.float().to(device)
                z = resnet(_data).cpu().detach()

                val_acc = torch.sum(
                    z.argmax(
                        dim=-1) == val_dataset.labels) / val_dataset.labels.shape[0]
                running_acc += val_acc
                tepoch.set_postfix(loss=loss.item(), val_acc=val_acc.item())

            train_loss.append(running_loss / total_step)
            validation_accuracies.append(running_acc/ total_step)

            if epoch % 10 == 0:  # TODO: check for model improvement
                # print validation loss

                torch.save(
                    resnet.state_dict(),
                    '{}/resnet.pt'.format(model_path))
            loss_curve(model_path,
                       epoch,
                       total_loss=train_loss,
                       validation_accuracy=validation_accuracies,
                       descriptor='resnet')
            resnet.train()
    return resnet

def eval(resnet:ResNet,
        test_dataloader:DataLoader,
        dists, thresholds):

    resnet.to(device)
    resnet.eval()

    predictions, targets  = [], []

    for _data, _target in test_dataloader:
        _target = _target.type(torch.LongTensor).to(device)
        _data = _data.type(torch.FloatTensor).to(device)

        c  = resnet(_data)
        c = c.argmax(dim=-1).cpu().detach()
        target = _target.cpu().detach()

        predictions.extend(c.numpy().flatten())
        targets.extend(target.numpy().flatten())

    predictions, targets = np.array(predictions), np.array(targets)
    # For the null class
    f1_scores = {'oscillating_tile':[],
                'real_high_noise':[],
                'data_loss':[],
                'lightning':[],
                'strong_radio_emitter':[],
                'solar_storm':[]}
    for t in thresholds:
        mask = dists>t
       # encoding = len(defaults.anomalies)
       # auroc, auprc, f1 = compute_metrics(targets,
       #                                    np.logical_or(predictions,np.invert(mask))==encoding, 
       #                                    anomaly=encoding, 
       #                                    multiclass=True)
       # print("Anomaly:{},  AUROC: {:.4f}, AUPRC: {:.4f}, F1: {:.4f}".format('normal',
       #                                                                               auroc,
       #                                                                               auprc,
       #                                                                               f1))
       # targets = targets[mask]
       # predictions=predictions[mask]
        for encoding, anomaly in enumerate(defaults.anomalies):
            if anomaly =='all': continue

            auroc, auprc, f1 = compute_metrics(targets[mask],
                                               predictions[mask]==encoding,
                                               anomaly=encoding,
                                               multiclass=True)
            f1_scores[anomaly].append(f1)
    return f1_scores

def unsup_detector(hf, test_indexes):
    """
        Separates anomalies from normal behaviour 
    """

    resnet = models.ResNet(out_dims=8,in_channels=4, latent_dim=128)
    resnet.load_state_dict(torch.load('outputs/position_classifier/private-cheetah-of-spectacular-influence/resnet.pt'))
    resnet.to(args.device)
    resnet.eval()

    train_data = hf['train_data/data'][:]
    train_labels = hf['train_data/labels'][:].astype(str)
    train_frequency_band  = hf['train_data/frequency_band'][:]
    train_source = hf['train_data/source'][:].astype(str)

    train_dataset = data.LOFARDataset(train_data, 
                                 train_labels, 
                                 train_frequency_band, 
                                 train_source,
                                 args,
                                 transform=None,
                                 roll=False)


    test_dataset = data.LOFARDataset(_join(hf, 'data')[test_indexes],
                                _join(hf, 'labels').astype(str)[test_indexes],
                                _join(hf, 'frequency_band')[test_indexes],
                                _join(hf, 'source').astype(str)[test_indexes],
                                args,
                                transform=None,
                                test=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    z_train, x_train = [],[]
    for _data, _target,_,_,_,_  in train_dataloader:
        _data = _data.float().to(args.device)
        z = resnet.embed(_data)
        z_train.append(z.cpu().detach().numpy())
        x_train.append(_data.cpu().detach().numpy())

    z_train = np.vstack(z_train) 
    x_train = np.vstack(x_train) 

    anomaly = 'all'
    z_test, x_test = [], []
    test_dataloader.dataset.set_anomaly_mask(anomaly)

    for _data, _target,_,_,_,_ in test_dataloader:
        _data = _data.float().to(args.device)
        z = resnet.embed(_data)
        z_test.append(z.cpu().detach().numpy())
        x_test.append(_data.cpu().detach().numpy())

    z_test, x_test = np.vstack(z_test), np.vstack(x_test)

    N = int(np.max(args.neighbours))
    _D, _I = nln(z_test, z_train, N, args)
    D,I = _D[:,:-1], _I[:,:-1]
    dists = integrate(D,  args)
    labels =test_dataloader.dataset.labels[::int(defaults.SIZE[0]//args.patch_size)**2]
    ground_truth = [l != '' for l in labels]

    precision, recall, thresholds = precision_recall_curve(ground_truth, dists)
    auprc = auc(recall, precision)

    return dists, thresholds

def main():
    for i,seed in enumerate([200,100,32,51,61,75,1,44,11,33]):

        hf = h5py.File(data_path,'r')
        _data = _join(hf, 'data')
        _labels = _join(hf, 'labels').astype(str)
        _source = _join(hf, 'source').astype(str)
        _indexes = np.arange(len(_labels))
        test_indexes, train_indexes = train_test_split(_indexes,
                                                       test_size=0.5,
                                                       random_state=seed)

        test_data, train_data = _data[test_indexes], _data[train_indexes]
        test_labels, train_labels= _labels[test_indexes], _labels[train_indexes]
        test_sources, train_sources = _source[test_indexes], _source[train_indexes]


        (train_data, val_data, 
        train_labels, val_labels) = train_test_split(train_data,
                                                     train_labels,
                                                     test_size=0.05, 
                                                     random_state=42)

        train_dataset = LOFARDataset(train_data, train_labels)
        val_dataset =   LOFARDataset(val_data, val_labels)
        test_dataset = LOFARDataset(test_data, test_labels)

        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # train model 
        resnet = ResNet(in_channels = 4, out_dims= len(defaults.anomalies)+1)
        resnet = train(train_dataloader, val_dataset, resnet)
        #resnet.load_state_dict(torch.load('_test/outputs/resnet/resnet.pt'))
        
        # function that separates anomalies from non-anomnalies using the SSL method
        # we need to tune treshold so that all anmolies are detected
        # returns updated test_dataloader
        dists, thresholds = unsup_detector(hf, test_indexes) 

        # evaluate performance
        thresholds = thresholds[:-10]
        f1_scores = eval(resnet, test_dataloader,dists, thresholds)
        vals = np.zeros([len(f1_scores.keys()), len(thresholds)])
        for i,k in enumerate(f1_scores.keys()):
            plt.plot(thresholds, f1_scores[k], label=k)
            vals[i] = f1_scores[k]

        plt.plot(thresholds, np.mean(vals,axis=0), label='mean')
        plt.xlabel('Thresholds')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid()
        plt.savefig(f'/tmp/temp_{seed}',dpi=200)
        plt.close('all')


if __name__ == '__main__':
    main()
