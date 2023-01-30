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
from data import get_data

from utils.vis import loss_curve
from eval import  compute_metrics, nln, integrate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = '/data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_14-01-23.h5'
epochs=50
batch_size=64

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
              model_name='spirited-impossible-penguin-of-fortitude',
              limit='None',
              data_path =data_path,
              patch_size=64,
              seed=42,
              batch_size=batch_size,
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


def train(train_dataloader, 
          val_dataset,
          resnet):

    model_path = '_test/outputs/resnet'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    resnet.to(device)
    optimizer = torch.optim.Adam(resnet.parameters(),lr=1e-3) 
    train_loss, validation_accuracies = [], []
    total_step = len(train_dataloader)

    train_dataloader.dataset.set_supervision(True)
    val_dataset.set_supervision(True)
    for epoch in range(1, epochs+1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss, running_acc = 0.0, 0.0
            for _data, _target, _source  in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                _target = _target.type(torch.LongTensor).to(device)
                _data = _data.type(torch.FloatTensor).to(device)

                optimizer.zero_grad()

                z = resnet(_data)
                loss = resnet.loss_function(z, _target)['loss']
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                #_data = val_dataset.data_original.float().to(device)
                #z = resnet(_data).cpu().detach()
                #print(z.shape)
                #print(z.argmax(dim=-1).shape)
                #print(val_dataset.labels_original.shape)

                #val_acc = torch.sum(
                #    z.argmax(
                #       dim=-1) == val_dataset.labels_original) / val_dataset.labels_original.shape[0]
                #running_acc += val_acc
                tepoch.set_postfix(loss=loss.item())#, val_acc=val_acc.item())

            train_loss.append(running_loss / total_step)
            #validation_accuracies.append(running_acc/ total_step)

            if epoch % 10 == 0:  # TODO: check for model improvement
                # print validation loss

                torch.save(
                    resnet.state_dict(),
                    '{}/resnet.pt'.format(model_path))
            loss_curve(model_path,
                       epoch,
                       total_loss=train_loss,
                       #validation_accuracy=validation_accuracies,
                       descriptor='resnet')
            resnet.train()
    return resnet

def eval(resnet:ResNet,
        test_dataloader:DataLoader,
        mask):

    resnet.to(device)
    resnet.eval()

    predictions, targets  = [], []
    test_dataloader.dataset.set_supervision(True)
    test_dataloader.dataset.set_anomaly_mask('all')

    for _data, _target, _source  in test_dataloader:
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

    for encoding, anomaly in enumerate([0,1]):#defaults.anomalies
        ground_truth = [t==encoding for t in targets]

        precision, recall, thresholds = precision_recall_curve(ground_truth, predictions)
        f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
        auprc = auc(recall, precision)
        print(f'Supervised Class: {encoding} AUPRC: {auprc}, F1: {np.max(f1_scores)}')

        precision, recall, thresholds = precision_recall_curve(ground_truth, (predictions)*mask)
        f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
        auprc = auc(recall, precision)
        print(f'MASK * Supervised Class: {encoding} AUPRC: {auprc}, F1: {np.max(f1_scores)}')


    encoding = 1#len(defaults.anomalies)
    ground_truth = [t!=encoding for t in targets]
    precision, recall, thresholds = precision_recall_curve(ground_truth, predictions)
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    sup_mask = predictions>thresholds[np.argmax(f1_scores)]
    print(f'Supervised, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, mask)
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    print(f'Mask, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, np.logical_and(mask,sup_mask))
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    print(f'Mask AND , Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, np.logical_or(mask,sup_mask))
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    print(f'Mask OR, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')


def unsup_detector(test_dataloader, train_dataloader):
    """
        Separates anomalies from normal behaviour 
    """

    resnet = models.ResNet(out_dims=8,in_channels=4, latent_dim=128)
    resnet.load_state_dict(torch.load(f'outputs/position_classifier/{args.model_name}/resnet.pt'))
    resnet.to(args.device)
    resnet.eval()


    z_train, x_train = [],[]
    train_dataloader.dataset.set_supervision(False)
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
    test_dataloader.dataset.set_supervision(False)

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
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    print(f'SSL: AUPRC {auprc}, F1 {np.max(f1_scores)}')

    return dists>thresholds[np.argmax(f1_scores)]

def main():
    for i,seed in enumerate([200]):
        
        (train_dataset, 
        val_dataset, 
        test_dataset, 
        supervised_train_dataset, 
        supervised_val_dataset) = get_data(args)   

        train_dataloader = DataLoader(supervised_train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)

        # train model 
        resnet = ResNet(in_channels = 4, out_dims= 2)
        resnet = train(train_dataloader, supervised_val_dataset, resnet)
        resnet.load_state_dict(torch.load('_test/outputs/resnet/resnet.pt'))
        
        # function that separates anomalies from non-anomnalies using the SSL method
        # we need to tune treshold so that all anmolies are detected
        # returns updated test_dataloader

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False)

        mask = unsup_detector(test_dataloader, train_dataloader) 
        eval(resnet, test_dataloader,mask)

        # evaluate performance
        #thresholds = thresholds[:-10]
        #vals = np.zeros([len(f1_scores.keys()), len(thresholds)])
        #for i,k in enumerate(f1_scores.keys()):
        #    plt.plot(thresholds, f1_scores[k], label=k)
        #    vals[i] = f1_scores[k]

        #plt.plot(thresholds, np.mean(vals,axis=0), label='mean')
        #plt.xlabel('Thresholds')
        #plt.ylabel('F1 Score')
        #plt.legend()
        #plt.grid()
        #plt.savefig(f'/tmp/temp_{seed}',dpi=200)
        #plt.close('all')


if __name__ == '__main__':
    main()
