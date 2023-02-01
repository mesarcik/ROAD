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
batch_size=32

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def set_amount(self, amount):
        self.amount = amount

args = Namespace(device = device,
              latent_dim = 128, 
              anomaly_class =-1,
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
def magic_combine(x, dim_begin, dim_end):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)

def eval(resnet:ResNet,
        test_dataloader:DataLoader,
        results:dict,
        mask):

    resnet.to(device)
    resnet.eval()

    predictions, targets  = [], []
    test_dataloader.dataset.set_supervision(True)
    test_dataloader.dataset.set_anomaly_mask(-1)

    for _data, _target, _source  in test_dataloader:
        _target = _target.type(torch.LongTensor).to(device)
        _data = _data.type(torch.FloatTensor).to(device)

        c  = resnet(_data)
        c = c.argmax(dim=-1).cpu().detach()
        target = _target.cpu().detach()

        predictions.extend(c.numpy().flatten())
        targets.extend(target.numpy().flatten())

    predictions, targets = np.array(predictions), np.array(targets)

    for encoding, anomaly in enumerate(defaults.anomalies):
        ground_truth = [t==encoding for t in targets]

        precision, recall, thresholds = precision_recall_curve(ground_truth, predictions==encoding)
        f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
        auprc = auc(recall, precision)
        results[anomaly]['sup'].append(np.max(f1_scores))
        print(f'Supervised Class: {encoding} AUPRC: {auprc}, F1: {np.max(f1_scores)}')

        precision, recall, thresholds = precision_recall_curve(ground_truth, (predictions==encoding)*mask)
        f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
        auprc = auc(recall, precision)
        results[anomaly]['unsup'].append(np.max(f1_scores))
        print(f'MASK * Supervised Class: {encoding} AUPRC: {auprc}, F1: {np.max(f1_scores)}')
        print('')


    encoding = len(defaults.anomalies)
    ground_truth = [t==encoding for t in targets]
    precision, recall, thresholds = precision_recall_curve(ground_truth, predictions==encoding)
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    sup_mask = predictions>thresholds[np.argmax(f1_scores)]
    results['normal']['sup'].append(np.max(f1_scores))
    print(f'Supervised, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, np.invert(mask))
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    print(f'Mask, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, (predictions==encoding)*np.invert(mask))
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    results['normal']['unsup'].append(np.max(f1_scores))
    print(f'MUL, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')


    encoding = len(defaults.anomalies)
    ground_truth = [t!=encoding for t in targets]
    precision, recall, thresholds = precision_recall_curve(ground_truth, predictions!=encoding)
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    sup_mask = predictions!=encoding #>thresholds[np.argmax(f1_scores)]
    results['anomaly']['sup'].append(np.max(f1_scores))
    print(f'Supervised, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, mask)
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    print(f'Mask, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, np.logical_and(mask,sup_mask))
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    results['anomaly']['unsup'].append(np.max(f1_scores))
    print(f'Mask AND , Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, np.logical_or(mask,sup_mask))
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    print(f'Mask OR, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')
    
    return results

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
        _data = magic_combine(_data,0,2).float().to(args.device)
        z = resnet.embed(_data)
        z_train.append(z.cpu().detach().numpy())
        x_train.append(_data.cpu().detach().numpy())

    z_train = np.vstack(z_train) 
    x_train = np.vstack(x_train) 

    anomaly = -1 
    z_test, x_test = [], []
    test_dataloader.dataset.set_anomaly_mask(anomaly)
    test_dataloader.dataset.set_supervision(False)

    for _data, _target,_,_,_,_ in test_dataloader:
        _data = magic_combine(_data,0,2).float().to(args.device)
        z = resnet.embed(_data)
        z_test.append(z.cpu().detach().numpy())
        x_test.append(_data.cpu().detach().numpy())

    z_test, x_test = np.vstack(z_test), np.vstack(x_test)

    N = int(np.max(args.neighbours))
    _D, _I = nln(z_test, z_train, N, args)
    D,I = _D[:,:-1], _I[:,:-1]
    dists = integrate(D,  args)
    labels =test_dataloader.dataset.labels
    ground_truth = [l != len(defaults.anomalies) for l in labels]

    precision, recall, thresholds = precision_recall_curve(ground_truth, dists)
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    #print(f'SSL: AUPRC {auprc}, F1 {np.max(f1_scores)}')
    #TODO this threhsold maximises anommalous class detection and not normal class
    # Maybe i need to investigate how to more appropriately set this threhold. 
    return dists>thresholds[np.argmax(f1_scores)]

def plot(results):
    pos = np.arange(len(results))
    width = 0.3
    fig, ax = plt.subplots()
    supmean, supstd = [], []
    unsupmean, unsupstd = [], []
    for i,k in enumerate(results.keys()):
        supmean.append(np.mean(results[k]['sup']))
        supstd.append(np.std(results[k]['sup']))
        unsupmean.append(np.mean(results[k]['unsup']))
        unsupstd.append(np.std(results[k]['unsup']))
    
    
    ax.bar(pos, supmean, yerr=supstd, width=width, label ='sup')
    ax.bar(pos+width, unsupmean, yerr=unsupstd, width=width, label ='unsup')
    
    ax.set_xticks(pos, list(results.keys()), rotation = 10)
    
    plt.legend()
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'/tmp/temp',dpi=300)
    plt.close('all')

def main():
    (train_dataset, 
    val_dataset, 
    test_dataset, 
    supervised_train_dataset, 
    supervised_val_dataset) = get_data(args)   

    supervised_train_dataloader = DataLoader(supervised_train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    # train model 
    resnet = ResNet(in_channels = 4, out_dims= len(defaults.anomalies)+1)
    resnet = train(supervised_train_dataloader, supervised_val_dataset, resnet)
    resnet.load_state_dict(torch.load('_test/outputs/resnet/resnet.pt'))
    
    # function that separates anomalies from non-anomnalies using the SSL method
    # we need to tune treshold so that all anmolies are detected
    # returns updated test_dataloader
    # For the null class
    results = {
                'normal':{'sup':[],'unsup':[]},
                'anomaly':{'sup':[], 'unsup':[]},
                'oscillating_tile':{'sup':[],'unsup':[]}, #[],
                'real_high_noise':{'sup':[],'unsup':[]},
                'data_loss':{'sup':[],'unsup':[]},
                'lightning':{'sup':[],'unsup':[]},
                'strong_radio_emitter':{'sup':[],'unsup':[]},
                'solar_storm':{'sup':[],'unsup':[]},
                }
    for seed in [1,10,42, 98, 102,4, 2, 101, 33]:
        print('')
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)
        test_dataset.set_seed(seed)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False)

        mask = unsup_detector(test_dataloader, train_dataloader) 
        results = eval(resnet, test_dataloader, results, mask)
    plot(results)


if __name__ == '__main__':
    main()
