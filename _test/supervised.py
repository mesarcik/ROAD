import torch
from torch import nn
import pprint
import torchvision 
import copy
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils.data import defaults, combine
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
import models
import h5py 
import os
from data import get_data

from utils.vis import loss_curve
from utils.data import reconstruct_distances
from eval import  compute_metrics, nln, integrate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#data_path = '/data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_14-01-23.h5'
data_path = '/data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_07-03-23.h5'
epochs=35
batch_size=32
beta=1

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def set_amount(self, amount):
        self.amount = amount

args = Namespace(device = device,
              latent_dim = 64, 
              anomaly_class =-1,
              model='position_classifier',
              amount=5,
              model_name='skilled-perfect-mongoose-of-respect',
              limit='None',
              data_path =data_path,
              patch_size=64,
              seed=42,
              batch_size=batch_size,
              epochs=epochs,
              neighbours=[5])


# get model 
class ResNet(nn.Module):
    def __init__(self,in_channels:int, out_dims: int,  **kwargs) -> None:
        super(ResNet, self).__init__()

        self.in_channels = in_channels

        # embedding 
        self.resnet = torchvision.models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(4, 64,  #increase the number of channels to channels
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
          resnet,
          remove,
          _model_seed):

    model_path = '_test/outputs/resnet'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    resnet.to(device)
    optimizer = torch.optim.Adam(resnet.parameters(),lr=1e-3) 
    train_loss, validation_accuracies = [], []
    train_dataloader.dataset.set_supervision(True)
    total_step = len(train_dataloader)
    prev_acc = 0

    _val_data = val_dataset.data.float().to(device)
    _labels = val_dataset.labels

    for epoch in range(1, 50+1):
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

                z = resnet(_val_data).cpu().detach()
                val_acc = torch.sum(
                    z.argmax(axis=-1) == _labels) / val_dataset.labels.shape[0]
                running_acc += val_acc
                tepoch.set_postfix(loss=loss.item(), val_acc=val_acc.item())

            train_loss.append(running_loss / total_step)
            validation_accuracies.append(running_acc/ total_step)

            acc = []
            for _encoding, _class in enumerate(defaults.anomalies):
                precision, recall, thresholds = precision_recall_curve(_labels==_encoding,  z.argmax(axis=-1) == _encoding)
                f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
                acc.append(np.max(f_scores))#auc(recall, precision)

            precision, recall, thresholds = precision_recall_curve(_labels==len(defaults.anomalies),  z.argmax(axis=-1) == len(defaults.anomalies))
            f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
            acc.append(np.nanmax(f_scores))#auc(recall, precision)

            acc = np.nanmean(acc)
            if prev_acc < acc:
                torch.save(
                    resnet.state_dict(),
                    '{}/resnet_{}_{}.pt'.format(model_path,remove,_model_seed))
                prev_acc = acc
                print(f'model saved {prev_acc}')

            loss_curve(model_path,
                       epoch,
                       total_loss=train_loss,
                       validation_accuracy=validation_accuracies,
                       descriptor='resnet')
            resnet.train()
    return resnet


def eval(resnet:ResNet,
        test_dataloader:DataLoader,
        results:dict,
        dists,
        thr,
        dists_nln,
        thr_nln,
        _class,
        OOD):

    resnet.to(device)
    resnet.eval()

    predictions, targets  = [], []
    test_dataloader.dataset.set_supervision(True)

    for _data, _target, _source  in test_dataloader:
        _target = _target.type(torch.LongTensor).to(device)
        _data = _data.type(torch.FloatTensor).to(device)

        c  = resnet(_data)
        c = c.argmax(dim=-1).cpu().detach()
        target = _target.cpu().detach()

        predictions.extend(c.numpy().flatten())
        targets.extend(target.numpy().flatten())

    predictions, targets = np.array(predictions), np.array(targets)
    unknown_anomaly = len(defaults.anomalies)+1

    masked_pred = []
    for p, d in zip(predictions, dists):
        if d<thr: # mask is true for anomalies 
            masked_pred.append(len(defaults.anomalies))
        elif d>=thr and p==len(defaults.anomalies):
            masked_pred.append(unknown_anomaly)
        elif d>=thr and p != len(defaults.anomalies):
            masked_pred.append(p)
    masked_pred = np.array(masked_pred)

    masked_pred_nln = []
    for p, d in zip(predictions, dists_nln):
        if d<thr_nln: # mask is true for anomalies 
            masked_pred_nln.append(len(defaults.anomalies))
        elif d>=thr_nln and p==len(defaults.anomalies):
            masked_pred_nln.append(unknown_anomaly)
        elif d>=thr_nln and p != len(defaults.anomalies):
            masked_pred_nln.append(p)
    masked_pred_nln = np.array(masked_pred_nln)

    anomalies = copy.deepcopy(defaults.anomalies)
    anomalies.append('normal')
    anomalies.append('unknown_anomaly')
    print('Supervised')
    print(confusion_matrix(targets==len(defaults.anomalies), predictions==len(defaults.anomalies)))
    for encoding, anomaly in enumerate(anomalies):
        normal, anomalous = [], [] 
        for i in range(len(predictions)):
            if targets[i] != len(defaults.anomalies):
                if targets[i] != predictions[i]:
                    anomalous.append(predictions[i])
            if targets[i] == len(defaults.anomalies):
                if targets[i] != predictions[i]:
                    normal.append(predictions[i])
    print('normal', encoding, anomaly, len(normal))
    print('anomalous', encoding, anomaly, len(anomalous))
    print('_'*10)
    print('SSL')
    print(confusion_matrix(targets==len(defaults.anomalies), masked_pred==len(defaults.anomalies)))
    for encoding, anomaly in enumerate(anomalies):
        normal, anomalous = [], [] 
        for i in range(len(masked_pred)):
            if targets[i] != len(defaults.anomalies):
                if targets[i] != masked_pred[i]:
                    anomalous.append(masked_pred[i])
            if targets[i] == len(defaults.anomalies):
                if targets[i] != masked_pred[i]:
                    normal.append(masked_pred[i])
    print('normal', encoding, anomaly, len(normal))
    print('anomalous', encoding, anomaly, len(anomalous))
    print('_'*10)
    print('Dists')
    print(confusion_matrix(targets==len(defaults.anomalies), masked_pred_nln==len(defaults.anomalies)))
    for encoding, anomaly in enumerate(anomalies):
        normal, anomalous = [], [] 
        for i in range(len(masked_pred_nln)):
            if targets[i] != len(defaults.anomalies):
                if targets[i] != masked_pred_nln[i]:
                    anomalous.append(masked_pred_nln[i])
            if targets[i] == len(defaults.anomalies):
                if targets[i] != masked_pred_nln[i]:
                    normal.append(masked_pred_nln[i])
    print('normal', encoding, anomaly, len(normal))
    print('anomalous', encoding, anomaly, len(anomalous))
    print('_'*10)


    temp = [] 
    for i in range(len(predictions)):
        if targets[i] == len(defaults.anomalies):
            if targets[i] != predictions[i]:
                temp.append(predictions[i])
    print(9, 'normal', np.unique(temp, return_counts=True))

    print(np.unique(targets, return_counts=True))

    if not OOD:
        for encoding, anomaly in enumerate(defaults.anomalies):
            precision, recall, thresholds = precision_recall_curve(targets==encoding, predictions==encoding)
            f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
            auprc = auc(recall, precision)
            results[anomaly]['sup'].append(np.max(f_scores))
            print(f'Supervised Class: {encoding} AUPRC: {auprc}, F1: {np.max(f_scores)}')

            precision, recall, thresholds = precision_recall_curve(targets==encoding, masked_pred==encoding)

            f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
            auprc = auc(recall, precision)
            results[anomaly]['unsup'].append(np.max(f_scores))
            print(f'Combined CLass: {encoding} AUPRC: {auprc}, F1: {np.max(f_scores)}') 

            precision, recall, thresholds = precision_recall_curve(targets==encoding, masked_pred_nln==encoding)

            f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
            auprc = auc(recall, precision)
            results[anomaly]['dists'].append(np.max(f_scores))
            print(f'Dists Combined CLass: {encoding} AUPRC: {auprc}, F1: {np.max(f_scores)}') 

    encoding = len(defaults.anomalies)
    ground_truth = [t==encoding for t in targets]
    precision, recall, thresholds = precision_recall_curve(ground_truth, predictions==encoding)
    f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
    auprc = auc(recall, precision)
    sup_mask = predictions>thresholds[np.argmax(f_scores)]
    results['normal']['sup'].append(np.max(f_scores))
    print(f'Supervised, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, dists)
    f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
    auprc = auc(recall, precision)
    print(f'Mask, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, dists_nln)
    f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
    auprc = auc(recall, precision)
    print(f'Dists Mask, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, masked_pred==encoding)
    f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
    auprc = auc(recall, precision)
    results['normal']['unsup'].append(np.max(f_scores))
    print(f'Combined Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, masked_pred_nln==encoding)
    f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
    auprc = auc(recall, precision)
    results['normal']['dists'].append(np.max(f_scores))
    print(f'Dists Combined Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f_scores)}')

    ground_truth = [t!=encoding for t in targets]
    precision, recall, thresholds = precision_recall_curve(ground_truth, predictions!=encoding)
    f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
    auprc = auc(recall, precision)
    sup_mask = predictions!=encoding #>thresholds[np.argmax(f_scores)]
    results['anomaly']['sup'].append(np.max(f_scores))
    print(f'Supervised, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth, dists)
    f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
    auprc = auc(recall, precision)
    print(f'Mask, class: {encoding}, auprc: {auprc}, f1: {np.max(f_scores)}')
    precision, recall, thresholds = precision_recall_curve(ground_truth, dists_nln)
    f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
    auprc = auc(recall, precision)
    print(f'Dists Mask, class: {encoding}, auprc: {auprc}, f1: {np.max(f_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth,masked_pred!=encoding)
    f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
    auprc = auc(recall, precision)
    results['anomaly']['unsup'].append(np.max(f_scores))
    print(f'Combined Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f_scores)}')

    precision, recall, thresholds = precision_recall_curve(ground_truth,masked_pred_nln!=encoding)
    f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
    auprc = auc(recall, precision)
    results['anomaly']['dists'].append(np.max(f_scores))
    print(f'Dists Combined Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f_scores)}')
    
    return results

def unsup_detector(supervised_train_dataloader, supervised_val_dataset, train_dataloader, test_dataloader, _class, _model_seed, pretrain):
    """
        Separates anomalies from normal behaviour 
    """

    dists = get_distances(supervised_train_dataloader, 
                          supervised_val_dataset,
                          train_dataloader, 
                          test_dataloader, 
                          _class,
                          _model_seed, 
                          pretrain)
    labels =test_dataloader.dataset.labels
    ground_truth = [l != len(defaults.anomalies) for l in labels]

    precision, recall, thresholds = precision_recall_curve(ground_truth, dists)
    f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
    auprc = auc(recall, precision)
    thr = thresholds[np.argmax(f_scores)]

    _output = []
    _remove = [] 
    for i in range(len(dists)):
        if ground_truth[i] != (dists[i]>thr):
            _output.append(labels[i])
            if labels[i] == len(defaults.anomalies):
                #fig, axs = plt.subplots(1,4,figsize=(8,2))
                #for p in range(4):
                #    axs[p].imshow(test_dataloader.dataset.data[i,p,...], aspect='auto', interpolation='nearest')
                #plt.savefig(f'/tmp/normal_{i}.pdf',dpi=300)
                #plt.close(fig)
                _remove.append(test_dataloader.dataset.source[i])
    print('total', np.unique(labels, return_counts=True))
    print('missclassified', np.unique(_output, return_counts=True))
    print('len', len(_remove))

    if os.path.exists('_test/excluded_sources.npy'):
        _remove = np.concatenate([_remove, np.load('_test/excluded_sources.npy')])
        #np.save('_test/excluded_sources.npy', np.unique(_remove))
    else:
        np.save('_test/excluded_sources.npy', np.unique(_remove))


    print(f'SSL: AUPRC {auprc}, F1 {np.max(f_scores)}')
    #TODO this threhsold maximises anommalous class detection and not normal class
    # Maybe i need to investigate how to more appropriately set this threhold. 
    return dists, thresholds[np.argmax(f_scores)]


def get_distances(supervised_train_dataloader, 
                  supervised_val_dataset, 
                  train_dataloader, 
                  test_dataloader,
                  _class,
                  _model_seed,
                  pretrain):
    """
        Separates anomalies from normal behaviour 
    """
    resnet = models.ResNet(out_dims=8,in_channels=4, latent_dim=args.latent_dim)
    classifier = models.ClassificationHead(out_dims=1, latent_dim=args.latent_dim)

    if os.path.exists(f'outputs/position_classifier/{args.model_name}/resnet_{_class}_{_model_seed}_{pretrain}.pt'):
        print('resnet loaded')
        print(f'outputs/position_classifier/{args.model_name}/resnet_{_class}_{_model_seed}.pt')
        resnet.load_state_dict(torch.load(f'outputs/position_classifier/{args.model_name}/resnet_{_class}_{_model_seed}_{pretrain}.pt'))
        resnet.to(args.device)
        resnet.eval()

        print('classifier loaded')
        classifier.load(args.model_name, _class, _model_seed, pretrain)
        classifier.to(args.device)
        classifier.eval()
    else:
        if pretrain:
            resnet.load_state_dict(torch.load(f'outputs/position_classifier/{args.model_name}/resnet_150.pt'))

        resnet.to(args.device)
        classifier.to(args.device)

        optimizer = torch.optim.Adam(list(classifier.parameters()) + list(resnet.parameters()),lr=1e-4) 
        supervised_train_dataloader.dataset.set_supervision(False)
        total_step = len(supervised_train_dataloader)

        loss_fn = nn.BCELoss()
        previous_max_f1 = 0
        for epoch in range(1,51):
            with tqdm(supervised_train_dataloader, unit="batch") as tepoch:
                for _data, _target,_,_,_,_  in tepoch:
                    _data = combine(_data,0,2).float().to(args.device)
                    _target = _target[:,0].float().to(args.device)

                    optimizer.zero_grad()
                    _z = resnet.embed(_data)
                    Z = _z.reshape([len(_z)//int(defaults.SIZE[0]//args.patch_size)**2, args.latent_dim*int(defaults.SIZE[0]//args.patch_size)**2]) 
                    _c = classifier(Z).squeeze(1)
                    _labels = _target!=len(defaults.anomalies)
                    loss = loss_fn(_c,_labels.to(device, dtype=torch.float))
                    loss.backward()
                    optimizer.step()

                    tepoch.set_postfix(epoch=epoch,
                                       total_loss=loss.item())

                resnet.eval()
                classifier.eval()
                outputs = eval_ssl(resnet, classifier, test_dataloader)#TODO: change to validation set 
                precision, recall, thresholds = precision_recall_curve(test_dataloader.dataset.labels!=len(defaults.anomalies), outputs)
                f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
                acc = np.max(f_scores)#auc(recall, precision)
                print(epoch, auc(recall, precision), acc)
                if previous_max_f1 < acc:
                    classifier.save(args.model_name, _class, _model_seed, pretrain)
                    torch.save(resnet.state_dict(), f'outputs/position_classifier/{args.model_name}/resnet_{_class}_{_model_seed}_{pretrain}.pt')
                    previous_max_f1 = acc

                resnet.train()
                classifier.train()

        resnet.load_state_dict(torch.load(f'outputs/position_classifier/{args.model_name}/resnet_{_class}_{_model_seed}_{pretrain}.pt'))
        resnet.to(args.device)
        resnet.eval()

        classifier.load(args.model_name, _class, _model_seed, pretrain)
        classifier.to(args.device)
        classifier.eval()

    outputs = eval_ssl(resnet, classifier, test_dataloader)

    return outputs 

def get_error_nln(train_dataloader, 
                  test_dataloader,
                  _class,
                  _model_seed):
    """
        Separates anomalies from normal behaviour 
    """

    resnet = models.ResNet(out_dims=8,in_channels=4, latent_dim=args.latent_dim)
    classifier = models.ClassificationHead(out_dims=1, latent_dim=args.latent_dim)
    resnet.load_state_dict(torch.load(f'outputs/position_classifier/{args.model_name}/resnet_{_class}_{_model_seed}.pt'))
    resnet.to(args.device)
    resnet.eval()

    z_train = []
    train_dataloader.dataset.set_supervision(False)
    for _data, _target,_,_,_,_ in train_dataloader:
        _data = combine(_data,0,2).float().to(args.device)
        z = resnet.embed(_data)
        z_train.append(z.cpu().detach().float().numpy())
    z_train = np.vstack(z_train)

    test_dataloader.dataset.set_supervision(False)
    z_test = []
    for _data, _target,_,_,_,_ in test_dataloader:
        _data = combine(_data,0,2).float().to(args.device)
        _z = resnet.embed(_data)
        z_test.append(_z.cpu().detach().numpy())
    z_test = np.vstack(z_test)

    d, _ = nln(z_test, z_train, 5, args)
    dists = integrate(d, args)

    labels =test_dataloader.dataset.labels
    ground_truth = [l != len(defaults.anomalies) for l in labels]

    precision, recall, thresholds = precision_recall_curve(ground_truth, dists)
    f_scores = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
    auprc = auc(recall, precision)
    thr = thresholds[np.argmax(f_scores)]
    print(f'Dists : AUPRC {auprc}, F1 {np.max(f_scores)}')

    return dists, thr 

def eval_ssl(resnet, classifier, test_dataloader): 
    anomaly = -1 
    test_dataloader.dataset.set_supervision(False)
    outputs = []

    for _data, _target,_,_,_,_ in test_dataloader:
        _data = combine(_data,0,2).float().to(args.device)
        _target = combine(_target,0,2).float().to(args.device)

        _z = resnet.embed(_data)
        Z = _z.reshape([len(_z)//int(defaults.SIZE[0]//args.patch_size)**2, args.latent_dim*int(defaults.SIZE[0]//args.patch_size)**2]) 
        _c = classifier(Z).squeeze(1)
        outputs.append(np.expand_dims(_c.float().cpu().detach().numpy(),-1))

    outputs = np.vstack(outputs)[:,0]
    return outputs


def plot(results,remove):
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
    
    ax.set_xticks(pos, list(results.keys()), rotation = 10, fontsize=5)
    
    plt.legend()
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'/tmp/temp_{remove}',dpi=300)
    plt.close('all')

def plot_missclassifications(resnet, test_dataloader, _class):
    resnet.to(device)
    resnet.eval()

    predictions, targets  = [], []
    test_dataloader.dataset.set_supervision(True)

    for _data, _target, _source  in test_dataloader:
        _target = _target.type(torch.LongTensor).to(device)
        _data = _data.type(torch.FloatTensor).to(device)

        c  = resnet(_data)
        c = c.argmax(dim=-1).cpu().detach()
        target = _target.cpu().detach()

        predictions.extend(c.numpy().flatten())
        targets.extend(target.numpy().flatten())

    predictions, targets = np.array(predictions), np.array(targets)

    results = {}
    _labels = copy.deepcopy(defaults.anomalies)
    _labels.append('normal')
    for encoding, anomaly in enumerate(_labels):
        results[encoding] = np.zeros(len(defaults.anomalies)+1) 
        for i in range(len(predictions)):
            if targets[i] == encoding:
                if targets[i] != predictions[i]:
                    results[encoding][predictions[i]]+=1
    #print('-'*10)
    #print('-'*10)
    #print(f'Class: {_class} -> {results[defaults.anomalies.index(_class)]}')
    #print('-'*10)
    #print('-'*10)
    pos = np.arange(len(defaults.anomalies)+1)
    width = 0.3
    fig, ax = plt.subplots(3,3, figsize=(10,10))
    cnt  = 0
    for i in range(3):
        for j in range(3):
            ax[i,j].bar(pos, results[cnt], width=width, label ='')
            ax[i,j].set_xticks(pos, list(results.keys()), rotation = 10, fontsize=5)
            ax[i,j].set_title(defaults.anomalies[cnt], fontsize=5)
            ax[i,j].set_xlabel('Classes')
            ax[i,j].set_ylabel('F1 Score')
            ax[i,j].grid()
            ax[i,j].set_ylim([0, 10])
            cnt+=1
    plt.tight_layout()
    plt.savefig(f'/tmp/OOD_misclassification_{_class}_OOD',dpi=300)
    plt.close('all')


def plot_comparison():
    means_normal_unsup ,stds_normal_unsup, means_normal_sup ,stds_normal_sup  = [], [], [] ,[]
    means_anom_unsup ,stds_anom_unsup, means_anom_sup ,stds_anom_sup  = [], [], [], []
    anomalies = []
    for fn in glob('_test/bak_results_*'):
        anomalies.append(fn.split('results_')[-1].split('.')[0])
        f = np.load(fn, allow_pickle=True).item()
        normal_sup = f['normal']['sup']
        anom_sup = f['anomaly']['sup']
        normal_unsup = f['normal']['unsup']
        anom_unsup  = f['anomaly']['unsup']

    means_normal_unsup.append(np.mean(normal_unsup))
    stds_normal_unsup.append(np.std(normal_unsup))
    means_anom_unsup.append(np.mean(anom_unsup))
    stds_anom_unsup.append(np.std(anom_unsup))

    means_normal_sup.append(np.mean(normal_sup))
    stds_normal_sup.append(np.std(normal_sup))
    means_anom_sup.append(np.mean(anom_sup))
    stds_anom_sup.append(np.std(anom_sup))

    x = np.arange(len(anomalies))  # the label locations
    width = 0.4  # the width of the bars
    fig, ax = plt.subplots(constrained_layout=True)

    rects = ax.bar(x + width/2*0, means_normal_sup, yerr=stds_normal_sup, width=width/2, label='Normal Supervised',color='C0')
    ax.bar_label(rects, padding=3,rotation =90, fontsize=5)
    rects = ax.bar(x + width/2*1, means_normal_unsup, yerr=stds_normal_unsup, width=width/2, label='Normal SSL',color='C1')
    ax.bar_label(rects, padding=3,rotation =90, fontsize=5)
    rects = ax.bar(x + width/2*2, means_anom_sup, yerr=stds_anom_sup, width=width/2, label='Anomalous Supervised',color='C0', hatch='/')
    ax.bar_label(rects, padding=3,rotation =90, fontsize=5)
    rects = ax.bar(x + width/2*3, means_anom_unsup, yerr=stds_anom_unsup, width=width/2, label='Anomalous SSL',color='C1', hatch='/')
    ax.bar_label(rects, padding=3,rotation =90, fontsize=5)
    ax.set_xticks(x + width, anomalies, rotation=10, fontsize=5)
    ax.set_ylim([0,1])
    ax.set_xlabel('OOD Classes')
    ax.set_ylabel('F1 Score')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f'/tmp/temp',dpi=300)

def main():
    OOD = False
    #for  _encoding, _class in enumerate(defaults.anomalies): 
    for i in range(1):
        _encoding, _class = -1,'None'
        (train_dataset, 
        val_dataset, 
        test_dataset, 
        supervised_train_dataset, 
        supervised_val_dataset) = get_data(args)   

        supervised_train_dataloader = DataLoader(supervised_train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)

        #test_dataset.remove_sources(np.load('_test/excluded_sources.npy')) 
        # train model 
        results = {
                    'normal':{'sup':[],'unsup':[], 'dists':[]},
                    'anomaly':{'sup':[], 'unsup':[], 'dists':[]},
                    'oscillating_tile':{'sup':[],'unsup':[], 'dists':[]}, #[],
                    #'real_high_noise':{'sup':[],'unsup':[]},
                    'first_order_high_noise':{'sup':[],'unsup':[], 'dists':[]},
                    #'third_order_high_noise':{'sup':[],'unsup':[]},
                    'first_order_data_loss':{'sup':[],'unsup':[], 'dists':[]},
                    'third_order_data_loss':{'sup':[],'unsup':[], 'dists':[]},
                    'lightning':{'sup':[],'unsup':[], 'dists':[]},
                    #'strong_radio_emitter':{'sup':[],'unsup':[]},
                    'galactic_plane':{'sup':[],'unsup':[], 'dists':[]},
                    'source_in_sidelobes':{'sup':[],'unsup':[], 'dists':[]},
                    #'ionosphere':{'sup':[],'unsup':[]},
                    'solar_storm':{'sup':[],'unsup':[], 'dists':[]},
                    }
        for _model_seed in range (2):
            test_dataset.set_seed(342)
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False)
            if OOD:
                test_dataloader.dataset.set_anomaly_mask(_encoding)
            else:
                test_dataloader.dataset.set_anomaly_mask(-1)

            resnet = ResNet(in_channels = 4, out_dims= len(defaults.anomalies) +1)
            if os.path.exists(f'_test/outputs/resnet/resnet_{_class}_{_model_seed}.pt'):
                resnet.load_state_dict(torch.load(f'_test/outputs/resnet/resnet_{_class}_{_model_seed}.pt'))
            else:
                resnet = train(supervised_train_dataloader, supervised_val_dataset, resnet, _class,_model_seed)
                resnet.load_state_dict(torch.load(f'_test/outputs/resnet/resnet_{_class}_{_model_seed}.pt'))
            #plot_missclassifications(resnet, test_dataloader, _class)
            
            for seed in [1]: #,23, 5, 123, 53, 78, 90, 42]:
                print(f'{_class}->{seed}')
                test_dataset.set_seed(seed)
                test_dataloader = DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False)
                if OOD:
                    test_dataloader.dataset.set_anomaly_mask(_encoding)
                else:
                    test_dataloader.dataset.set_anomaly_mask(-1)

                dists, thr = unsup_detector(supervised_train_dataloader,
                                                  supervised_val_dataset,
                                                  train_dataloader, 
                                                  test_dataloader, 
                                                  _class,
                                                  _model_seed,
                                                  True) 

                dists_nln, thr_nln = unsup_detector(supervised_train_dataloader,
                                                  supervised_val_dataset,
                                                  train_dataloader, 
                                                  test_dataloader, 
                                                  _class,
                                                  _model_seed,
                                                  False) 

                #dists_nln, thr_nln= get_error_nln(train_dataloader,
                #                                    test_dataloader,
                #                                    _class,
                #                                    _model_seed)
                results = eval(resnet, 
                               test_dataloader, 
                               results, 
                               dists, 
                               thr, 
                               dists_nln, 
                               thr_nln, 
                               _encoding, 
                               OOD)
                #results = eval_OOD(resnet, test_dataloader, mask, dists, thr, _encoding, results)
        plot(results, _class)
        np.save(f'_test/results_resnet50_100_double_{_class}', results)
            


if __name__ == '__main__':
    main()
