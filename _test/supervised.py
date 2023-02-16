import torch
from torch import nn
import torchvision 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils.data import defaults, combine
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import models
import h5py 
import os
from data import get_data

from utils.vis import loss_curve
from utils.data import reconstruct_distances
from eval import  compute_metrics, nln, integrate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#data_path = '/data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_14-01-23.h5'
data_path = '/data/mmesarcik/LOFAR/LOFAR_AD/constructed_lofar_ad_dataset_12-02-23.h5'
epochs=25
batch_size=16

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
              model_name='vigorous-authentic-nyala-of-piety',
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
          remove):

    model_path = '_test/outputs/resnet'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    resnet.to(device)
    optimizer = torch.optim.Adam(resnet.parameters(),lr=1e-3) 
    train_loss, validation_accuracies = [], []
    train_dataloader.dataset.set_supervision(True)
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
                _labels = val_dataset.labels

                val_acc = torch.sum(
                    z.argmax(axis=-1) == _labels) / val_dataset.labels.shape[0]
                running_acc += val_acc
                tepoch.set_postfix(loss=loss.item(), val_acc=val_acc.item())

            train_loss.append(running_loss / total_step)
            validation_accuracies.append(running_acc/ total_step)

            if epoch % 10 == 0:  # TODO: check for model improvement
                # print validation loss

                torch.save(
                    resnet.state_dict(),
                    '{}/resnet_{}.pt'.format(model_path,remove))
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
        mask):

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

    masked_pred = []
    for p, m in zip(predictions, mask):
        if not m: # mask is true for anomalies 
            masked_pred.append(len(defaults.anomalies))
        else:
            masked_pred.append(p)
    masked_pred = np.array(masked_pred)

    for encoding, anomaly in enumerate(defaults.anomalies):
        precision, recall, thresholds = precision_recall_curve(targets==encoding, predictions==encoding)
        f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
        auprc = auc(recall, precision)
        results[anomaly]['sup'].append(np.max(f1_scores))
        print(f'Supervised Class: {encoding} AUPRC: {auprc}, F1: {np.max(f1_scores)}')


        precision, recall, thresholds = precision_recall_curve(targets==encoding, masked_pred==encoding)
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

    precision, recall, thresholds = precision_recall_curve(ground_truth, masked_pred==encoding)
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    results['normal']['unsup'].append(np.max(f1_scores))
    print(f'MUL, Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')


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

    precision, recall, thresholds = precision_recall_curve(ground_truth,masked_pred!=encoding)
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    results['anomaly']['unsup'].append(np.max(f1_scores))
    print(f'Mask AND , Class: {encoding}, AUPRC: {auprc}, F1: {np.max(f1_scores)}')
    
    return results

def unsup_detector(supervised_train_dataloader, supervised_val_dataset, train_dataloader, test_dataloader):
    """
        Separates anomalies from normal behaviour 
    """

    dists = get_error_nln(supervised_train_dataloader, supervised_val_dataset,train_dataloader, test_dataloader)
    labels =test_dataloader.dataset.labels
    ground_truth = [l != len(defaults.anomalies) for l in labels]

    precision, recall, thresholds = precision_recall_curve(ground_truth, dists)
    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    auprc = auc(recall, precision)
    thr = thresholds[np.argmax(f1_scores)]

    _output = []
    _remove = [] 
    for i in range(len(dists)):
        if ground_truth[i] != (dists[i]>thr):
            _output.append(labels[i])
            if labels[i] == len(defaults.anomalies):
                _remove.append(test_dataloader.dataset.source[i])
    print('total', np.unique(labels, return_counts=True))
    print('missclassified', np.unique(_output, return_counts=True))
    print('len', len(_remove))

    if os.path.exists('_test/excluded_sources.npy'):
        _remove = np.concatenate([_remove, np.load('_test/excluded_sources.npy')])
        np.save('_test/excluded_sources.npy', np.unique(_remove))
    else:
        np.save('_test/excluded_sources.npy', np.unique(_remove))


    print(f'SSL: AUPRC {auprc}, F1 {np.max(f1_scores)}')
    #TODO this threhsold maximises anommalous class detection and not normal class
    # Maybe i need to investigate how to more appropriately set this threhold. 
    return dists>thresholds[np.argmax(f1_scores)]


def get_distances(supervised_train_dataloader, 
                  supervised_val_dataset, 
                  train_dataloader, 
                  test_dataloader):
    """
        Separates anomalies from normal behaviour 
    """

    resnet = models.ResNet(out_dims=8,in_channels=4, latent_dim=8)
    resnet.load_state_dict(torch.load(f'outputs/position_classifier/{args.model_name}/resnet.pt'))
    resnet.to(args.device)
    #resnet.eval()

    z_train = []
    train_dataloader.dataset.set_supervision(False)
    for _data, _target,_,_,_,_ in train_dataloader:
        _data = combine(_data,0,2).float().to(args.device)
        z = resnet.embed(_data)
        z_train.append(z.cpu().detach().float().numpy())
    z_train = np.vstack(z_train)

    classifier = models.ClassificationHead(out_dims=1, latent_dim=128)
    classifier.to(args.device)
    optimizer = torch.optim.Adam(list(classifier.parameters()) + list(resnet.parameters()),lr=1e-4) 
    supervised_train_dataloader.dataset.set_supervision(False)
    total_step = len(supervised_train_dataloader)

    loss_fn = nn.BCELoss()
    for epoch in range(1,25+1):
        with tqdm(supervised_train_dataloader, unit="batch") as tepoch:
            for _data, _target,_,_,_,_  in tepoch:
                _data = combine(_data,0,2).float().to(args.device)
                _target = _target[:,0].float().to(args.device)

                _z = resnet.embed(_data)
                # compute distances 
                #d, _ = nln(_z.cpu().detach().numpy(), z_train, 5, args)
                #d = np.mean(d, axis=tuple(range(1,d.ndim)))
                Z = _z.reshape([len(_z)//int(defaults.SIZE[0]//args.patch_size)**2, 8*int(defaults.SIZE[0]//args.patch_size)**2]) 
                #D = torch.from_numpy(D).float().to(device)
                _c = classifier(Z).squeeze(1)
                #_z = _z.view(_z.shape[0]//(defaults.SIZE[0]//args.patch_size)**2,
                #                         ((defaults.SIZE[0]//args.patch_size)**2)*args.latent_dim)
                #_c = classifier(_z)#.argmax(dim=-1).type(torch.FloatTensor).to(args.device)
                _labels = _target!=len(defaults.anomalies)
                loss = loss_fn(_c,_labels.to(device, dtype=torch.float))
                loss.backward()
                optimizer.step()

                #_data = supervised_val_dataset.patch(supervised_val_dataset.data)
                #_labels = supervised_val_dataset.labels==len(defaults.anomalies)
                #_data = _data.float().to(args.device)
                #_z = resnet.embed(_data)
                #d, _ = nln(_z.cpu().detach().numpy(), z_train, 5, args)
                #d = np.mean(d, axis=tuple(range(1,d.ndim)))
                #D = d.reshape([len(d)//int(defaults.SIZE[0]//args.patch_size)**2 , int(defaults.SIZE[0]//args.patch_size)**2]) 

                #d = torch.from_numpy(D).float().to(device)
                ##_z = _z.view(_z.shape[0]//(defaults.SIZE[0]//args.patch_size)**2,
                ##                         ((defaults.SIZE[0]//args.patch_size)**2)*args.latent_dim)
                ##_c = classifier(_z)#.argmax(dim=-1).type(torch.FloatTensor).to(args.device)
                #_c = classifier(d).squeeze(1).cpu().detach().numpy()
                #outputs = eval_ssl(resnet, classifier, test_dataloader, z_train)
                tepoch.set_postfix(epoch=epoch,
                                   total_loss=loss.item())
            outputs = eval_ssl(resnet, classifier, test_dataloader, z_train)
            resnet.train()
            classifier.train()
            precision, recall, thresholds = precision_recall_curve(test_dataloader.dataset.labels!=len(defaults.anomalies), outputs)
            f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
            acc = np.max(f1_scores)#auc(recall, precision)
            print(epoch, auc(recall, precision), acc)


    classifier.save(args.model_name)
    outputs = eval_ssl(resnet, classifier, test_dataloader, z_train)

    return outputs 

def get_error_nln(supervised_train_dataloader, 
                  supervised_val_dataset, 
                  train_dataloader, 
                  test_dataloader):
    """
        Separates anomalies from normal behaviour 
    """

    resnet = models.ResNet(out_dims=8,in_channels=4, latent_dim=8)
    resnet.load_state_dict(torch.load(f'outputs/position_classifier/{args.model_name}/resnet.pt'))
    resnet.to(args.device)
    resnet.eval()

    z_train = []
    train_dataloader.dataset.set_supervision(False)
    for _data, _target,_,_,_,_ in train_dataloader:
        _data = combine(_data,0,2).float().to(args.device)
        z = resnet.embed(_data)
        z_train.append(z.cpu().detach().float().numpy())
    z_train = np.vstack(z_train)


    anomaly = -1 
    test_dataloader.dataset.set_supervision(False)
    z_test = []
    for _data, _target,_,_,_,_ in test_dataloader:
        _data = combine(_data,0,2).float().to(args.device)
        _z = resnet.embed(_data)
        z_test.append(_z.cpu().detach().numpy())
    z_test = np.vstack(z_test)

    d, _ = nln(z_test, z_train, 5, args)
    #outputs = integrate(d, args)

    return d#outputs 

def eval_ssl(resnet, classifier, test_dataloader, z_train): 
    anomaly = -1 
    test_dataloader.dataset.set_supervision(False)
    resnet.eval()
    classifier.eval()
    outputs = []

    for _data, _target,_,_,_,_ in test_dataloader:
        _data = combine(_data,0,2).float().to(args.device)
        _target = combine(_target,0,2).float().to(args.device)

        _z = resnet.embed(_data)
        Z = _z.reshape([len(_z)//int(defaults.SIZE[0]//args.patch_size)**2, 8*int(defaults.SIZE[0]//args.patch_size)**2]) 
        #d, _ = nln(_z.cpu().detach().numpy(), z_train, 5, args)
        #d = np.mean(d, axis=tuple(range(1,d.ndim)))
        #D = d.reshape([len(d)//int(defaults.SIZE[0]//args.patch_size)**2, int(defaults.SIZE[0]//args.patch_size)**2])
        #d = torch.from_numpy(D).float().to(device)
        _c = classifier(Z).squeeze(1)
        #_z = _z.view(_z.shape[0]//(defaults.SIZE[0]//args.patch_size)**2,
        #                         ((defaults.SIZE[0]//args.patch_size)**2)*args.latent_dim)
        #_c = classifier(_z)#.argmax(dim=-1).type(torch.FloatTensor).to(args.device)
        outputs.append(np.expand_dims(_c.cpu().detach().numpy(),-1))

    outputs = np.vstack(outputs)[:,0]
    classifier.train()
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
    
    ax.set_xticks(pos, list(results.keys()), rotation = 10)
    
    plt.legend()
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'/tmp/temp_{remove}',dpi=300)
    plt.close('all')

def main():
    for _class in defaults.anomalies:#_class in ['', 'None']:#
        (train_dataset, 
        val_dataset, 
        test_dataset, 
        supervised_train_dataset, 
        supervised_val_dataset) = get_data(args, remove=_class)   

        supervised_train_dataloader = DataLoader(supervised_train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)

        # train model 
        test_dataset.set_seed(342)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False)
        test_dataloader.dataset.set_anomaly_mask(-1)

        resnet = ResNet(in_channels = 4, out_dims= len(defaults.anomalies)+1)
        resnet = train(supervised_train_dataloader, supervised_val_dataset, resnet, _class)
        resnet.load_state_dict(torch.load(f'_test/outputs/resnet/resnet_{_class}.pt'))
        
        # function that separates anomalies from non-anomnalies using the SSL method
        # we need to tune treshold so that all anmolies are detected
        # returns updated test_dataloader
        # For the null class
        results = {
                    'normal':{'sup':[],'unsup':[]},
                    'anomaly':{'sup':[], 'unsup':[]},
                    'oscillating_tile':{'sup':[],'unsup':[]}, #[],
                    #'real_high_noise':{'sup':[],'unsup':[]},
                    'first_order_high_noise':{'sup':[],'unsup':[]},
                    'first_order_data_loss':{'sup':[],'unsup':[]},
                    #'third_order_data_loss':{'sup':[],'unsup':[]},
                    'lightning':{'sup':[],'unsup':[]},
                    #'strong_radio_emitter':{'sup':[],'unsup':[]},
                    'galactic_plane':{'sup':[],'unsup':[]},
                    'source_in_sidelobes':{'sup':[],'unsup':[]},
                    'ionosphere':{'sup':[],'unsup':[]},
                    'solar_storm':{'sup':[],'unsup':[]},
                    }
        for seed in [1,10,42, 98, 102,4, 2, 101, 33]:
            print('')
            test_dataset.set_seed(seed)
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False)
            test_dataloader.dataset.set_anomaly_mask(-1)
            test_dataloader.dataset.remove_sources(np.load('_test/excluded_sources.npy')) 
            mask = unsup_detector(supervised_train_dataloader,supervised_val_dataset, test_dataloader) 
            results = eval(resnet, test_dataloader, results, mask)
        plot(results, _class)
        np.save(f'_test/results_{_class}', results)
        


if __name__ == '__main__':
    main()
