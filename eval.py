import torch
import faiss
import numpy as np
import gc
import copy
from torch.utils.data import DataLoader
from models import VAE, BackBone, ClassificationHead, Decoder
from utils import args
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from utils.reporting import save_results
from utils.vis import io, knn_io
from data import get_data
from utils.data import defaults, combine, reconstruct, reconstruct_distances

def knn(z_test:np.array, 
        z_train:np.array, 
        N:int, 
        args:args,
        x_hat:np.array=None)->(np.array, 
                     np.array, 
                     np.array):
    """
        Performs KNN Lookup
        
        Parameters
        ----------
        z_test: the vae projection of the test data
        z_train: the vae projection of the train data
        x_hat: the vae reconstruction of training data (optional)
        args: utils.args
        N: number of neigbours

        Returns
        -------
        recon_neighbours: reconstructed neighbours if x_hat supplied
        dists: distance to neighbours 
        indx: indices of z_train that correspond to the neighbours 
    """

    torch.cuda.empty_cache() 

    index_flat = faiss.IndexFlatL2(z_train.shape[-1])

    res = faiss.StandardGpuResources()
    index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    
    index_flat.add(z_train.astype('float32'))         # add vectors to the index
    D, I = index_flat.search(z_test.astype('float32'), N)  # actual search
    
    if x_hat is not None:
        return x_hat[I], D, I
    else:
        return D, I

def integrate(error:np.array, args:args)->np.array:
    """
        Reassembles the error vectors to the same resolution as the input

        Parameters
        ----------
        error: error metric
        args: utils.args

        Returns
        -------
        dists: re-assembled distances

    """
    dists = np.mean(error, axis = tuple(range(1,error.ndim)))
    n_patches = int(defaults.SIZE[0]//args.patch_size)
    dists = dists.reshape(len(error[::n_patches**2]),
            n_patches,
            n_patches)
    dists = np.mean(dists, axis = tuple(range(1,dists.ndim)))
    return dists

def compute_metrics(targets:np.array, 
                    predictions:np.array, 
                    beta:int=2,
                    multiclass=False)->(list, list):
    """
        Computes AUROC, AUPRC and F-beta for integrated data

        Parameters
        ----------
        labels: labels from dataset
        pred: the integrated prediction from model
        beta: the beta score 
        anomaly: used for multiclass classifcation
        multiclass: indicates if we are doing multiclass classifcation

        Returns
        -------
        AUPRC: auprc
        F-beta: Optimal f-beta-score based on auprc
    """
    assert len(targets) == len(predictions), "The length of predictions != length of targets"
    auprcs, f_scores = [], []

    if multiclass:
        anomalies = copy.deepcopy(defaults.anomalies)
        anomalies.append('normal')
        
        for encoding, anomaly in enumerate(anomalies):
            precision, recall, thresholds = precision_recall_curve(targets==encoding, 
                    predictions==encoding)
            auprcs.append(auc(recall, precision))

            f_betas = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
            f_scores.append(np.max(f_betas))
    else:

        precision, recall, thresholds = precision_recall_curve(targets!=len(defaults.anomalies), 
                predictions!=len(defaults.anomalies))
        auprc.append(auc(recall, precision))

        f_betas = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
        f_scores.append(np.max(f_betas))

    return auprcs, f_scores

def eval_vae(vae:VAE, 
            train_dataloader: DataLoader, 
            test_dataloader: DataLoader,
            args:args)->None:
    """
        Computes AUPRC and F-beta for VAE for multiple calculation types 

        Parameters
        ----------
        vae: trained VAE on the cpu
        train_dataloader: Dataloader for the training data
        test_dataloader: Dataloader for the training data
        args: args
        error: calculation method used for VAE ~ ('nln','recon')

        Returns
        -------
        None
    """
    vae.to(args.device, dtype=torch.bfloat16)
    vae.eval()
    
    z_train = []
    x_hat_train = []
    for _data, _target, _freq, _station, _context,_ in train_dataloader:
        _data = _data.float().to(args.device, dtype=torch.bfloat16)
        [_decoded, _input, _mu, _log_var] = vae(_data)
        z_train.append(vae.reparameterize(_mu, _log_var).cpu().detach().numpy())
        x_hat_train.append(_decoded.cpu().detach().numpy())

    z_train = np.vstack(z_train) 
    x_hat_train = np.vstack(x_hat_train)
    
    defaults.anomalies.append(-1)
    for anomaly in defaults.anomalies:
        gc.collect()
        z_test= []
        x_hat_test = []
        test_dataloader.dataset.set_anomaly_mask(anomaly)

        for _data, _target, _freq, _station, _context, _ in test_dataloader:
            _data = combine(_data,0,2).float().to(args.device, dtype=torch.bfloat16)
            [_decoded, _input, _mu, _log_var] = vae(_data)
            z_test.append(vae.reparameterize(_mu, _log_var).cpu().detach().numpy())
            x_hat_test.append(_decoded.cpu().detach().numpy())
        z_test = np.vstack(z_test)
        x_hat_test = np.vstack(x_hat_test)

        if error == 'nln':
            for N in args.neighbours:
                # build a flat (CPU) index
                D, I = knn(z_test, z_train, N, args)
                dists = integrate(D, args)
                auroc, auprc, f1 = compute_metrics(test_dataloader.dataset.labels,
                                                    dists)

                print("N:{}, AUROC: {:.4f}, AUPRC: {:.4f}, F1: {:.4f}".format(N, auroc, auprc, f1))

                save_results(args, 
                        anomaly=anomaly,
                        epoch=args.epochs,
                        neighbour=N,
                        auroc=auroc, 
                        auprc=auprc, 
                        f1_score=f1)

        elif error == 'recon':
            error = (x_hat_test - test_dataloader.dataset.data.numpy())**2
            error = integrate(error, args)
            auroc, auprc, f1 = compute_metrics(test_dataloader.dataset.labels,
                                                error)

            print("AUROC: {:.4f}, AUPRC: {:.4f}, F1: {:.4f}".format(auroc, auprc, f1))

            save_results(args, 
                    anomaly=anomaly,
                    neighbour=None,
                    auroc=auroc, 
                    auprc=auprc, 
                    f1_score=f1)

def eval_supervised(backbone:BackBone, 
                    test_dataloader: DataLoader, 
                    args:args, 
                    plot:bool=True)->None:
    """
        Computes AUPRC and F-beta for Supervised model for multiple calculation types 

        Parameters
        ----------
        backbone: Backbone 
        test_dataloader: Dataloader for the test data
        args: args

        Returns
        -------
        None
    """
    backbone.to(args.device, dtype=torch.bfloat16)
    backbone.eval()
    test_dataloader.dataset.set_supervision(True)
    
    targets, predictions = [], []
    for _data, _target, _ in test_dataloader:
        _data = _data.to(args.device, dtype=torch.bfloat16)
        c = backbone(_data)
        c = c.argmax(dim=-1).cpu().detach()

        predictions.extend(c.numpy().flatten())
        targets.extend(_target.numpy().flatten())
        
    predictions, targets = np.array(predictions), np.array(targets)
    auprcs, f_scores = compute_metrics(targets, predictions, multiclass=True)
    anomalies = copy.deepcopy(defaults.anomalies)
    anomalies.append('normal')

    for i,anomaly in enumerate(anomalies):
        save_results(args, 
                anomaly=anomaly,
                epoch=args.epochs,
                neighbour=-1,
                beta=2, 
                error_type='None',
                auprc=auprcs[i], 
                f_score=f_scores[i])
    return predictions
def eval_ssl():
    if plot:
        x_recon = reconstruct(x_test, args)

        neighbours = x_train[_I]
        neighbours_recon = np.stack([reconstruct(neighbours[:,i,:],args) 
                                    for i in range(neighbours.shape[1])])
        D = np.stack([_D[:,i].reshape(len(_D[:,i])//int(defaults.SIZE[0]//args.patch_size)**2 ,
                                                   int(defaults.SIZE[0]//args.patch_size),
                                                   int(defaults.SIZE[0]//args.patch_size))
                                for i in range(neighbours.shape[1])])

        _min, _max = np.percentile(D, [1,99])
        D = np.clip(D,_min, _max)
        D = ((D - D.min()) / (D.max() - D.min()))

        knn_io(5, 
            x_recon, 
            neighbours_recon,
            test_dataloader.dataset.labels,
            D,
            'outputs/{}/{}'.format(args.model, args.model_name),
            args.epochs, 
            anomaly) 

        for n in range(1,N+1):
            # build a flat (CPU) index
            D,I = _D[:,:n], _I[:,:n]
            dists = integrate(D,  args)

            auroc, auprc, f1 = compute_metrics(test_dataloader.dataset.labels, dists)

            print("Epoch {}: N:{}, AUROC: {:.4f}, AUPRC: {:.4f}, F1: {:.4f}".format(args.epochs,
                                                                                    n,
                                                                                    auroc,
                                                                                    auprc,
                                                                                    f1))

            save_results(args, 
                    anomaly=anomaly,
                    epoch=args.epochs,
                    neighbour=n,
                    auroc=auroc, 
                    auprc=auprc, 
                    f1_score=f1)
