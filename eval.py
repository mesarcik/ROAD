import torch
import faiss
from torch.utils.data import DataLoader
from models import VAE
from utils import args
import numpy as np
from sklearn.metrics import roc_curve,precision_recall_curve, auc
from utils.reporting import save_results
from data import get_data

def nln(z_test:np.array, 
        z_train:np.array, 
        x_hat:np.array,
        N:int, 
        args:args)->(np.array, 
                     np.array, 
                     np.array):
    """
        Performs NLN calculation
        
        Parameters
        ----------
        z_test: the vae projection of the test data
        z_train: the vae projection of the train data
        x_hat: the vae reconstruction of training data
        args: utils.args
        N: number of neigbours
        verbose: prints tqdm output

        Returns
        -------
        recon_neighbours: reconstructed neighbours
        dists: distance to neighbours 
        indx: indices of z_train that correspond to the neighbours 
    """

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(args.latent_dim)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(z_train)         # add vectors to the index
    D, I = gpu_index_flat.search(z_test, N)  # actual search

    return x_hat[I], D, I

def integrate_dists(D:np.array, test_dims: tuple, args:args)->np.array:
    """
        Reassembles the distance vectors to the same resolution as the input

        Parameters
        ----------
        D: KNN distances
        test_dims: dimensions of test data
        args: utils.args

        Returns
        -------
        dists: re-assembled distances

    """
    dists = np.mean(D, axis = tuple(range(1,D.ndim)))
    n_patches = int(256//args.patch_size)
    dists = dists.reshape(test_dims[0], 
            n_patches,
            n_patches)
    dists = np.mean(dists, axis = tuple(range(1,dists.ndim)))
    return dists

def compute_metrics(labels:np.array, anomaly:str, pred:np.array)->(float, float, float):
    """
        Computes AUROC, AUPRC and F1 for integrated data

        Parameters
        ----------
        labels: labels from dataset
        anomaly: anomalous class
        pred: the integrated prediction from model

        Returns
        -------
        AUROC: auroc  
        AUPRC: auprc
        F1-Score: Optimal f1-score based on auprc
    """
    assert len(labels) == len(pred), "The length of predictions != length of labels"

    _ground_truth = [l == anomaly for l in labels]

    fpr,tpr, thr = roc_curve(_ground_truth, pred)
    auroc = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(_ground_truth, pred)
    auprc = auc(recall, precision)

    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    f1_score = np.max(f1_scores)

    return (auroc,aurpc,f1_score)

def eval_vae(vae:VAE, train_dataloader: DataLoader, args:args)->None:
    vae.to(args.device)
    vae.eval()
    
    z_train = []
    x_hat = []
    for data_, target_ in train_dataloader:
        data_ = data_.float().to(args.device)
        [_decoded, _input, _mu, _log_var] = vae(data_)
        z_train.append(vae.reparameterize(_mu, _log_var).cpu().detach().numpy())
        x_hat.append(_decoded.cpu().detach().numpy())

    z_train = np.vstack(z_train) 
    x_hat = np.vstack(x_hat)

    for anomaly in ['oscillating_tile', 'electric_fence','data_loss', 'lightning','strong_radio_emitter']:
        z_test= []
        _, test_dataset = get_data(args,anomaly=anomaly)# TODO make this more elegant
        test_dataloader = DataLoader(test_dataset, 
                batch_size=args.batch_size, 
                shuffle=False)

        for data_, target_ in test_dataloader:
            data_ = data_.float().to(args.device)
            [_decoded, _input, _mu, _log_var] = vae(data_)
            z_test.append(vae.reparameterize(_mu, _log_var).cpu().detach().numpy())
        z_test = np.vstack(z_test)

        for N in args.neighbours:
            # build a flat (CPU) index
            neighbours, D, I = nln(z_test, z_train, x_hat, N, args)
            dists = integrate_dists(D, test_dataloader.dataset.original_shape, args)

            auroc, auprc, f1 = compute_metrics(test_dataloader.dataset.labels, 
                                                anomaly, 
                                                dists)

            print("N:{}, AUROC: {:.4f}, AUPRC: {:.4f}, F1: {:.4f}".format(N, auroc, auprc, f1_score))

            save_results(args, 
                    anomaly=anomaly,
                    neighbour=N,
                    auroc=auroc, 
                    auprc=auprc, 
                    f1_score=f1_score)

