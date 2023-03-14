import torch
import faiss
import numpy as np
import gc
import copy
from torch.utils.data import DataLoader
from models import VAE, ResNet, ClassificationHead
from utils import args
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from utils.reporting import save_results
from utils.vis import io, nln_io
from data import get_data
from utils.data import defaults, combine, reconstruct, reconstruct_distances

def nln(z_test:np.array, 
        z_train:np.array, 
        N:int, 
        args:args,
        x_hat:np.array=None)->(np.array, 
                     np.array, 
                     np.array):
    """
        Performs NLN calculation
        
        Parameters
        ----------
        z_test: the vae projection of the test data
        z_train: the vae projection of the train data
        x_hat: the vae reconstruction of training data (optional)
        args: utils.args
        N: number of neigbours
        verbose: prints tqdm output

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

def compute_metrics(labels:np.array, pred:np.array, anomaly:int=0, multiclass=False)->(float, float, float):
    """
        Computes AUROC, AUPRC and F1 for integrated data

        Parameters
        ----------
        labels: labels from dataset
        pred: the integrated prediction from model
        multiclass: indicates if we are doing multiclass classifcation
        anomaly: used for multiclass classifcation

        Returns
        -------
        AUROC: auroc  
        AUPRC: auprc
        F1-Score: Optimal f1-score based on auprc
    """
    assert len(labels) == len(pred), "The length of predictions != length of labels"

    if multiclass:
        _ground_truth = [l == anomaly for l in labels]
    else:
        _ground_truth = [l != len(defaults.anomalies) for l in labels]

    fpr,tpr, thr = roc_curve(_ground_truth, pred)
    auroc = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(_ground_truth, pred)
    auprc = auc(recall, precision)

    f1_scores = np.nan_to_num(2*recall*precision/(recall+precision))
    f1_score = np.max(f1_scores)

    return (auroc,auprc,f1_score)

def eval_vae(vae:VAE, 
            train_dataloader: DataLoader, 
            test_dataloader: DataLoader,
            args:args, 
            error:str="nln")->None:
    """
        Computes AUROC, AUPRC and F1 for VAE for multiple calculation types 
        and writes them to file

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
                D, I = nln(z_test, z_train, N, args)
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

def eval_resnet(resnet:ResNet, 
                train_dataloader: DataLoader, 
                test_dataloader: DataLoader, 
                args:args, 
                plot:bool=True,
                error:str="nln")->None:
    """
        Computes AUROC, AUPRC and F1 for Resnet for multiple calculation types 
        and writes them to file

        Parameters
        ----------
        resnet : trained Resnet on the cpu
        train_dataloader: Dataloader for the training data
        test_dataloader: Dataloader for the test data
        args: args
        error: calculation method used for Resnet ~ ('nln')

        Returns
        -------
        None
    """
    resnet.to(args.device, dtype=torch.bfloat16)
    resnet.eval()
    
    z_train, x_train = [],[]
    for _data, _target, _freq, _station, _context,_  in train_dataloader:
        _data = combine(_data,0,2).float().to(args.device, dtype=torch.bfloat16)
        z = resnet.embed(_data)
        z_train.append(z.float().cpu().detach().numpy())
        x_train.append(_data.float().cpu().detach().numpy())

    z_train = np.vstack(z_train) 
    x_train = np.vstack(x_train) 
    
    anomalies = list(np.arange(len(defaults.anomalies)))
    anomalies.append(-1)
    for anomaly in anomalies:
        gc.collect()
        z_test, x_test = [], []
        print(f'anomaly={anomaly}')
        test_dataloader.dataset.set_anomaly_mask(anomaly)

        for _data, _target, _freq, _station, _context,_ in test_dataloader:
            _data = combine(_data,0,2).float().to(args.device, dtype=torch.bfloat16)
            z = resnet.embed(_data)
            z_test.append(z.float().cpu().detach().numpy())
            x_test.append(_data.float().cpu().detach().numpy())
        z_test, x_test = np.vstack(z_test), np.vstack(x_test)

        N = int(np.max(args.neighbours))
        _D, _I = nln(z_test, z_train, N, args)

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

            nln_io(5, 
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
