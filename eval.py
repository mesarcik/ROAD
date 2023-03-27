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
        N:int)->(np.array,np.array):
    """
        Performs KNN Lookup
        
        Parameters
        ----------
        z_test: the vae projection of the test data
        z_train: the vae projection of the train data
        N: number of neigbours

        Returns
        -------
        dists: distance to neighbours 
        indx: indices of z_train that correspond to the neighbours 
    """

    torch.cuda.empty_cache() 

    index_flat = faiss.IndexFlatL2(z_train.shape[-1])

    res = faiss.StandardGpuResources()
    index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    
    index_flat.add(z_train.astype('float32'))         # add vectors to the index
    D, I = index_flat.search(z_test.astype('float32'), N)  # actual search
    
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
        tholds: Thresholds corresponding to optimal score
    """
    assert len(targets) == len(predictions), "The length of predictions != length of targets"
    auprcs, f_scores, tholds = [], [], []

    if multiclass:
        anomalies = copy.deepcopy(defaults.anomalies)
        anomalies.append('normal')
        
        for encoding, anomaly in enumerate(anomalies):
            precision, recall, thresholds = precision_recall_curve(targets==encoding, 
                    predictions==encoding)
            auprcs.append(auc(recall, precision))

            f_betas = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
            f_scores.append(np.max(f_betas))
            tholds.append(thresholds[np.argmax(f_betas)])
    else:

        precision, recall, thresholds = precision_recall_curve(targets!=len(defaults.anomalies), 
                predictions)
        auprcs.append(auc(recall, precision))

        f_betas = np.nan_to_num((1+beta**2)*recall*precision/((beta**2)*recall+precision))
        f_scores.append(np.max(f_betas))
        tholds.append(thresholds[np.argmax(f_betas)])

    return auprcs, f_scores, tholds

def combine_predictions(supervised_pred:np.array,
                        ssl_pred:np.array,
                        threshold:np.array,
                        )->np.array:
    """
        Combines SSL and supervised predictions 

        Parameters
        ----------
        supervised_pred: multiclass predictions from the supervised classifier
        ssl_pred: binary predictions from the ssl detector
        threshold: maximum f-beta score threshold

        Returns
        -------
        combined_predictions: ...

    """
    unknown_anomaly = len(defaults.anomalies)+1
    combined_predictions = []

    for sup, ssl in zip(supervised_pred, ssl_pred):
        if ssl<threshold: # mask is true for anomalies 
            combined_predictions.append(len(defaults.anomalies))
        elif ssl>=threshold and sup==len(defaults.anomalies):
            combined_predictions.append(unknown_anomaly)
        elif ssl>=threshold and sup!= len(defaults.anomalies):
            combined_predictions.append(sup)
    return np.array(combined_predictions)

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
                D, I = knn(z_test, z_train, N)
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
                    ssl_predictions: np.array=None,
                    ssl_threshold: np.array=None,
                    plot:bool=True)->(np.array, np.array):
    """
        Computes AUPRC and F-beta for Supervised model for multiple calculation types 

        Parameters
        ----------
        backbone: Backbone 
        test_dataloader: Dataloader for the test data
        args: args
        ssl_predictions: optional argument used to modify the supervised predictions
        ssl_tholds: optional argument used to modify the supervised predictions

        Returns
        -------
        predictions: predictions
        thresholds: thresholds 
    """
    output_label = 'supervisd'
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
    if ssl_predictions is not None and ssl_threshold is not None:
        predictions = combine_predictions(predictions, ssl_predictions, ssl_threshold)
        output_label = 'ssl'


    auprcs, f_scores, tholds = compute_metrics(targets, predictions, multiclass=True)
    anomalies = copy.deepcopy(defaults.anomalies)
    anomalies.append('normal')

    for i,anomaly in enumerate(anomalies):
        save_results(args, 
                anomaly=anomaly,
                epoch=args.epochs,
                neighbour=-1,
                beta=2, 
                error_type=output_label,
                auprc=auprcs[i], 
                f_score=f_scores[i])
    return predictions, tholds

def eval_classification_head(backbone:BackBone, 
                            classification_head: ClassificationHead,
                            test_dataloader: DataLoader, 
                            args:args, 
                            plot:bool=True)->(np.array, np.array):
    """
        Computes AUPRC and F-beta for classification head for fine tuned classification head

        Parameters
        ----------
        backbone: Backbone 
        classification_head: ClassificationHead
        test_dataloader: Dataloader for the test data
        args: args

        Returns
        -------
        predictions: boolean predictions
    """
    backbone.to(args.device, dtype=torch.bfloat16)
    backbone.eval()

    classification_head.to(args.device, dtype=torch.bfloat16)
    classification_head.eval()

    test_dataloader.dataset.set_supervision(False)
    
    targets, predictions = [], []
    for _data, _target, _ ,_ in test_dataloader:
        _data = combine(_data,0,2).float().to(args.device, dtype=torch.bfloat16)
        _z = backbone(_data)
        Z = _z.reshape([len(_z)//int(defaults.SIZE[0]//args.patch_size)**2, 
                        args.latent_dim*int(defaults.SIZE[0]//args.patch_size)**2])
        c = classification_head(Z)
        c = c.squeeze(1).cpu().detach()

        predictions.extend(c.float().numpy().flatten())
        targets.extend(_target[:,0].numpy().flatten())
        
    predictions, targets = np.array(predictions), np.array(targets)
    auprcs, f_scores, tholds = compute_metrics(targets, predictions, multiclass=False)
    print("AUPRC: {:.4f}, F2: {:.4f}".format(auprcs[0],f_scores[0]))
    save_results(args, 
            anomaly='-1',
            epoch=args.epochs,
            neighbour=-1,
            beta=2, 
            error_type='fine_tuning',
            auprc=auprcs[0], 
            f_score=f_scores[0])

    return predictions, tholds

def eval_knn(backbone: BackBone,
             decoder:Decoder,
             test_dataloader: DataLoader, 
             train_dataloader: DataLoader, 
             args:args, 
             plot:bool=True)->(np.array, np.array):
    """
        Computes AUPRC and F-beta for Backbone using KNN-based methods

        Parameters
        ----------
        backbone: Backbone 
        decoder: Decoder
        test_dataloader: Dataloader for the test data
        train_dataloader: Dataloder for the test data
        args: args

        Returns
        -------
        predictions: boolean predictions
    """
    if args.model == 'vae':
        backbone.to(args.device, dtype=torch.bfloat16)
        backbone.eval()

    else:
        backbone.to(args.device, dtype=torch.bfloat16)
        decoder.to(args.device, dtype=torch.bfloat16)
        backbone.eval()
        decoder.eval()

    test_dataloader.dataset.set_supervision(False)
    train_dataloader.dataset.set_supervision(False)
    
    z_train, x_train, x_hat = [],[],[]
    for _data, _target, _ ,_ in train_dataloader:
        _data = combine(_data,0,2).float().to(args.device, dtype=torch.bfloat16)
        if args.model == 'vae':
            [_x_hat, _input, _mu, _log_var] = backbone(_data)
            _z = backbone.reparameterize(_mu, _log_var)
        else:
            _z = backbone(_data)
            _x_hat = decoder(_z)

        z_train.append(_z.float().cpu().detach().numpy())
        x_train.append(_data.float().cpu().detach().numpy())
        x_hat.append(_x_hat.float().cpu().detach().numpy())

    z_train = np.vstack(z_train)
    x_train = np.vstack(x_train)
    x_hat = np.vstack(x_hat)

    anomalies = list(np.arange(len(defaults.anomalies)))
    anomalies.append(-1)
    for anomaly in anomalies:
        z_test, x_test, targets = [], [], []
        test_dataloader.dataset.set_anomaly_mask(anomaly)
        for _data, _target, _ ,_ in test_dataloader:
            _data = combine(_data,0,2).float().to(args.device, dtype=torch.bfloat16)
            if args.model == 'vae':
                [_x_hat, _input, _mu, _log_var] = backbone(_data)
                _z = backbone.reparameterize(_mu, _log_var)
            else:
                _z = backbone(_data)

            z_test.append(_z.float().cpu().detach().numpy())
            x_test.append(_data.float().cpu().detach().numpy())
            targets.extend(_target[:,0].numpy().flatten())

        z_test, x_test = np.vstack(z_test), np.vstack(x_test)
        targets = np.array(targets)

        N = int(np.max(args.neighbours))
        _D, _I = knn(z_test, z_train, N)
        x_hat_neigh = x_hat[_I]

        knn_io(5, 
            x_test, 
            x_train, 
            x_hat_neigh,
            _D,
            _I,
            test_dataloader.dataset.labels,
            'outputs/{}/{}'.format(args.model, args.model_name),
            args.epochs, 
            args,
            anomaly=anomaly) 

            
        D,I = _D[:,:N], _I[:,:N]
        dists = integrate(D, args)
        mse = np.stack([ np.absolute(x_test - x_hat_neigh[:,n,...]) for n in range(N)], axis=-1)
        _mse = np.mean(mse, axis = tuple(range(1,mse.ndim)))
        mse = integrate(mse, args)

        knn_auprcs, knn_f_scores, _ = compute_metrics(targets, dists, multiclass=False)
        print("N:{}, NLN_AUPRC: {:.4f}, NLN_F2: {:.4f}".format(N,knn_auprcs[0],knn_f_scores[0]))
        save_results(args, 
                anomaly=anomaly,
                epoch=args.epochs,
                neighbour=N,
                beta=2, 
                error_type='knn',
                auprc=knn_auprcs[0], 
                f_score=knn_f_scores[0])
        nln_auprcs, nln_f_scores, _ = compute_metrics(targets, mse, multiclass=False)
        print("N:{}, KNN_AUPRC: {:.4f}, KNN_F2: {:.4f}".format(N,nln_auprcs[0],nln_f_scores[0]))
        save_results(args, 
                anomaly=anomaly,
                epoch=args.epochs,
                neighbour=N,
                beta=2, 
                error_type='nln',
                auprc=nln_auprcs[0], 
                f_score=nln_f_scores[0])

        comb = 0.5*dists + 0.5*mse
        auprcs, f_scores, tholds  = compute_metrics(targets, comb, multiclass=False)
        print("N:{}, SUM_AUPRC: {:.4f}, SUM_F2: {:.4f}".format(N,auprcs[0],f_scores[0]))
    return comb, tholds[0]


