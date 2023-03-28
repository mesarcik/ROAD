import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from torch import nn

from utils.vis import imscatter, io, loss_curve
from utils.data import defaults, combine
from models import BackBone, ClassificationHead
from eval import compute_metrics, eval_classification_head

def fine_tune(
        supervised_train_dataloader: DataLoader,
        val_dataset:Dataset,
        test_dataloader:DataLoader,
        backbone:BackBone,
        classification_head:ClassificationHead,
        args):
    """
        Fine tunes classification head

        Parameters
        ----------
        supervised_train_dataloader: dataloader
        val_dataset: supervised validation dataset
        backbone:backbone
        classification_head: classifcation head
        args: runtime arguments

        Returns
        -------
        classification_head: trained classification_head

    """
    model_path = 'outputs/{}/{}'.format(args.model,
                                           args.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    backbone.train()
    classification_head.train()
    backbone.to(args.device, dtype=torch.bfloat16)
    classification_head.to(args.device, dtype=torch.bfloat16)

    optimizer = torch.optim.Adam(
        list(classification_head.parameters())+list(backbone.parameters()),
        lr=1e-4)  # , lr=0.0001, momentum=0.9)

    total_train_loss = []
    accuracies= []
    total_step = len(supervised_train_dataloader)
    supervised_train_dataloader.dataset.set_supervision(False)
    _val_data = val_dataset.patch(val_dataset.data).float().to(args.device, dtype=torch.bfloat16)
    _val_targets = val_dataset.labels !=len(defaults.anomalies)
    prev_acc = 0

    for epoch in range(1, 51):
        with tqdm(supervised_train_dataloader, unit="batch") as tepoch:
            running_loss, running_acc  = 0.0,0.0
            for _data, _target, _, _  in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                _data = combine(_data,0,2).float().to(args.device, dtype=torch.bfloat16)
                _target = _target[:,0].to(args.device, dtype=torch.long)

                optimizer.zero_grad()

                _z = backbone(_data)
                Z = _z.reshape([len(_z)//int(defaults.SIZE[0]//args.patch_size)**2, 
                                args.latent_dim*int(defaults.SIZE[0]//args.patch_size)**2])
                _c = classification_head(Z).squeeze(1)
                _labels = _target!=len(defaults.anomalies)
                loss = classification_head.loss_fn(_c,_labels.to(args.device, dtype=torch.bfloat16))

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                ####
                backbone.eval()
                classification_head.eval()
                _z = backbone(_val_data)
                Z = _z.reshape([len(_z)//int(defaults.SIZE[0]//args.patch_size)**2, 
                                args.latent_dim*int(defaults.SIZE[0]//args.patch_size)**2])
                _c = classification_head(Z).squeeze(1)
                backbone.train()
                classification_head.train()

                auprc, f_score, _ = compute_metrics(_val_targets.numpy(), 
                                                _c.float().cpu().detach().numpy(),
                                                beta=2,
                                                multiclass=False)
                val_acc = f_score[0]
                #val_acc = torch.sum(_c == _labels) / _labels.shape[0]
                running_acc += val_acc

                tepoch.set_postfix(total_loss=loss.item(), 
                                   train_accuracy=val_acc.item())

            total_train_loss.append(running_loss / total_step)
            accuracies.append(running_acc/ total_step)

            #if prev_acc < accuracies[-1]:  # TODO: check for model improvement, validation size too small
            classification_head.save(args)
            backbone.save(args,ft=True)
            prev_acc=accuracies[-1]
            backbone.eval()
            classification_head.eval()
            pred_ft, thr_ft = eval_classification_head(backbone.cpu(), classification_head.cpu(), test_dataloader, args)

            loss_curve(model_path,
                       epoch,
                       total_loss=total_train_loss,
                       train_accuracies=accuracies,
                       descriptor='finetune')
            backbone.train()
            classification_head.train()

    classification_head.load(args)
    backbone.load(args,ft=True)
    return backbone, classification_head 
