import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch import nn

from utils.vis import imscatter, io, loss_curve
from utils.data import SIZE
from eval import eval_finetune

def fine_tune(
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        resnet,
        classification_head,
        args):
    """
        Fine tunes classification head

        Parameters
        ----------
        train_dataloader: dataloader
        val_dataset: validation dataset
        resnet:resnet
        classification_head: classifcaiotn head
        args: runtime arguments

        Returns
        -------
        classification_head: trained classification_head

    """
    model_path = 'outputs/{}/{}'.format(args.model,
                                           args.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    resnet.eval()
    classification_head.train()
    resnet.to(args.device)
    classification_head.to(args.device)

    classification_head_optimizer = torch.optim.Adam(
        classification_head.parameters(),
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)

    total_train_loss = []
    accuracies= []
    total_step = len(train_dataloader)


    for epoch in range(1, 101):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss = 0.0
            running_acc = 0.0
            for _data, _target, _, _, _, _ in tepoch:
                if _data.shape[0] != args.batch_size: continue
                tepoch.set_description(f"Epoch {epoch}")

                _target = _target.type(torch.LongTensor).to(args.device)
                _data = _data.type(torch.FloatTensor).to(args.device)

                classification_head_optimizer.zero_grad()

                _z = resnet.embed(_data)
                _z = _z.view(_z.shape[0]//(SIZE[0]//args.patch_size)**2,
                                ((SIZE[0]//args.patch_size)**2)*args.latent_dim)
                _c = classification_head(_z)#.argmax(dim=-1).type(torch.FloatTensor).to(args.device)
                _target = _target[::(SIZE[0]//args.patch_size)**2]

                loss = classification_head.loss_fn(_c, _target)

                loss.backward()
                classification_head_optimizer.step()
                running_loss += loss.item()

                val_acc = torch.sum(
                    _c.argmax(
                        dim=-1) == _target) / _target.shape[0]
                running_acc += val_acc.cpu().detach()

                tepoch.set_postfix(total_loss=loss.item(), 
                                   train_accuracy=val_acc.item())

            total_train_loss.append(running_loss / total_step)

            accuracies.append(running_acc/ total_step)

            if epoch % 10 == 0:  # TODO: check for model improvement
                torch.save(
                    classification_head.state_dict(),
                    '{}/classification_head.pt'.format(model_path))

                with open('{}/model.config'.format(model_path), 'w') as fp:
                    for arg in args.__dict__:
                        fp.write('{}: {}\n'.format(arg, args.__dict__[arg]))

            loss_curve(model_path,
                       epoch,
                       total_loss=total_train_loss,
                       train_accuracies=accuracies,
                       descriptor='finetune')

    return classification_head 
