import torch
from torch.utils.data import Dataset, DataLoader
from models import VAE, Decoder, PositionClassifier, BackBone
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
from torch import nn


from utils.args import args
from utils.data import combine
from utils.vis import imscatter, io, loss_curve
from eval import eval_resnet


def train_vae(train_dataloader: DataLoader, vae: VAE, args: args) -> VAE:
    """
        Trains VAE

        Parameters
        ----------
        train_dataloader: dataloader
        vae: VAE
        args: runtime arguments

        Returns
        -------
        data: list of baselines with single channels removed

    """
    model_path = 'outputs/{}/{}'.format(args.model,
                                           args.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    vae.to(args.device, dtype=torch.bfloat16)
    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)
    train_loss = []
    total_step = len(train_dataloader)

    for epoch in range(1, args.epochs + 1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss = 0.0
            for _data, _, _, _, _, _ in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                _data = combine(_data,0,2).float().to(args.device, dtype=torch.bfloat16)
                optimizer.zero_grad()

                [_decoded, _input, _mu, _log_var] = vae(_data)
                z = vae.reparameterize(_mu, _log_var)
                loss = vae.loss_function(_decoded, _input, _mu, _log_var)
                loss['loss'].backward()
                optimizer.step()

                running_loss += loss['loss'].item()
                tepoch.set_postfix(
                    recon_loss=loss['Reconstruction_Loss'].item(),
                    total_loss=loss['loss'].item())
            train_loss.append(running_loss / total_step)

            if epoch % 50 == 0:  # TODO: check for model improvement
                torch.save(vae.state_dict(), '{}/vae.pt'.format(model_path))
                vae.eval()
                mu, log_var = vae.encode(_data)
                Z = vae.reparameterize(mu, log_var).cpu().detach().numpy()
                z = TSNE(n_components=2,
                         learning_rate='auto',
                         init='random',
                         perplexity=30).fit_transform(Z)

                _inputs = _data.cpu().detach().numpy()
                _reconstructions = _decoded.cpu().detach().numpy()
                io(10, _inputs, _reconstructions, model_path, epoch)
                imscatter(z, _inputs, model_path, epoch)
            loss_curve(model_path, epoch, total_loss=train_loss, descriptor='vae')
            vae.train()
    return vae


def train_supervised(
        train_dataloader: DataLoader,
        val_dataset,
        resnet,
        args: args):
    """
        Trainsa Resnet

        Parameters
        ----------
        train_dataloader: dataloader
        val_dataset: validation dataset
        resnet:resnet
        args: runtime arguments

        Returns
        -------
        data: list of baselines with single channels removed

    """
    model_path = 'outputs/{}/{}'.format(args.model,
                                           args.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    resnet.to(args.device, dtype=torch.bfloat16)
    optimizer = torch.optim.Adam(
        resnet.parameters(),
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)
    train_loss, validation_accuracies = [], []
    total_step = len(train_dataloader)

    for epoch in range(1, args.epochs + 1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss, running_acc = 0.0, 0.0
            for _data, _target, _freq, _context, _, _ in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                _data = combine(_data,0,2).float().to(args.device, dtype=torch.bfloat16)
                _out = _context.to(args.device)

                optimizer.zero_grad()

                z = resnet(_data)
                loss = resnet.loss_function(z, _out)['loss']
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _data = val_dataset.data.float().to(args.device, dtype=torch.bfloat16)
                z = resnet(_data).cpu().detach()

                val_acc = torch.sum(
                    z.argmax(
                        dim=-1) == val_dataset.context_labels) / val_dataset.context_labels.shape[0]
                running_acc += val_acc
                tepoch.set_postfix(loss=loss.item(), val_acc=val_acc.item())

            train_loss.append(running_loss / total_step)
            validation_accuracies.append(running_acc/ total_step)

            if epoch % 50 == 0:  # TODO: check for model improvement
                # print validation loss

                torch.save(
                    resnet.state_dict(),
                    '{}/resnet.pt'.format(model_path))
                resnet.eval()
                Z = resnet.embed(_data).cpu().detach().numpy()
                z = TSNE(n_components=2,
                         learning_rate='auto',
                         init='random',
                         perplexity=3).fit_transform(Z)

                _inputs = _data.cpu().detach().numpy()
                imscatter(z, _inputs, model_path, epoch)
            loss_curve(model_path,
                       epoch,
                       total_loss=train_loss,
                       validation_accuracy=validation_accuracies,
                       descriptor='resnet')
            resnet.train()
    return resnet

def train_ssl(
        train_dataloader: DataLoader,
        val_dataset:Dataset,
        backbone:BackBone,
        position_classifier:PositionClassifier,
        decoder:Decoder,
        args: args):
    """
        Trains Position Clasifier

        Parameters
        ----------
        train_dataloader: dataloader
        val_dataset: validation dataset
        backbone: backbone
        classifier: position_classifier
        args: runtime arguments

        Returns
        -------
        model: trained resnet

    """
    model_path = 'outputs/{}/{}'.format(args.model,
                                           args.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    backbone.to(args.device, dtype=torch.bfloat16)
    position_classifier.to(args.device, dtype=bfloat16)
    decoder.to(args.device, dtype=torch.bfloat16)

    backbone_optimizer = torch.optim.Adam(
        resnet.parameters(),
        lr=args.learning_rate)  

    position_classifier_optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate) 

    decoder_optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=args.learning_rate)  

    total_loss, ssl_loss, decoder_loss, regulation_loss  = [], [], [], []
    validation_accuracies = []
    total_step = len(train_dataloader)
    prev_acc = 0

    for epoch in range(1, args.epochs + 1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss, d_loss, l_loss, r_loss   = 0.0, 0.0, 0.0, 0.0
            running_acc = 0.0
            for _data, _target, _context_label, _context_neighbour in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                _data = combine(_data,0,2).float().to(args.device, dtype=torch.bfloat16)
                _context_neighbour = combine(_context_neighbour,0,2).float().to(args.device, dtype=torch.bfloat16)
                _context_label= combine(_context_label,0,2).to(args.device, dtype=torch.long)

                backbone_optimizer.zero_grad()
                position_classifier_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                z_data  = backbone(_data)
                z_neighbour = backbone(_context_neighbour)
                c_pos = position_classifier(z_data, z_neighbour)

                hat_data = decoder(z_data)
                hat_neighbour = decoder(z_neighbour)


                location_loss = position_classifier.loss_function(c_pos, _context_label)
                decoder_loss = decoder.loss_function(_context_neighbour, hat_neighbour) + decoder.loss_function(_data, hat_data)
                reg_loss = 0.01*(torch.sum(torch.square(z_data)) + torch.sum(torch.square(z_neighbour)))

                loss = location_loss + decoder_loss +  reg_loss
                loss.backward()
                backbone_optimizer.step()
                position_classifier_optimizer.step()
                decoder_optimizer.step()

                running_loss += loss.item()
                l_loss += location_loss.item()
                d_loss += decoder_loss.item()
                r_loss += reg_loss.item()

                # Validation 
                _data = val_dataset.patch(val_dataset.data)
                _labels, _neighbour = val_dataset.context_prediction(_data)
                _data = _data.float().to(args.device, dtype=torch.bfloat16)

                _neighbour = _neighbour.float().to(args.device, dtype=torch.bfloat16)
                z_data  = backbone(_data)
                z_neighbour = backbone(_neighbour)

                c_pos  = position_classifier(z_data, z_neighbour).cpu().detach()
                val_acc_context = torch.sum(
                    c_pos.argmax(
                        dim=-1) == _labels) / _labels.shape[0]
                running_acc += val_acc_context

                tepoch.set_postfix(total_loss=loss.item(),
                        location_loss = location_loss.item(),
                        decoder_loss = decoder_loss.item(),
                        regulation_loss = reg_loss.item()
                        location_accuracy=val_acc_context.item())

            total_loss.append(running_loss/total_step)
            ssl_loss.append(l_loss/total_step)
            decoder_loss.append(d_loss / total_step)
            regulation_loss.append(r_loss / total_step)

            validation_accuracies.append(running_acc/ total_step)

            if epoch % 50 == 0:  # TODO: check for model improvement
                # print validation loss
                if val_acc_context>prev_acc:
                    prev_acc = val_acc_context
                    backbone.save(args)
                    decoder.save(args)
                    position_classifier.save(args)

                    with open('{}/model.config'.format(model_path), 'w') as fp:
                        for arg in args.__dict__:
                            fp.write('{}: {}\n'.format(arg, args.__dict__[arg]))

            loss_curve(model_path,
                       epoch,
                       total_loss=total_train_loss,
                       location_loss=encoder_train_loss,
                       #frequency_loss=location_train_loss,
                       validation_accuracy=validation_accuracies,
                       #context_accuracies=context_accuracies,
                       descriptor='position')
            classifier.train()
            resnet.train()

    with open('{}/model.config'.format(model_path), 'w') as fp:
        for arg in args.__dict__:
            fp.write('{}: {}\n'.format(arg, args.__dict__[arg]))

    return backbone, position_classifier, decoder
