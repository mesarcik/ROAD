"""
    Train ROAD models
"""

import os
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models import VAE, Decoder, PositionClassifier, BackBone
from utils.args import args
from utils.data import combine
from utils.vis import imscatter, io, loss_curve


def train_vae(train_dataloader: DataLoader,
              vae: VAE,
              args: args) -> VAE:
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
    model_path = f'{args.model_path}/outputs/models/{args.model_name}'
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
                _data = combine(_data, 0, 2).float().to(args.device,
                                                        dtype=torch.bfloat16)
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
                vae.save(args)
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
            loss_curve(model_path,
                       epoch,
                       total_loss=train_loss,
                       descriptor='vae')
            vae.train()
    return vae


def train_supervised(
        supervised_train_dataloader: DataLoader,
        val_dataset: Dataset,
        backbone: BackBone,
        args: args) -> BackBone:
    """
        Trains Backbone in a supervised fashion

        Parameters
        ----------
        supervised_train_dataloader: dataloader
        val_dataset: validation dataset
        backbone: backbone
        args: runtime arguments

        Returns
        -------
        backbone: backbone

    """
    model_path = f'{args.model_path}/outputs/models/{args.model_name}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    backbone.to(args.device, dtype=torch.bfloat16)

    optimizer = torch.optim.Adam(
        backbone.parameters(),
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)

    train_loss, validation_accuracies = [], []
    total_step = len(supervised_train_dataloader)
    supervised_train_dataloader.dataset.set_supervision(True)

    _val_data = val_dataset.data.float().to(args.device, dtype=torch.bfloat16)
    _val_targets = val_dataset.labels
    prev_acc = 0

    for epoch in range(1, args.epochs + 1):
        with tqdm(supervised_train_dataloader, unit="batch") as tepoch:
            running_loss, running_acc = 0.0, 0.0
            for _data, _target, _source in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                _data = _data.to(args.device, dtype=torch.bfloat16)
                _target = _target.type(torch.LongTensor).to(args.device)

                optimizer.zero_grad()

                c = backbone(_data)
                loss = backbone.loss_function(c, _target)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Validation
                backbone.eval()
                c = backbone(_val_data).cpu().detach()
                val_acc = torch.sum(
                    c.argmax(
                        dim=-1) == _val_targets) / _val_targets.shape[0]
                running_acc += val_acc
                tepoch.set_postfix(loss=loss.item(), val_acc=val_acc.item())
                backbone.train()

            train_loss.append(running_loss / total_step)
            validation_accuracies.append(running_acc/total_step)

            if prev_acc < validation_accuracies[-1]:
                backbone.save(args, 'supervised', False)
                prev_acc = validation_accuracies[-1]

            loss_curve(model_path,
                       epoch,
                       total_loss=train_loss,
                       validation_accuracy=validation_accuracies,
                       descriptor='supervised')
            backbone.train()
    backbone.load(args, 'supervised', False)
    return backbone


def train_ssl(train_dataloader: DataLoader,
              val_dataset: Dataset,
              backbone: BackBone,
              position_classifier: PositionClassifier,
              decoder: Decoder,
              args: args):
    """
        Trains SSL model

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
    model_path = f'{args.model_path}/outputs/models/{args.model_name}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    backbone.to(args.device, dtype=torch.bfloat16)
    position_classifier.to(args.device, dtype=torch.bfloat16)
    decoder.to(args.device, dtype=torch.bfloat16)

    backbone_optimizer = torch.optim.Adam(
        backbone.parameters(),
        lr=args.learning_rate)

    position_classifier_optimizer = torch.optim.Adam(
        position_classifier.parameters(),
        lr=args.learning_rate)

    decoder_optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=args.learning_rate)

    total_loss, position_loss = [], []
    reconstruction_loss, regulation_loss = [], []
    validation_accuracies = []
    total_step = len(train_dataloader)
    prev_acc = 0

    for epoch in range(1, args.epochs + 1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss, d_loss, l_loss, r_loss = 0.0, 0.0, 0.0, 0.0
            running_acc = 0.0
            for _data, _target, _context_label, _context_neighbour in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                _data = combine(_data, 0, 2).float().to(args.device,
                                                        dtype=torch.bfloat16)
                _context_neighbour = combine(_context_neighbour, 0, 2).float()
                _context_neighbour = _context_neighbour.to(args.device,
                                                           dtype=torch.bfloat16)
                _context_label = combine(_context_label, 0, 2)
                _context_label = _context_label.to(args.device, dtype=torch.long)

                backbone_optimizer.zero_grad()
                position_classifier_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                z_data = backbone(_data)
                z_neighbour = backbone(_context_neighbour)
                c_pos = position_classifier(z_data, z_neighbour)

                hat_data = decoder(z_data)
                hat_neighbour = decoder(z_neighbour)

                location_loss = position_classifier.loss_function(c_pos,
                                                                  _context_label)
                decoder_loss = (decoder.loss_function(_context_neighbour,
                                                      hat_neighbour) +
                                decoder.loss_function(_data,
                                                      hat_data))
                reg_loss = 0.00001*(torch.sum(torch.square(z_data)) +
                                    torch.sum(torch.square(z_neighbour)))

                loss = location_loss + decoder_loss + reg_loss
                loss.backward()
                backbone_optimizer.step()
                position_classifier_optimizer.step()
                decoder_optimizer.step()

                running_loss += loss.item()
                l_loss += location_loss.item()
                d_loss += decoder_loss.item()
                r_loss += reg_loss.item()

                # Validation
                backbone.eval()
                decoder.eval()
                position_classifier.eval()
                _data = val_dataset.patch(val_dataset.data)
                _labels, _neighbour = val_dataset.context_prediction(_data)
                _data = _data.float().to(args.device, dtype=torch.bfloat16)

                _neighbour = _neighbour.float()
                _neighbour = _neighbour.to(args.device, dtype=torch.bfloat16)
                z_data = backbone(_data)
                z_neighbour = backbone(_neighbour)

                c_pos = position_classifier(z_data, z_neighbour).cpu().detach()
                val_acc_context = torch.sum(
                    c_pos.argmax(
                        dim=-1) == _labels) / _labels.shape[0]
                running_acc += val_acc_context

                backbone.train()
                decoder.train()
                position_classifier.train()

                tepoch.set_postfix(total_loss=loss.item(),
                                   location_loss=location_loss.item(),
                                   decoder_loss=decoder_loss.item(),
                                   regulation_loss=reg_loss.item(),
                                   location_accuracy=val_acc_context.item())

            total_loss.append(running_loss/total_step)
            position_loss.append(l_loss/total_step)
            reconstruction_loss.append(d_loss / total_step)
            regulation_loss.append(r_loss / total_step)

            validation_accuracies.append(running_acc/total_step)

            if val_acc_context > prev_acc:
                prev_acc = val_acc_context
                backbone.save(args, 'ssl', False)
                decoder.save(args)
                position_classifier.save(args)

            loss_curve(model_path,
                       epoch,
                       total_loss=total_loss,
                       position_loss=position_loss,
                       reconstruciton_loss=reconstruction_loss,
                       regulation_loss=regulation_loss,
                       validation_accuracy=validation_accuracies,
                       descriptor='position')

    with open('{}/model.config'.format(model_path), 'w') as fp:
        for arg in args.__dict__:
            fp.write('{}: {}\n'.format(arg, args.__dict__[arg]))

    backbone.load(args, 'ssl', False)
    position_classifier.load(args)
    decoder.load(args)
    return backbone, position_classifier, decoder
