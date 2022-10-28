import torch
from torch.utils.data import DataLoader
from models import VAE
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
from functools import reduce


from utils.args import args
from utils.vis import imscatter, io, loss_curve


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

    vae.to(args.device)
    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)
    train_loss = []
    total_step = len(train_dataloader)

    for epoch in range(1, args.epochs + 1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss = 0.0
            for _data, _target, _freq, _station, _context in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                _data = _data.float().to(args.device)
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
            batch_loss = 0

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
            loss_curve(model_path, epoch, total_loss=train_loss)
            vae.train()
    return vae


def train_resnet(
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

    resnet.to(args.device)
    optimizer = torch.optim.Adam(
        resnet.parameters(),
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)
    train_loss, validation_accuracies = [], []
    total_step = len(train_dataloader)

    for epoch in range(1, args.epochs + 1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss, running_acc = 0.0, 0.0
            for _data, _target, _freq, _station, _context in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                _data = _data.float().to(args.device)
                _out = _context.to(args.device)

                optimizer.zero_grad()

                z = resnet(_data)
                loss = resnet.loss_function(z, _out)['loss']
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _data = val_dataset.data.float().to(args.device)
                z = resnet(_data).cpu().detach()

                val_acc = torch.sum(
                    z.argmax(
                        dim=-1) == val_dataset.context_labels) / val_dataset.context_labels.shape[0]
                running_acc += val_acc
                tepoch.set_postfix(loss=loss.item(), val_acc=val_acc.item())

            train_loss.append(running_loss / total_step)
            validation_accuracies.append(running_acc/ total_step)
            batch_loss = 0

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
                       validation_accuracy=validation_accuracies)
            resnet.train()
    return resnet

def train_position_classifier(
        train_dataloader: DataLoader,
        val_dataset,
        resnet,
        classifier,
        args: args):
    """
        Trains Position Clasifier

        Parameters
        ----------
        train_dataloader: dataloader
        val_dataset: validation dataset
        resnet:resnet
        classifier: position_classifier
        args: runtime arguments

        Returns
        -------
        resnet: trained resnet

    """
    model_path = 'outputs/{}/{}'.format(args.model,
                                           args.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    resnet.to(args.device)
    classifier.to(args.device)
    modules = [resnet, classifier]
    params = [list(module.parameters()) for module in modules]
    params = reduce(lambda x, y: x + y, params)
    optimizer = torch.optim.Adam(
        params,
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)

    train_loss, validation_accuracies, context_accuracies = [], [],[]
    total_step = len(train_dataloader)

    for epoch in range(1, args.epochs + 1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss, running_acc, running_cont = 0.0, 0.0, 0.0
            for _, _target, _freq, _station, _context_label, _context_pivot, _context_neighbour in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                _context_pivot = _context_pivot.float().to(args.device)
                _context_neighbour = _context_neighbour.float().to(args.device)
                _freq = _freq.to(args.device)
                _context_label= _context_label.to(args.device)

                optimizer.zero_grad()

                z0 = resnet(_context_pivot)
                resnet_loss = resnet.loss_function(z0, _freq)['loss']

                z1 = resnet(_context_neighbour)
                c = classifier(z0,z1)
                classifier_loss = classifier.loss_function(c, _context_label)['loss']

                loss = resnet_loss + classifier_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _data = val_dataset.data.float().to(args.device)
                z = resnet(_data).cpu().detach()

                val_acc_freq = torch.sum(
                    z.argmax(
                        dim=-1) == val_dataset.frequency_band) / val_dataset.frequency_band.shape[0]
                running_acc += val_acc_freq
                
                _pivot = val_dataset.context_images_pivot.float().to(args.device)
                _neighbour = val_dataset.context_images_neighbour.float().to(args.device)

                z0 = resnet(_pivot)
                z1 = resnet(_neighbour)
                c = classifier(z0,z1).cpu().detach()

                val_acc_context = torch.sum(
                    c.argmax(
                        dim=-1) == val_dataset.context_labels) / val_dataset.context_labels.shape[0]
                running_cont += val_acc_context

                tepoch.set_postfix(loss=loss.item(), val_acc_freq=val_acc_freq.item(), val_acc_context=val_acc_context.item())

            train_loss.append(running_loss / total_step)
            validation_accuracies.append(running_acc/ total_step)
            context_accuracies.append(running_cont/ total_step)
            batch_loss = 0

            if epoch % 20 == 0:  # TODO: check for model improvement
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
                       context_accuracies=context_accuracies)
            for module in modules:
                module.train()
            resnet.train()
    return resnet
