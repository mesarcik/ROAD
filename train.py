import torch
from torch.utils.data import DataLoader
from models import VAE
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
from torch import nn


from utils.args import args
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

    vae.to(args.device)
    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)
    train_loss = []
    total_step = len(train_dataloader)

    for epoch in range(1, args.epochs + 1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss = 0.0
            for _data, _target, _freq, _station, _context, _ in tepoch:
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
    #classifier.to(args.device)

    encoder_optimizer = torch.optim.Adam(
        resnet.parameters(),
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)

    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)

    mse = nn.MSELoss()

    total_train_loss, encoder_train_loss, location_train_loss, dist_train_loss, jitter_train_loss  = [], [], [], [],[]
    validation_accuracies, context_accuracies = [], []
    total_step = len(train_dataloader)

    for epoch in range(1, args.epochs + 1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss, encoder_loss, location_loss, jitter_loss  = 0.0, 0.0, 0.0, 0.0
            running_acc, running_cont = 0.0, 0.0
            for _data, _target, _freq, _station, _context_label, _context_neighbour, _jittered in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                _data = _data.float().to(args.device)
                _context_neighbour = _context_neighbour.float().to(args.device)
                #_freq = _freq.to(args.device)
                _context_label= _context_label.to(args.device)
                _jittered = _jittered.to(args.device)

                encoder_optimizer.zero_grad()
                classifier_optimizer.zero_grad()

                c = resnet(_data, _context_neighbour)
                resnet_loss = resnet.loss_function(c, _context_label)['loss']

                _z = resnet.embed(_data)

                #c = classifier(_z)
                classifier_loss = torch.Tensor([0.0])#classifier.loss_function(c, _freq)['loss']

                _z0 = resnet.embed(_jittered)
                embedding_loss = torch.Tensor([0.0])#100*mse(_z, _z0)
                

                loss =   resnet_loss #+ 0.00001*torch.sum(torch.square(_z))#classifier_loss +  

                loss.backward()
                encoder_optimizer.step()
                classifier_optimizer.step()

                running_loss += loss.item()
                encoder_loss += resnet_loss.item()
                location_loss += classifier_loss.item()
                jitter_loss += embedding_loss.item()

                _data = val_dataset.data.float().to(args.device)
                _neighbour = val_dataset.context_images_neighbour.float().to(args.device)

                c = resnet(_data, _neighbour).cpu().detach()
                val_acc_context = torch.sum(
                    c.argmax(
                        dim=-1) == val_dataset.context_labels) / val_dataset.context_labels.shape[0]
                running_acc += val_acc_context

                #z0 = resnet.embed(_data)
                #c = classifier(z0).cpu().detach()

                val_acc_freq = torch.Tensor([0.0])#torch.sum(
                #    c.argmax(
                #        dim=-1) == val_dataset.frequency_band) / val_dataset.frequency_band.shape[0]
                running_cont+= 0#val_acc_freq
                

                tepoch.set_postfix(total_loss=loss.item(), 
                                   location_loss=resnet_loss.item(), 
                                   frequency_loss=classifier_loss.item(), 
                                   jitter_loss=embedding_loss.item(), 
                                   regularisation=0.00001*torch.sum(torch.square(_z)).cpu().detach().item(),
                                   val_acc_freq=val_acc_freq.item(),
                                   val_acc_context=val_acc_context.item())

            total_train_loss.append(running_loss / total_step)
            encoder_train_loss.append(encoder_loss / total_step)
            location_train_loss.append(location_loss / total_step)
            jitter_train_loss.append(jitter_loss/ total_step)

            validation_accuracies.append(running_acc/ total_step)
            context_accuracies.append(running_cont/ total_step)

            if epoch % 50 == 0:  # TODO: check for model improvement
                # print validation loss

                torch.save(
                    classifier.state_dict(),
                    '{}/classifier.pt'.format(model_path))
                
                torch.save(
                    resnet.state_dict(),
                    '{}/resnet.pt'.format(model_path))

                with open('{}/model.config'.format(model_path), 'w') as fp:
                    for arg in args.__dict__:
                        fp.write('{}: {}\n'.format(arg, args.__dict__[arg]))


                resnet.eval()
                #eval_resnet(resnet, train_dataloader, args, epoch=epoch,error='nln')

                Z = resnet.embed(_data).cpu().detach().numpy()
                z = TSNE(n_components=2,
                         learning_rate='auto',
                         init='random',
                         perplexity=30).fit_transform(Z)

                _inputs = _data.cpu().detach().numpy()
                imscatter(z, _inputs, model_path, epoch)
            loss_curve(model_path,
                       epoch,
                       total_loss=total_train_loss,
                       location_loss=encoder_train_loss,
                       frequency_loss=location_train_loss,
                       jitter_loss=jitter_train_loss,
                       validation_accuracy=validation_accuracies,
                       context_accuracies=context_accuracies)
            classifier.train()
            resnet.train()
    return resnet
