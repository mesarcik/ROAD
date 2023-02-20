import torch
from torch.utils.data import DataLoader
from models import VAE, Decoder
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

    decoder = Decoder(out_channels=4, patch_size=args.patch_size, latent_dim=args.latent_dim)
    #decoder.load_state_dict(torch.load('outputs/position_classifier/{}/decoder.pt'.format(args.model_name)))

    resnet.to(args.device, dtype=torch.bfloat16)
    #classifier.to(args.device)
    decoder.to(args.device, dtype=torch.bfloat16)

    encoder_optimizer = torch.optim.Adam(
        resnet.parameters(),
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)

    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)

    decoder_optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=args.learning_rate)  # , lr=0.0001, momentum=0.9)


    mse = nn.MSELoss()

    total_train_loss, encoder_train_loss, location_train_loss, dist_train_loss, jitter_train_loss  = [], [], [], [],[]
    validation_accuracies, context_accuracies = [], []
    total_step = len(train_dataloader)
    prev_acc = 0

    for epoch in range(1, args.epochs + 1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss, encoder_loss, freq_loss, d_loss  = 0.0, 0.0, 0.0, 0.0
            running_acc, running_cont = 0.0, 0.0
            for _data, _target, _freq,  _context_label, _context_neighbour, _context_frequency in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                _data = combine(_data,0,2).float().to(args.device, dtype=torch.bfloat16)
                _context_neighbour = combine(_context_neighbour,0,2).float().to(args.device, dtype=torch.bfloat16)
                _context_frequency= combine(_context_frequency,0,2).to(args.device, dtype=torch.bfloat16)
                _context_label= combine(_context_label,0,2).to(args.device, dtype=torch.long)

                encoder_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                z_data  = resnet.embed(_data)
                x_hat = decoder(z_data)
                z_neighbour = resnet.embed(_context_neighbour)
                c_pos = resnet(z_data, z_neighbour)
                #c_freq = classifier(z_data, z_neighbour)

                location_loss = resnet.loss_function(c_pos, _context_label)['loss']
                #frequency_loss = classifier.loss_function(c_freq, _context_frequency)['loss']
                decoder_loss = mse(_context_neighbour, x_hat)

                loss = location_loss + decoder_loss #+ 1.00001*torch.sum(torch.square(_z))#classifier_loss +  

                loss.backward()
                encoder_optimizer.step()
                #classifier_optimizer.step()
                decoder_optimizer.step()

                running_loss += loss.item()
                encoder_loss += location_loss.item()
                #freq_loss += frequency_loss.item()
                d_loss += decoder_loss.item()

                ##########
                _data = val_dataset.patch(val_dataset.data)
                _labels, _neighbour, _freq = val_dataset.context_prediction(_data)
                _data = _data.float().to(args.device, dtype=torch.bfloat16)

                _neighbour = _neighbour.float().to(args.device, dtype=torch.bfloat16)
                z_data  = resnet.embed(_data)
                z_neighbour = resnet.embed(_neighbour)

                c_pos  = resnet(z_data, z_neighbour).cpu().detach()
                val_acc_context = torch.sum(
                    c_pos.argmax(
                        dim=-1) == _labels) / _labels.shape[0]
                running_acc += val_acc_context

                #c_freq= classifier(z_data, z_neighbour).cpu().detach()
                #val_acc_freq = torch.sum(
                #    c_freq.argmax(
                #        dim=-1) == val_dataset.context_frequency_neighbour) / val_dataset.context_frequency_neighbour.shape[0]
                #running_cont+= val_acc_freq

                tepoch.set_postfix(total_loss=loss.item(),
                                   location_loss=location_loss.item(),
                                   #frequency_loss=frequency_loss.item(),
                                   decoder_loss=decoder_loss.item(),
                                   #val_acc_freq=val_acc_freq.item(),
                                   val_acc_context=val_acc_context.item())

            total_train_loss.append(running_loss / total_step)
            encoder_train_loss.append(encoder_loss / total_step)
            #location_train_loss.append(freq_loss / total_step)

            validation_accuracies.append(running_acc/ total_step)
            #context_accuracies.append(running_cont/ total_step)

            if epoch % 50 == 0:  # TODO: check for model improvement
                # print validation loss
                if val_acc_context>prev_acc:
                    prev_acc = val_acc_context
                    torch.save(
                        decoder.state_dict(),
                        '{}/decoder_{}.pt'.format(model_path,epoch))
                    
                    torch.save(
                        resnet.state_dict(),
                        '{}/resnet_{}.pt'.format(model_path,epoch))

                    with open('{}/model.config'.format(model_path), 'w') as fp:
                        for arg in args.__dict__:
                            fp.write('{}: {}\n'.format(arg, args.__dict__[arg]))


                #resnet.eval()

                #Z = z_data.cpu().detach().float().numpy()  #resnet.embed(_data).cpu().detach().numpy()
                #z = TSNE(n_components=2,
                #         learning_rate='auto',
                #         init='random',
                #         perplexity=30).fit_transform(Z)

                #_inputs = _data.cpu().detach().numpy().float()

                #imscatter(z, _inputs, model_path, epoch)

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
    return resnet
