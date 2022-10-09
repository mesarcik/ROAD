import torch
from torch.utils.data import DataLoader
from models import VAE
from sklearn.manifold import TSNE
from tqdm import tqdm
import os 

from utils.args import args
from utils.vis import imscatter, io, loss_curve


def train_vae(train_dataloader:DataLoader, vae:VAE, args:args) -> VAE:
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
    model_path = 'outputs/{}/{}/{}'.format(args.model, args.anomaly_class, args.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    vae.to(args.device)
    optimizer = torch.optim.Adam(vae.parameters(),lr=args.learning_rate)#, lr=0.0001, momentum=0.9)
    train_loss = []
    total_step = len(train_dataloader)

    for epoch in range(1, args.epochs+1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss = 0.0
            for _data, _target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                _data = _data.float().to(args.device)
                optimizer.zero_grad()

                [_decoded, _input, _mu, _log_var] = vae(_data)
                z = vae.reparameterize(_mu, _log_var)
                loss = vae.loss_function(_decoded, _input, _mu, _log_var)
                loss['loss'].backward()
                optimizer.step()

                running_loss += loss['loss'].item()
                tepoch.set_postfix(recon_loss=loss['Reconstruction_Loss'].item(), 
                                   total_loss=loss['loss'].item())
            train_loss.append(running_loss/total_step)
            batch_loss = 0

            if epoch %10 ==0: #TODO: check for model improvement
                torch.save(vae.state_dict(), '{}/vae.pt'.format(model_path))
                vae.eval()
                mu, log_var = vae.encode(_data)
                Z = vae.reparameterize(mu, log_var).cpu().detach().numpy()
                z = TSNE(n_components=2, 
                        learning_rate='auto',
                        init='random', 
                        perplexity=3).fit_transform(Z)
                
                _inputs= _data.cpu().detach().numpy()
                _reconstructions = _decoded.cpu().detach().numpy()
                io(10, _inputs, _reconstructions, model_path, epoch)
                imscatter(z, _inputs,model_path, epoch)
            loss_curve(model_path, total_loss=train_loss)
            vae.train()
    return vae 

def train_resnet(train_dataloader:DataLoader, resnet, args:args) :
    """
        Trainsa Resnet 
        
        Parameters
        ----------
        train_dataloader: dataloader
        resnet:resnet  
        args: runtime arguments

        Returns
        -------
        data: list of baselines with single channels removed

    """
    model_path = 'outputs/{}/{}/{}'.format(args.model, args.anomaly_class, args.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    resnet.to(args.device)
    optimizer = torch.optim.Adam(resnet.parameters(),lr=args.learning_rate)#, lr=0.0001, momentum=0.9)
    loss_function = torch.nn.BCEWithLogitsLoss()
    train_loss = []
    total_step = len(train_dataloader)

    for epoch in range(1, args.epochs+1):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            running_loss = 0.0
            for _data, _target, _freq in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                _data = _data.float().to(args.device)
                _freq= _freq.float().to(args.device)
                optimizer.zero_grad()

                z = resnet(_data)
                loss = loss_function(z, _freq)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(recon_loss=loss.item())

            train_loss.append(running_loss/total_step)
            batch_loss = 0

            if epoch %10 ==0: #TODO: check for model improvement
                torch.save(resnet.state_dict(), '{}/resnet.pt'.format(model_path))
                resnet.eval()
                z = resnet(_data).cpu().detach().numpy()
                
                _inputs= _data.cpu().detach().numpy()
                imscatter(z, _inputs,model_path, epoch)
            loss_curve(model_path, total_loss=train_loss)
            resnet.train()
    return resnet 
