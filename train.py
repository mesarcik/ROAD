import torch
from torch.utils.data import DataLoader
from models import VAE
from sklearn.manifold import TSNE
import os 

from utils.args import args
from utils.vis import imscatter, io, loss_curve


def train_vae(train_dataloader:DataLoader, vae:VAE, args:args):
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
        running_loss = 0.0
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_ = data_.float().to(args.device)
            optimizer.zero_grad()

            [_decoded, _input, _mu, _log_var] = vae(data_)
            z = vae.reparameterize(_mu, _log_var)
            loss = vae.loss_function(_decoded, _input, _mu, _log_var)
            loss['loss'].backward()
            optimizer.step()

            running_loss += loss['Reconstruction_Loss'].item()

        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, args.epochs, loss['Reconstruction_Loss'].item()))
        train_loss.append(running_loss/total_step)
        batch_loss = 0

        if True: #TODO: check for model improvement
            torch.save(vae.state_dict(), '{}/vae.pt'.format(model_path))
            vae.eval()
            mu, log_var = vae.encode(data_)
            Z = vae.reparameterize(mu, log_var).cpu().detach().numpy()
            z = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(Z)
            
            _inputs= data_.cpu().detach().numpy()
            _reconstructions = _decoded.cpu().detach().numpy()
            io(10, _inputs, _reconstructions, model_path, epoch)
            imscatter(z, _inputs,model_path, epoch)
        loss_curve(model_path, total_loss=train_loss)
        vae.train()
        
