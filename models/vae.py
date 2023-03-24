"""
    Taken from https://github.com/AntixK/PyTorch-VAE
"""
import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self,in_channels: int, latent_dim: int, patch_size: int, hidden_dims: list = [32,64,128,256,512], **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.patch_size = patch_size

        modules = []
        self.hidden_dims = hidden_dims

        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)

        self.fc_size = self.hidden_dims[-1]*(int(self.patch_size/2**(len(self.hidden_dims)))**2) 

        self.fc_mu =  nn.Linear(self.fc_size, latent_dim)
        self.fc_var = nn.Linear(self.fc_size, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.fc_size)

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 4,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: torch.tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.tensor) Input torch.tensor to encoder [N x C x H x W]
        :return: (torch.tensor) list of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.tensor) -> torch.tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (torch.tensor) [B x D]
        :return: (torch.tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0],
                             (int(self.patch_size/2**(len(self.hidden_dims)))),
                             (int(self.patch_size/2**(len(self.hidden_dims)))))
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (torch.tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (torch.tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (torch.tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu#eps * std + mu

    def forward(self, input: torch.tensor, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = 0.00025
        recons_loss =F.mse_loss(recons, input)


        kld_loss = -0.5* torch.mean(torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.tensor, **kwargs) -> torch.tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.tensor) [B x C x H x W]
        :return: (torch.tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def save(self, args):
        fpath = self.fpath_from_name(args)
        torch.save(self.state_dict(), fpath)

    def load(self, args):
        fpath = self.fpath_from_name(args)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self,args)->str:
        return f'outputs/{args.model}/{args.model_name}/vae_{args.ood}_{args.seed}_{args.pretrain}.pkl'
