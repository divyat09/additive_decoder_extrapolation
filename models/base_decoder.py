import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

class BaseDecoder(torch.nn.Module):
    """
    Decoder architecture for the Auto Encoder

    Attributes:
        latent_dim: Dimension of the latent space
        image_width: Width of the output image
        nc: Number of channels in the output image
        hidden_dim: Width of the hidden layers
        linear: Fully connected layers network
        conv: Convolutional layers network
    """

    def __init__(self, latent_dim: int= 2, image_width: int= 64, image_ch: int= 3):
        super(BaseDecoder, self).__init__()        
        
        self.latent_dim = latent_dim
        self.image_width= image_width
        self.nc= image_ch        
        self.hidden_dim= 512
        self.relu_slope= 0.01
        
        #TODO: Add more linear layers and check whether things work with 2 dimensional case
        self.linear =  [
                    nn.Linear(self.latent_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(self.relu_slope),
                        
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(self.relu_slope),
            
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(self.relu_slope),
            
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(self.relu_slope),
            
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(self.relu_slope),

                    nn.Linear(self.hidden_dim, 1024),
                    nn.BatchNorm1d(1024),
                    nn.LeakyReLU(self.relu_slope),
        ]        
        self.linear= nn.Sequential(*self.linear)

        self.conv= [
                    nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),           
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, self.nc, 4, stride=2, padding=1),
        ]
        self.conv= nn.Sequential(*self.conv)


    def forward(self, z: torch.tensor) -> torch.tensor:
        """
            Inputs:
                z: Latent representation tensor; expected shape: (batch size, latent dimension)
            Returns:
                Reconstructed image; expected shape: (batch size, 3, 64, 64)
        """
        x = self.linear(z)
        x = x.view(z.size(0), 64, 4, 4)
        x = self.conv(x)
            
        return x
