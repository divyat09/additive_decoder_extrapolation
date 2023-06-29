import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock

class BaseDecoder(torch.nn.Module):
    
    def __init__(self, latent_dim, total_blocks= 2, image_width= 64, image_ch= 3, device= 'cpu'):
        super(BaseDecoder, self).__init__()        
        
        self.latent_dim = latent_dim
        self.image_width= image_width
        self.nc= image_ch        
        self.hidden_dim= 512
        self.device= device
        
        #TODO: Add more linear layers and check whether things work with 2 dimensional case
        self.linear =  [
                    nn.Linear(self.latent_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(0.1),            
            
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(0.1),            
            
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(0.1),            
            
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(0.1),            
            
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(0.1),            
            
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(0.1),            
            
                    nn.Linear(self.hidden_dim, 1024),
                    nn.BatchNorm1d(1024),
                    nn.LeakyReLU(),
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


    def forward(self, z):
        
        x = self.linear(z)
        x = x.view(z.size(0), 64, 4, 4)
        x = self.conv(x)
            
        return x
