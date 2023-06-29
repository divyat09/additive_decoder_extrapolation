import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock

class AdditiveDecoder(torch.nn.Module):
    
    def __init__(self, latent_dim, total_blocks= 2, image_width= 64, image_ch= 3, device= 'cpu'):
        super(AdditiveDecoder, self).__init__()        
        
        self.latent_dim = latent_dim
        self.total_blocks = total_blocks
        self.eff_latent_dim = int(latent_dim/total_blocks)
        self.image_width= image_width
        self.nc= image_ch        
        self.hidden_dim= 512
        self.device= device
        
        self.layers= {}
        for idx in range(total_blocks):
            
            key= 'linear_' + str(idx)
            self.layers[key]= [
                    nn.Linear(self.eff_latent_dim, self.hidden_dim),
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
                    nn.LeakyReLU(0.1),
                    ]
            self.layers[key]= nn.Sequential(*self.layers[key]).to(self.device)
        
            key= 'conv_' + str(idx)

            self.layers[key]= [
                    nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),           
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, self.nc, 4, stride=2, padding=1),
                    ]
            self.layers[key]= nn.Sequential(*self.layers[key]).to(self.device)        
        
        for layer_name, layer in self.layers.items():
            self.add_module(layer_name, layer)
            for param_name, param in layer.named_parameters():
                param_name = param_name.replace('.', '_')
                self.register_parameter(str(layer_name)+'_'+str(param_name), nn.Parameter(param.data))

    def forward(self, z):
        
        batch_size= z.shape[0]
        final_out= torch.zeros((batch_size, self.nc, self.image_width, self.image_width)).to(self.device)
        
        list_z = torch.split(z, self.eff_latent_dim, dim=1)
        for idx, z in enumerate(list_z):
            x = self.layers['linear_'+str(idx)](z)
            x = x.view(z.size(0), 64, 4, 4)
            x = self.layers['conv_'+str(idx)](x)
            final_out+= x
            
        return final_out

    def render_block(self, z, idx):
        
        batch_size= z.shape[0]
        list_z = torch.split(z, self.eff_latent_dim, dim=1)
        x = self.layers['linear_'+str(idx)](list_z[idx])
        x = x.view(z.size(0), 64, 4, 4)
        x = self.layers['conv_'+str(idx)](x)
            
        return x
