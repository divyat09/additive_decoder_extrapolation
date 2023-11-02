import torch
from torch import nn
from torchvision import models as vision_models
from torchvision.models import resnet18, resnet50
from torchvision import transforms

class BaseEncoder(torch.nn.Module):
    """
    Encoder architecture for the Auto Encoder

    Attributes:
        latent_dim: Dimension of the latent space
        base_architecture: Convolutional base architectures
        width: Width of the hidden layers
        feat_net: Convolutional layers network
        fc_net: Fully connected layers network
    """

    def __init__(self, latent_dim: int= 2):
        super(BaseEncoder, self).__init__()        
        
        self.latent_dim = latent_dim
        self.base_architecture= 'resnet18'
        self.width = 512
        self.relu_slope = 0.01
        
        #Conv Layers
        self.base_model = resnet18(pretrained=False)        
        self.feat_layers= list(self.base_model.children())[:-1]
        self.feat_net= nn.Sequential(*self.feat_layers)
        
        #Fully Connecter Layers
        self.fc_layers= [                    
                    nn.Linear(512, self.width),
                    nn.BatchNorm1d(self.width),
                    nn.LeakyReLU(self.relu_slope),     
           
                    nn.Linear(self.width, self.width),
                    nn.BatchNorm1d(self.width),
                    nn.LeakyReLU(self.relu_slope),            
            
                    nn.Linear(self.width, self.width),
                    nn.BatchNorm1d(self.width),
                    nn.LeakyReLU(self.relu_slope),            
            
                    nn.Linear(self.width, self.width),
                    nn.BatchNorm1d(self.width),
                    nn.LeakyReLU(self.relu_slope),            
            
                    nn.Linear(self.width, self.width),
                    nn.BatchNorm1d(self.width),
                    nn.LeakyReLU(self.relu_slope),

                    nn.Linear(self.width, self.latent_dim),
                    nn.BatchNorm1d(self.latent_dim),
                ]
        
        self.fc_net = nn.Sequential(*self.fc_layers)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
            Inputs:
                x: Input image tensor; expected shape: (batch size, 3, 64, 64)
            Returns:
                Latent representations of the image
        """

        x= self.feat_net(x)
        x= x.view(x.shape[0], x.shape[1])
        x= self.fc_net(x)
        return x
