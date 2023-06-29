import os
import sys
import math

import torch
import torch.utils.data as data_utils
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from utils.metrics import *

from models.base_encoder import BaseEncoder as Encoder
from models.additive_decoder_shared import AdditiveDecoder as Decoder

#Base Class
from algorithms.base_auto_encoder import AE

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

from utils.helper import ValidationHelper
from utils.metrics import *

import wandb


class AE_Additive_Shared(AE):
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset, seed=0, device= None):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, seed, device)        
        
        self.encoder= Encoder(self.args.latent_dim).to(self.device)
        self.decoder= Decoder(self.args.latent_dim, device=self.device).to(self.device)
        self.opt, self.scheduler= self.get_optimizer()                             
        
        if self.args.wandb_log:
            wandb.init(project="additive-decoder-identification", reinit=True)
            wandb.run.name=  'ae-additive-shared/' + self.args.save_dir + 'seed_' + str(seed) + '/'