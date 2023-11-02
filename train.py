#Common imports
import sys
import os
import argparse
import random
import copy

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import FastICA

#Algorithms
from algorithms.base_auto_encoder import AE
from algorithms.additive_auto_encoder import AE_Additive

#DataLoaders
from data.balls_dataset_loader import BallsDataLoader
from data.balls_dataset_loader import sample_base_data_loaders

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--method_type', type=str, default='ae_additive',
                   help= '')
parser.add_argument('--data_dim', type=int, default= 200,
                    help='')
parser.add_argument('--latent_dim', type=int, default= 2,
                    help='')
parser.add_argument('--total_blocks', type=int, default= 2,
                    help='')
parser.add_argument('--latent_case', type=str, default= 'balls_supp_l_shape',
                    help='')
parser.add_argument('--train_size', type=int, default= 10000,
                    help='')
parser.add_argument('--batch_size', type=int, default= 64,
                    help='')
parser.add_argument('--lr', type=float, default= 5e-4,
                    help='')
parser.add_argument('--weight_decay', type=float, default= 5e-4,
                    help='')
parser.add_argument('--num_epochs', type=int, default= 1000,
                    help='')
parser.add_argument('--seed', type=int, default=0,
                    help='')
parser.add_argument('--input_normalization', type=str, default='none',
                   help= '')
parser.add_argument('--wandb_log', type=int, default=0,
                   help='')
parser.add_argument('--cuda_device', type=int, default=0, 
                    help='Select the cuda device by id among the avaliable devices' )

args = parser.parse_args()
method_type= args.method_type
latent_case= args.latent_case
total_blocks= args.total_blocks
data_dim= args.data_dim
latent_dim= args.latent_dim
train_size= args.train_size
batch_size= args.batch_size
lr= args.lr
weight_decay= args.weight_decay
num_epochs= args.num_epochs
seed= args.seed
input_normalization= args.input_normalization
wandb_log= args.wandb_log
cuda_device= args.cuda_device

#GPU
if cuda_device == -1:
    device= torch.device("cpu")
else:
    device= torch.device("cuda:" + str(cuda_device))
    
if device:
    kwargs = {'num_workers': 0, 'pin_memory': False} 
else:
    kwargs= {}

#Seed values
random.seed(seed*10)
np.random.seed(seed*10) 
torch.manual_seed(seed*10)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed*10)
        
    
# Load Dataset
train_dataset, val_dataset, test_dataset= sample_base_data_loaders(
                                                                  latent_case= latent_case, 
                                                                  num_balls= total_blocks,
                                                                  train_size= train_size,
                                                                  batch_size= batch_size, 
                                                                  input_normalization= input_normalization,
                                                                  kwargs=kwargs
                                                                 )

#Load Algorithm
if method_type == 'ae_base':
    method= AE(args, train_dataset, val_dataset, test_dataset, seed=seed, device= device)    
elif method_type == 'ae_additive':
    method= AE_Additive(args, train_dataset, val_dataset, test_dataset, seed=seed, device= device)    
else:
    print('Error: Incorrect method type')
    sys.exit(-1)

# Training
method.train()