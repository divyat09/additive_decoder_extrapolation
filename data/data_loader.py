import os
import sys
import copy
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

class BaseDataLoader(data_utils.Dataset):
    def __init__(self, data_dir='', data_case='train', seed= 0):
        self.data_case= data_case
        self.seed= seed
        self.data_dir = 'data/datasets/' + data_dir + 'observation/'
        
        self.data, self.latents= self.load_data(self.data_case)
        
    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        z = self.latents[index]
            
        return x, z
    
    def load_data(self, data_case):
        
        x= np.load(self.data_dir + data_case +  '_' + 'x' + '.npy')
        z= np.load(self.data_dir + data_case +  '_' + 'z' + '.npy')
            
        x= torch.tensor(x).float()
        z= torch.tensor(z).float()
        
        return x, z