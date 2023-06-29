import os
import sys
import copy
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler

# Base Class
from data.data_loader import BaseDataLoader

class BallsDataLoader():
    def __init__(self, data_case='train', latent_case= '', num_balls= 2, train_size= 50000):
        
        self.data_case= data_case
        self.latent_case= latent_case
        self.num_balls= num_balls
        self.train_size= train_size
        self.data_dir = 'data/datasets/' + str(self.latent_case) + '/' + 'num_balls_' + str(self.num_balls) + '/'
        
        self.data, self.latents= self.load_data()
        print('Dataset Size: ', self.__len__())
            
    def __len__(self):
        return self.latents.shape[0]
    
    def __getitem__(self, index):
        x = self.data[index]
        z = self.latents[index]
        
        data_transform=  transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

#         data_transform=  transforms.Compose([
#             transforms.Normalize([0, 0, 0], [255, 255, 255])
#         ])
        
        x= data_transform(x)        
        
        return x, z
            
    def load_data(self):
        
        x= torch.tensor( np.load(self.data_dir + self.data_case +  '_' + 'x' + '.npy') ).float()
        z= torch.tensor( np.load(self.data_dir + self.data_case +  '_' + 'z' + '.npy') ).float()
            
        # Change the dimension from (B, H, W, C) to (B, C, H ,W)
        x= x.permute(0, 3, 1, 2)
        
        #Remove the x-coordinate from the latents for the following cases as we only allow for motion along the y-axis
        if self.latent_case in ['balls_supp_l_shape', 'balls_supp_extrapolate', 'balls_supp_disconnected']:
            z= np.delete(z, [0, 2], axis=1)
        
        #Undersampling the dataset for training and validation for sample complexity plots
        #To use the full dataset keep train size to its maximal value
        if self.data_case == 'train':
            x= x[:self.train_size]
            z= z[:self.train_size]
        elif self.data_case == 'val':
            x= x[: int(self.train_size/4)]
            z= z[: int(self.train_size/4)]          
        
        return x, z