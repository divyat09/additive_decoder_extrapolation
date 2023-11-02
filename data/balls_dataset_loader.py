import os
import sys
import copy
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler


class BallsDataLoader():
    def __init__(self, data_case: str='train', latent_case: str= '', num_balls: int= 2, train_size: int= 50000, input_normalization: str= ''):
        """
        Inputs:
            data_case: Split of the dataset to be used. Options: 'train', 'val', 'test'
            latent_case: Type of the dataset to be used. Options: 'balls_supp_l_shape', 'balls_supp_extrapolate', 'balls_supp_disconnected', etc. 
            num_balls: Number of balls in the dataset.
            train_size: Size of the training dataset.
            input_normalization: Type of input normalization to be used. Options: 'none', 'imagenet', 'tensor_imagenet', 'tensor_uniform'        
        """
        
        self.data_case= data_case
        self.latent_case= latent_case
        self.num_balls= num_balls
        self.train_size= train_size
        self.input_normalization= input_normalization
        self.data_dir = 'data/datasets/' + str(self.latent_case) + '/' + 'num_balls_' + str(self.num_balls) + '/'
        
        self.data, self.latents= self.load_data()
        print('Dataset Size: ', self.__len__())
            
    def __len__(self):
        return self.latents.shape[0]
    
    def __getitem__(self, index):
        x = self.data[index]
        z = self.latents[index]
        
        if self.input_normalization != 'none':
            if self.input_normalization == 'imagenet':
                #Applies imagenet specific normalization without converting to tensor [0, 1] first
                x= torch.tensor(x, dtype=torch.float)
                x= x.permute(2, 0, 1)
                self.data_transform=  transforms.Compose([
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            elif self.input_normalization == 'tensor_imagenet':
                #Converts image to tensor [0, 1] and scales with imagenet specific normalization
                self.data_transform=  transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            elif self.input_normalization == 'tensor_uniform':
                #Scales final image to [-1, 1] range
                self.data_transform=  transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

            x= self.data_transform(x)
        else:
            #Does not change the range of pixels; keeps them in [0, 255] range.
            x= torch.tensor(x, dtype=torch.float)
            x= x.permute(2, 0, 1)

        z= torch.tensor(z, dtype=torch.float)

        return x, z
            
    def load_data(self):
        """"
        Loads the dataset from the specified path, and determines the size of the dataset based on the data split.
        """
        
        x= np.load(self.data_dir + self.data_case +  '_' + 'x' + '.npy')
        z=  np.load(self.data_dir + self.data_case +  '_' + 'z' + '.npy')
        
        #Remove the x-coordinate from the latents for the following cases as we only allow for motion along the y-axis
        if self.latent_case in ['balls_supp_l_shape', 'balls_supp_extrapolate', 'balls_supp_disconnected', 'diff_balls_supp_l_shape', 'diff_balls_supp_extrapolate']:
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


def sample_base_data_loaders(batch_size: int= 64, latent_case: str='', num_balls: int= 2, train_size: int= 50000, input_normalization: str= '',  kwargs={}):
    """
    Returns train/val/test dataloader given arguments necessary to specify the path for the current dataset.
    """
        
    data_obj= BallsDataLoader(data_case='train', latent_case= latent_case, num_balls= num_balls, train_size= train_size, input_normalization= input_normalization)
    train_loader= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

    data_obj= BallsDataLoader(data_case='val', latent_case= latent_case, num_balls= num_balls, train_size= train_size, input_normalization= input_normalization)
    val_loader= data_utils.DataLoader(data_obj, batch_size= 1024, shuffle=True, **kwargs )

    data_obj= BallsDataLoader(data_case='test', latent_case= latent_case, num_balls= num_balls, train_size= train_size, input_normalization= input_normalization)
    test_loader= data_utils.DataLoader(data_obj, batch_size= 1024, shuffle=True, **kwargs )    
    
    return train_loader, val_loader, test_loader

def sample_latent_traversal_data(batch_size: int= 81, latent_case: str='', num_balls: int= 2, input_normalization: str= '',  kwargs={}):
    """
    Returns dataloader for latent traversal plots; which correspond to balls moving sparsely along a particular axis.
    """
    traversal_size= 81
    data_obj= BallsDataLoader(data_case='train', latent_case= latent_case, num_balls= num_balls, train_size= traversal_size, input_normalization= input_normalization)
    #Keep shuffle to be False as we want the data to be in order (Ball 1 Grid * Ball 2 Grid) for latent traversal plots.
    train_loader= data_utils.DataLoader(data_obj, batch_size=traversal_size, shuffle=False, **kwargs )
        
    return train_loader

