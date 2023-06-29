import os
import sys
import numpy as np

import torch
import torch.utils.data as data_utils

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

from data.data_loader import BaseDataLoader
from data.balls_dataset_loader import BallsDataLoader


class ValidationHelper:
    def __init__(self, patience=1000, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_epoch = -1
        
    def save_model(self, validation_loss, epoch):
        if validation_loss < (self.min_validation_loss + self.min_delta):
            self.min_validation_loss = validation_loss
            self.best_epoch = epoch
            self.counter= 0
            return True
        return False

    def early_stop(self, validation_loss):        
        if validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    
def sample_base_data_loaders(batch_size= 64, latent_case='', num_balls= 2, train_size= 50000, seed=-1, kwargs={}):
        
    if 'balls' in latent_case:
        
        data_obj= BallsDataLoader(data_case='train', latent_case= latent_case, num_balls= num_balls, train_size= train_size)
        train_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

        data_obj= BallsDataLoader(data_case='val', latent_case= latent_case, num_balls= num_balls, train_size= train_size)
        val_dataset= data_utils.DataLoader(data_obj, batch_size= 1024, shuffle=True, **kwargs )

        data_obj= BallsDataLoader(data_case='test', latent_case= latent_case, num_balls= num_balls, train_size= train_size)
        test_dataset= data_utils.DataLoader(data_obj, batch_size= 1024, shuffle=True, **kwargs )    
    
    return train_dataset, val_dataset, test_dataset
