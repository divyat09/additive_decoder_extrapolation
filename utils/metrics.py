import sys
import copy
import torch
import torchvision
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from scipy.optimize import linear_sum_assignment
from sklearn.feature_selection import mutual_info_regression
import scipy
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from itertools import combinations

def invert_imagenet_transform(data: torch.Tensor) -> torch.Tensor:
    """
    Inverts the imagenet normalization transform.

    Inputs:
        data: Image to be transformed; Expected shape: (3, 64, 64)
    Returns:
        Transformed image; Expected shape: (3, 64, 64)        
    """

    mean_norm= [0.485, 0.456, 0.406]
    sem_norm= [0.229, 0.224, 0.225]
    for idx in range(3):
        data[idx, :, : ] = data[idx, :, : ] * sem_norm[idx] + mean_norm[idx]

    return data

def plot_transform(img: torch.Tensor, input_normalization: str='none') -> torch.Tensor:
    """
    Converts each image to [0, 1] range for plotting.

    Inputs:
        data: img to be transformed; Expected shape: (3, 64, 64)
    Returns:
        Transformed image; Expected shape: (3, 64, 64)        
    """

    if input_normalization == 'none':
         img= img/255

    elif input_normalization == 'imagenet':
        img= invert_imagenet_transform(img)
        img= img/255
 
    elif input_normalization == 'tensor_uniform':
        img= (img+1)/2

    elif input_normalization == 'tensor_imagenet': 
        img= invert_imagenet_transform(img)

    img= torch.clamp(img, 0, 1)
    return img


def regression_approx(x: np.array, y: np.array, model: str, fit_intercept: bool=False):
    """
    Learns a regression model to approximate y from x.

    Inputs:
        x: Input features; expected shape: (batch size, feature dim)
        y: Target values; expected shape: (batch size, target dim)
        model: Type of regression model to be used.
        fit_intercept: Whether to have intercept term in regression.

    Returns:
        Trained regression model.
    """

    if model == 'lr':
        reg= LinearRegression(fit_intercept= fit_intercept).fit(x, y)
    elif model == 'lasso':
#         reg= Lasso(alpha=0.001, fit_intercept= True).fit(x, y)
        reg= LassoCV(fit_intercept=True, cv=3).fit(x, y)
    elif model == 'ridge':
#         reg= Ridge(alpha=0.001, fit_intercept= True).fit(x, y)
        alphas_list = np.linspace(1e-2, 1e0, num=10).tolist()
        alphas_list += np.linspace(1e0, 1e1, num=10).tolist()
        reg= RidgeCV(fit_intercept= True, cv=3, alphas=alphas_list).fit(x, y)
    elif model == 'mlp':
        reg= MLPRegressor(random_state=1, max_iter= 1000).fit(x, y)
    elif model == 'd-tree':
        reg= DecisionTreeRegressor(max_depth=10).fit(x, y)
        
    return reg


def get_cross_correlation(z: np.ndarray, z_hat: np.ndarray, corr_case: str='pearson') -> float:
    """
    Computes the MCC using Hungarian matching.
    
    Inputs:
        z: True Latent, expected shape: (batch size, latent dim)
        z_hat: Predicted Latent, expected shape: (batch size, latent dim)
        corr_case: Type of correlation to be used; Pearson/Spearman
    """
    
    num_samples= z_hat.shape[0]
    dim= z_hat.shape[1]
        
    cross_corr= np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if corr_case == 'pearson':
                cross_corr[i,j]= (np.cov( z_hat[:,i], z[:,j] )[0,1]) / ( np.std(z_hat[:,i])*np.std(z[:,j]) )
            elif corr_case == 'spearman':
                cross_corr[i, j]= spearmanr(z_hat[:,i], z[:,j])[0]
        
    cost= -1*np.abs(cross_corr)
    row_ind, col_ind= linear_sum_assignment(cost)
    score= 100*( -1*cost[row_ind, col_ind].sum() )/(dim)
        
    return score

def mcc_block_reg_model(z: np.ndarray, z_hat: np.ndarray, total_blocks: int=2, partition=None, reg_case: str='d-tree'):
    """
    Computes the MCC for blockwise identification by regressing over latent blocks to compute correlation scores.

    Inputs:
        z: True Latent, expected shape: (batch size, latent dim)
        z_hat: Predicted Latent, expected shape: (batch size, latent dim)
        total_blocks: Total blocks in the latent partition
        parition: Partition of the predicted latent space into blocks
        reg_case: Type of model to be used for regression

    Returns:
        Dictionary of regression models for predicting the true latent blocks from predicted latent blocks
    """
    
    feat_block_size= int(z_hat.shape[1]/total_blocks)
    target_block_size= int(z.shape[1]/total_blocks)
        
    #Arrange the predicted latents as per the input partition
    z_hat= z_hat[:, partition]    
     
    reg_model={}
    for i in range(total_blocks):
        for j in range(total_blocks):
            curr_feat= z_hat[:, i*feat_block_size : (i+1)*feat_block_size ]
            curr_target= z[:, j*target_block_size : (j+1)*target_block_size ]         
            reg_model[str(i) + '_' + str(j)] = regression_approx(curr_feat, curr_target, reg_case)
   
    return reg_model

def get_block_mcc(z: np.ndarray, z_hat: np.ndarray, total_blocks: int=2, partition= None, reg_model= None) -> float:
    """
    Computes the MCC for blockwise identification by regressing over latent blocks to compute correlation scores.

    Inputs:
        z: True Latent, expected shape: (batch size, latent dim)
        z_hat: Predicted Latent, expected shape: (batch size, latent dim)
        total_blocks: Total blocks in the latent partition
        parition: Partition of the predicted latent space into blocks
        reg_model: Regression model for predicting the true latent blocks from predicted latent blocks
    """
    
    feat_block_size= int(z_hat.shape[1]/total_blocks)
    target_block_size= int(z.shape[1]/total_blocks)
    
    #Arrange the predicted latents as per the input partition
    z_hat= z_hat[:, partition]    
        
    cross_corr= np.zeros((total_blocks, total_blocks))
    for i in range(total_blocks):
        for j in range(total_blocks):
            curr_feat= z_hat[:, i*feat_block_size : (i+1)*feat_block_size ]
            curr_target= z[:, j*target_block_size : (j+1)*target_block_size ]
            
            score= r2_score( curr_target, reg_model[str(i) + '_' + str(j)].predict(curr_feat) )
#             print('Pred Block: ', i, 'True Block', j, score, np.var(curr_feat), curr_feat.shape )
            if score < 0:
                score= 0.0
            cross_corr[i, j]= score

    cost= -1*np.abs(cross_corr)
    row_ind, col_ind= linear_sum_assignment(cost)
    score= 100*( -1*cost[row_ind, col_ind].sum() )/(total_blocks)    
    
    return score

