#Common imports
import sys
import os
import argparse
import random
import copy
import seaborn

import pandas as pd
import pickle

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data_utils
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy import stats
from sklearn.decomposition import FastICA

from algorithms.base_auto_encoder import AE
from algorithms.additive_auto_encoder import AE_Additive

from utils.metrics import *
from utils.helper import *

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--method_type', type=str, default='ae_additive',
                   help= 'ae, ae_poly')
parser.add_argument('--latent_case', type=str, default='balls_supp_l_shape',
                    help='laplace; uniform')
parser.add_argument('--eval_latent_case', type=str, default='balls_supp_l_shape',
                    help='laplace; uniform')
parser.add_argument('--plot_case', type=str, default='extrapolate',
                   help= 'extrapolation, block_rendering, latent_traversal')
parser.add_argument('--data_dim', type=int, default= 200,
                    help='')
parser.add_argument('--latent_dim', type=int, default= 2,
                    help='')
parser.add_argument('--total_blocks', type=int, default= 2,
                    help='')
parser.add_argument('--train_size', type=int, default= 50000,
                    help='')
parser.add_argument('--batch_size', type=int, default= 64,
                    help='')
parser.add_argument('--lr', type=float, default= 5e-4,
                    help='')
parser.add_argument('--weight_decay', type=float, default= 5e-4,
                    help='')
parser.add_argument('--num_seeds', type=int, default=10,
                    help='')
parser.add_argument('--target_seed', type=int, default=-1,
                    help='')
parser.add_argument('--wandb_log', type=int, default=0,
                   help='')
parser.add_argument('--cuda_device', type=int, default=0, 
                    help='Select the cuda device by id among the avaliable devices' )

args = parser.parse_args()
method_type= args.method_type
latent_case= args.latent_case
plot_case= args.plot_case
total_blocks= args.total_blocks
eval_latent_case= args.eval_latent_case
data_dim= args.data_dim
latent_dim= args.latent_dim
train_size= args.train_size
batch_size= args.batch_size
lr= args.lr
weight_decay= args.weight_decay
num_seeds= args.num_seeds
target_seed= args.target_seed
plot_case= args.plot_case
wandb_log= args.wandb_log
cuda_device= args.cuda_device
    
#GPU
if cuda_device == -1:
    device= "cpu"
else:
    device= torch.device("cuda:" + str(cuda_device))
    
if device:
    kwargs = {'num_workers': 0, 'pin_memory': False} 
else:
    kwargs= {}

res={}

print('Details')
print(method_type, eval_latent_case, train_size)

for seed in range(num_seeds):
    
    if target_seed != -1 and seed!=target_seed:
        continue
        
    #Seed values
    random.seed(seed*10)
    np.random.seed(seed*10)
    torch.manual_seed(seed*10)
    
    # Load Dataset
    eval_batch_size= 10000
    train_dataset, val_dataset, test_dataset= sample_base_data_loaders(
                                                                      latent_case= eval_latent_case,
                                                                      num_balls= total_blocks,
                                                                      train_size= train_size,
                                                                      batch_size= eval_batch_size, 
                                                                      seed= seed, 
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
        
    # Evaluate the base model
    method.load_model()        
    
    #Obtain Predictions and Reconstruction Loss
    save_dir= 'results/' +  args.latent_case + '_eval_latent_' +  args.eval_latent_case + \
             '/' + args.method_type + '/' +  str(args.train_size) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if method_type == 'ae_additive':
        logs= get_predictions(method.encoder, method.decoder, train_dataset, val_dataset, test_dataset, device=method.device, save_dir= save_dir, seed= seed, plot= True)
    else:
        logs= get_predictions(method.encoder, method.decoder, train_dataset, val_dataset, test_dataset, device=method.device, save_dir= save_dir, seed= seed, plot= False)
    

    if plot_case == 'extrapolate':

        fontsize= 30
        fontsize_lgd= fontsize/1.5        

        pred_latent= logs['pred_z']['te']
        true_latent= logs['true_z']['te']

        # Latent Traversal 2
        width= 64
        height= 64
        padding= 10
        grid_size= 7
        final_image= Image.new('RGB', ( 3*padding +  (width+padding)*grid_size, 3*padding + (height+padding)*grid_size))

        init_val= np.min(pred_latent[:, 0])
        final_val= np.max(pred_latent[:, 0])
        grid_range_x= np.linspace(init_val, final_val, num=grid_size)

        init_val= np.min(pred_latent[:, 1])
        final_val= np.max(pred_latent[:, 1])
        grid_range_y= np.linspace(init_val, final_val, num=grid_size)

        for idx_x in range(grid_size):

            supp_val= grid_range_x[idx_x]

            latent_traverse= []
            for val in grid_range_y:
                latent_traverse.append([supp_val, val])
            latent_traverse= torch.tensor( np.array(latent_traverse) ).float()

            x_pred= method.decoder( latent_traverse.to(device) ).to('cpu').detach()
            transform = transforms.Compose(
                                [
                                    transforms.ToPILImage(),
                                ]
                            )   

            for idx_y in range(x_pred.shape[0]):
                data= x_pred[idx_y]
                data= (data - data.min()) / (data.max() - data.min())
                data= transform(data)
                final_image.paste(data, ( 2*padding+ (width+padding) * idx_x, 2*padding + (height+padding)*(grid_size - idx_y -1)))

        final_image.save(save_dir +  'traversed_image_seed_' + str(seed) + '.jpg')

        color_latent= []
        for idx in range(pred_latent.shape[0]):
            color_latent.append( true_latent[idx, 0] )
#             color_latent.append( true_latent[idx, 1] )

        plt.scatter(pred_latent[:, 0], pred_latent[:, 1], c=color_latent)
        plt.xlabel('Predicted Latent 1', fontsize= fontsize)
        plt.ylabel('Predicted Latent 2', fontsize= fontsize)
#         plt.title('Predicted Latent Support', fontsize= fontsize)

        ood_x= []
        ood_y= []

        for idx_x in range(grid_size):
            for idx_y in range(grid_size):
                ood_x.append(grid_range_x[idx_x])
                ood_y.append(grid_range_y[idx_y])

        ood_x= np.array(ood_x)
        ood_y= np.array(ood_y)

        plt.scatter(ood_x, ood_y, c='red')
        
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(save_dir + 'latent_support_seed_' + str(seed) + '.jpg') 
        plt.clf()

        print('Done')

    elif plot_case == 'latent_traversal':

        x_ticks= 0.1 + 0.1 * np.array( range(9) )
        fontsize=48
        fontsize_lgd= fontsize/1.25
        marker_list = ['o', '^', 's', '*']
        
        fig, ax = plt.subplots(2, 2, figsize=(23, 16))
        
        movement_axis= ['x', 'y']
        for idx, axis in enumerate(movement_axis):

            x= torch.load('data/datasets/balls_latent_traversal/' + 'supp_grid_' + axis + '_axis_x.pt')
            with torch.no_grad():
                pred_latent= method.encoder(x.to(device)).to('cpu').detach().view(9, 9, latent_dim).numpy()  
            
            curr_latent= pred_latent[:, 4, :]
            
            ax[idx, 0].grid(True) 
            ax[idx, 0].set_ylim(-3.5, 4.5)
            ax[idx, 0].tick_params(labelsize=fontsize)
            ax[idx, 0].set_ylabel('Predicted Latents', fontsize=fontsize)
            ax[idx, 0].set_xlabel('Ball 1 moving along ' + axis  + ' axis', fontsize=fontsize)
            ax[idx, 0].plot(x_ticks, curr_latent[:, 0], marker= marker_list[0], markersize= fontsize_lgd/1.2, linewidth=4, ls='-', label='Latent 1')
            ax[idx, 0].plot(x_ticks, curr_latent[:, 1], marker= marker_list[1], markersize= fontsize_lgd/1.2, linewidth=4, ls='--', label='Latent 2')
            ax[idx, 0].plot(x_ticks, curr_latent[:, 2], marker= marker_list[2], markersize= fontsize_lgd/1.2, linewidth=4, ls='-.', label='Latent 3')
            ax[idx, 0].plot(x_ticks, curr_latent[:, 3], marker= marker_list[3], markersize= fontsize_lgd/1.2, linewidth=4, ls=':', label='Latent 4')

            curr_latent= pred_latent[4, :, :]

            ax[idx, 1].grid(True) 
            ax[idx, 1].set_ylim(-3.5, 4.5)
            ax[idx, 1].tick_params(labelsize=fontsize)
#             ax[idx, 1].set_ylabel('Predicted Latents', fontsize=fontsize)
            ax[idx, 1].set_xlabel('Ball 2 moving along ' + axis  + ' axis', fontsize=fontsize)
            ax[idx, 1].plot(x_ticks, curr_latent[:, 0], marker= marker_list[0], markersize= fontsize_lgd/1.2, linewidth=4, ls='-', label='Latent 1')
            ax[idx, 1].plot(x_ticks, curr_latent[:, 1], marker= marker_list[1], markersize= fontsize_lgd/1.2, linewidth=4, ls='--', label='Latent 2')
            ax[idx, 1].plot(x_ticks, curr_latent[:, 2], marker= marker_list[2], markersize= fontsize_lgd/1.2, linewidth=4, ls='-.', label='Latent 3')
            ax[idx, 1].plot(x_ticks, curr_latent[:, 3], marker= marker_list[3], markersize= fontsize_lgd/1.2, linewidth=4, ls=':', label='Latent 4')

        lines, labels = fig.axes[-1].get_legend_handles_labels()    
        lgd= fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=fontsize_lgd, ncol=4)
        plt.tight_layout()
        plt.savefig(save_dir + 'latent_traversal_seed_' + str(seed) + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()
    
    #Latent Prediction Error
    rmse,r2= method.eval_identification()

    key= 'latent_pred_rmse'
    if key not in res.keys():
        res[key]=[]
    res[key].append(rmse)

    key= 'latent_pred_r2'
    if key not in res.keys():
        res[key]=[]
    res[key].append(r2)

    #Prediction RMSE
    key= 'recon_rmse'    
    if key not in res.keys():
        res[key]= []
    res[key].append(logs['recon_loss']['te'])
    
    if latent_case in ['balls_supp_l_shape', 'balls_supp_extrapolate', 'balls_supp_disconnected']:
        mcc= get_cross_correlation(copy.deepcopy(logs['pred_z']), copy.deepcopy(logs['true_z']), corr_case= 'pearson')
        key= 'mcc_pearson'
        if key not in res.keys():
            res[key]= []
        res[key].append(mcc)                    
        
        mcc= get_cross_correlation(copy.deepcopy(logs['pred_z']), copy.deepcopy(logs['true_z']), corr_case= 'spearman')
        key= 'mcc_spearman'
        if key not in res.keys():
            res[key]= []
        res[key].append(mcc)         
        
    else:
        if method_type in ['ae_base']:
            mcc= get_block_mcc_base_decoder(copy.deepcopy(logs['pred_z']), copy.deepcopy(logs['true_z']), total_blocks= total_blocks)            
        else:
            mcc= get_block_mcc(copy.deepcopy(logs['pred_z']), copy.deepcopy(logs['true_z']), total_blocks= total_blocks)
            print(mcc)
        key= 'mcc_block'
        if key not in res.keys():
            res[key]= []
        res[key].append(mcc)

if target_seed == -1:
    print('Dataframe')
    print(res)

    f= open(save_dir + 'logs.pickle', 'wb')
    pickle.dump(res, f)

    for key in res.keys():
        res[key]= np.array(res[key])
        print('Metric: ', key, np.mean(res[key]), np.std(res[key])/np.sqrt(num_seeds))
