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
from PIL import Image
from itertools import combinations


def invert_transform(data):

    mean_norm= [0.485, 0.456, 0.406]
    sem_norm= [0.229, 0.224, 0.225]
    for idx in range(3):
        data[:, idx, :, : ] = data[:, idx, :, : ] * sem_norm[idx] + mean_norm[idx]

    data= (data - data.min()) / (data.max() - data.min())
    return data

def regression_approx(x, y, model, fit_intercept=False):
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

def get_predictions_check(train_dataset, test_dataset):    
    
    true_x={'tr':[], 'te':[]}
    true_z= {'tr':[], 'te':[]}
    
    data_case_list= ['train', 'test']
    for data_case in data_case_list:
        
        if data_case == 'train':
            dataset= train_dataset
            key='tr'
        elif data_case == 'test':
            dataset= test_dataset
            key='te'
    
        for batch_idx, (x, z, _) in enumerate(dataset):

            with torch.no_grad():
                                
                true_x[key].append(x)
                true_z[key].append(z)

        true_x[key]= torch.cat(true_x[key]).detach().numpy()
        true_z[key]= torch.cat(true_z[key]).detach().numpy()
    
    return true_x, true_z


def get_predictions(encoder, decoder, train_dataset, val_dataset, test_dataset, device= None, save_dir='plots/', seed=-1,  plot=True):
        
    true_z= {'tr':[], 'val':[], 'te':[]}
    pred_z= {'tr':[], 'val':[], 'te':[]}
    recon_loss= {'tr':0.0, 'val':0.0, 'te':0.0}
    
    data_case_list= ['train', 'val', 'test']
    for data_case in data_case_list:
        
        if data_case == 'train':
            dataset= train_dataset
            key='tr'
        elif data_case == 'val':
            dataset= val_dataset
            key='val'
        elif data_case == 'test':
            dataset= test_dataset
            key='te'
        
        count=0
        for batch_idx, (x, z) in enumerate(dataset):
            with torch.no_grad():
                
                #Forward Pass
                x= x.to(device)
                pred= encoder(x)
                x_pred= decoder(pred)

                #Comuting Reconstruction Loss with reverse normalization to keep the statistics sane
                # loss= torch.mean((x-x_pred)**2)
                loss= torch.mean( (  invert_transform(x) - invert_transform(x_pred) )**2 )

                true_z[key].append(z)
                pred_z[key].append(pred)
                recon_loss[key]+= loss.item()
                count+=1                
                
                width= 64
                height= 64
                padding= 10
                final_image= Image.new('RGB', (3*padding + (width+padding)*10, 3*padding + (height+padding)*4))
                
                #Block Rendering
                if plot and data_case == "test" and batch_idx == 0:
                    for idx in range(10):
                        
                        transform = transforms.Compose(
                        [
                            transforms.ToPILImage(),
                        ]
                        )
                        
                        data= x[idx].cpu()
                        data= (data - data.min()) / (data.max() - data.min())
                        data= transform(data)            
                        # data.save( save_dir  + 'real_image_' + str(idx) + '_seed_' + str(seed) + '.jpg')
                        final_image.paste(data, ( 2*padding +(width+padding) * idx, 2*padding + (height+padding)*0))

                        data= x_pred[idx].cpu()
                        data= (data - data.min()) / (data.max() - data.min())
                        data= transform(data)
                        # data.save( save_dir  + 'fake_image_' + str(idx) + '_seed_' + str(seed) + '.jpg')
                        final_image.paste(data,  (2*padding + (width+padding) * idx, 2*padding + (height+padding)*1))
                        
                        total_blocks= 2
                        for block_idx in range(total_blocks):
                            x_pred_block= decoder.render_block(pred, block_idx)
                            data= x_pred_block[idx].cpu()
                            data= (data - data.min()) / (data.max() - data.min())
                            data= transform(data)
                            # data.save( save_dir  + 'fake_image_block_' + str(block_idx) + '_' + str(idx) + '_seed_' + str(seed) + '.jpg')
                            final_image.paste(data,  (2*padding + (width+padding) * idx, 2*padding + (height+padding)*(2+block_idx)))
               
                    final_image.save(save_dir + 'block_rendering_seed_' + str(seed) + '.jpg')                

        true_z[key]= torch.cat(true_z[key]).detach().numpy()
        pred_z[key]= torch.cat(pred_z[key]).cpu().detach().numpy()
        recon_loss[key]= recon_loss[key]/count
    
#     print('Sanity Check: ')
#     print( true_y['tr'].shape, pred_y['tr'].shape, true_z['tr'].shape, pred_z['tr'].shape )
#     print( true_y['te'].shape, pred_y['te'].shape, true_z['te'].shape, pred_z['te'].shape )
    return {'true_z': true_z, 'pred_z': pred_z, 'recon_loss': recon_loss}


def get_indirect_prediction_error(pred_latent, true_score, case='test', model='lr'):
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'
                
    reg= regression_approx(pred_latent['tr'], true_score['tr'], model, fit_intercept=True)
    pred_score= reg.predict(pred_latent[key])
    if len(pred_score.shape) == 1:
        pred_score= np.reshape(pred_score, (pred_score.shape[0], 1))
    
    rmse= np.sqrt(np.mean((true_score[key] - pred_score)**2))
    r2= r2_score(true_score[key], pred_score)  
        
#     mat= reg.coef_
#     _, sig_values ,_ = np.linalg.svd(mat)
#     print(mat)
#     print(np.mean(pred_latent['tr']), np.var(pred_latent['tr']))
#     sys.exit()
            
    return rmse, r2


def get_cross_correlation(pred_latent, true_latent, case='test', batch_size= 5000, corr_case='pearson'):
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'
    
        
    num_samples= pred_latent[key].shape[0]
    dim= pred_latent[key].shape[1]
    total_batches= int( num_samples / batch_size )  

    mcc_arr= []
    for batch_idx in range(total_batches):
        
        z_hat= copy.deepcopy( pred_latent[key][ (batch_idx)*batch_size : (batch_idx+1)*batch_size ] )
        z= copy.deepcopy( true_latent[key][ (batch_idx)*batch_size : (batch_idx+1)*batch_size ] )
        batch_idx += 1
        
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
    
        mcc_arr.append(score)
    
    return np.mean(mcc_arr)



def get_block_mcc(pred_latent, true_latent, case='test', batch_size= 5000, reg_case='d-tree', total_blocks=2):
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'    
        
    total_batches= int(pred_latent[key].shape[0]/batch_size)
    feat_block_size= int(pred_latent[key].shape[1]/total_blocks)
    target_block_size= int(true_latent[key].shape[1]/total_blocks)

    mcc_arr= []
    for batch_idx in range(total_batches):
        
        z_hat= copy.deepcopy( pred_latent[key][ (batch_idx)*batch_size : (batch_idx+1)*batch_size ] )
        z= copy.deepcopy( true_latent[key][ (batch_idx)*batch_size : (batch_idx+1)*batch_size ] )
        batch_idx += 1
        
        cross_corr= np.zeros((total_blocks, total_blocks))
        for i in range(total_blocks):
            for j in range(total_blocks):
                curr_feat= z_hat[:, i*feat_block_size : (i+1)*feat_block_size ]
                curr_target= z[:, j*target_block_size : (j+1)*target_block_size ]
                
                reg_model= regression_approx(curr_feat, curr_target, reg_case)
                score= r2_score( curr_target, reg_model.predict(curr_feat) )
                if score < 0:
                    score= 0.0
                cross_corr[i, j]= score

        cost= -1*np.abs(cross_corr)
        row_ind, col_ind= linear_sum_assignment(cost)
        score= 100*( -1*cost[row_ind, col_ind].sum() )/(total_blocks)
    
        mcc_arr.append(score)
    
    return np.mean(mcc_arr)


def generate_partitions(indices, partition_size):
    partitions = []

    for indices_partition in combinations(indices, partition_size):
        
        remaining_indices = np.setdiff1d(indices, list(indices_partition))
        
        if remaining_indices.shape[0] > 2:
            sub_paritions= generate_partitions(remaining_indices, 2)
            for sub_partition in sub_partitions:
                partitions.append(np.concatenate([indices_partition, sub_partition], axis=0))                
        else:
            partitions.append(np.concatenate([indices_partition, remaining_indices], axis=0))
    
    return partitions


def get_block_mcc_base_decoder(pred_latent, true_latent, case='test', batch_size= 5000, reg_case='d-tree', total_blocks=2):
    
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'    
        
#     feature_dim= pred_latent[key].shape[1]    
#     partitions= generate_partitions(list(range(feature_dim)), 2)
    print('Here')
    partitions= [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2], [1, 2, 0, 3], [1, 3, 0, 2], [2, 3, 0, 1]]
    
    mcc_arr=[]
    for partition in partitions:
        curr_pred_latent= {}
        curr_pred_latent[key]= pred_latent[key][:, partition]
        mcc= get_block_mcc(curr_pred_latent, true_latent, case= case, batch_size= batch_size, reg_case= reg_case, total_blocks= total_blocks)
        mcc_arr.append(mcc)
    
    mcc_arr= np.array(mcc_arr)
    print(mcc_arr)
    
    return np.max(mcc_arr)

def compute_importance_matrix(z_pred, z, case='disentanglement', lasso=False, fit_intercept=True):
    true_latent_dim = z.shape[1]
    pred_latent_dim = z_pred.shape[1]
    imp_matrix = np.zeros((pred_latent_dim, true_latent_dim))
    for idx in range(true_latent_dim):
        if lasso:
            model = LassoCV(fit_intercept=fit_intercept, cv=3, tol=1e-10).fit(z_pred, z[:, idx])
        else:
            model = LinearRegression(fit_intercept=fit_intercept).fit(z_pred, z[:, idx])
        imp_matrix[:, idx] = model.coef_

    # Taking the absolute value for weights to encode relative importance properly
    imp_matrix = np.abs(imp_matrix)
    if case == 'disentanglement':
        imp_matrix = imp_matrix / np.reshape(np.sum(imp_matrix, axis=1), (pred_latent_dim, 1))
    elif case == 'completeness':
        imp_matrix = imp_matrix / np.reshape(np.sum(imp_matrix, axis=0), (1, true_latent_dim))

    return imp_matrix


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                    base=importance_matrix.shape[1])


def disentanglement(z_pred, z, lasso=False):
    """Compute the disentanglement score of the representation."""
    importance_matrix = compute_importance_matrix(z_pred, z, case='disentanglement', lasso=lasso, fit_intercept=True)
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                    base=importance_matrix.shape[0])


def completeness(z_pred, z, lasso=False):
    """"Compute completeness of the representation."""
    importance_matrix = compute_importance_matrix(z_pred, z, case='completeness', lasso=lasso, fit_intercept=True)
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)