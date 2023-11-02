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
from models.base_decoder import BaseDecoder as Decoder

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

from utils.metrics import *

import wandb

class ValidationHelper:
    def __init__(self, patience: int =10000, min_delta: float=1e-4):
        """
        Inputs:
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum change in validation loss to be considered as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_epoch = -1
        
    def save_model(self, validation_loss, epoch):
        """
        Inputs:
            validation_loss: Validation loss of the current epoch
            epoch: Current epoch
        """
        if validation_loss < (self.min_validation_loss + self.min_delta):
            self.min_validation_loss = validation_loss
            self.best_epoch = epoch
            self.counter= 0
            return True
        return False

    def early_stop(self, validation_loss):        
        """
        Inputs:
            validation_loss: Validation loss of the current epoch
        """
        if validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class AE():
    
    def __init__(self, args, train_dataset: data_utils.DataLoader, val_dataset: data_utils.DataLoader, test_dataset: data_utils.DataLoader, seed: int=0, device= None):        
        self.args= args
        self.seed= seed
        self.device= device
        self.train_dataset= train_dataset
        self.val_dataset= val_dataset
        self.test_dataset= test_dataset
                
        self.encoder= Encoder(latent_dim= self.args.latent_dim).to(self.device)
        self.decoder= Decoder(latent_dim= self.args.latent_dim).to(self.device)
        
        self.opt, self.scheduler= self.get_optimizer()        
        self.validation_helper= ValidationHelper()
        
        self.res_dir= os.environ['SCRATCH'] + '/Additive_Decoder_Logs/' + str(args.input_normalization) + '/'
        self.save_path= self.res_dir +  str(self.args.method_type) + '/' + str(self.args.latent_case) + '/' + \
                      'num_balls_' + str(self.args.total_blocks)  + '/'  + \
                      'seed_' + str(seed) + '/'
        
        if self.args.wandb_log:
            wandb.init(project="additive-decoder-identification", reinit=True)
            wandb.run.name=  'ae/' + self.save_path
            
    def get_optimizer(self):        
                    
        opt= optim.Adam([
                    {'params': filter(lambda p: p.requires_grad, list(self.encoder.parameters()) + list(self.decoder.parameters()) )}, 
                    ], lr= self.args.lr, weight_decay= self.args.weight_decay )
        
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.75)
        return opt, scheduler
        
    def save_model(self):
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        filename= 'lr_' + str(self.args.lr) + \
                  '_weight_decay_' + str(self.args.weight_decay) + \
                  '_batch_size_' +  str(self.args.batch_size) + \
                  '_latent_dim_' + str(self.args.latent_dim)  + \
                  '_train_size_' + str(self.args.train_size)
        
        torch.save(self.encoder.state_dict(), self.save_path + filename +  '_encoder.pth')        
        
        torch.save(self.decoder.state_dict(), self.save_path + filename +  '_decoder.pth')        
        return
    
    def load_model(self):
        
        filename= 'lr_' + str(self.args.lr) + \
                  '_weight_decay_' + str(self.args.weight_decay) + \
                  '_batch_size_' +  str(self.args.batch_size) + \
                  '_latent_dim_' + str(self.args.latent_dim)  + \
                  '_train_size_' + str(self.args.train_size)
        
        self.encoder.load_state_dict(torch.load(self.save_path + filename + '_encoder.pth', map_location=torch.device('cpu')))
        self.encoder.eval()
        
        self.decoder.load_state_dict(torch.load(self.save_path + filename + '_decoder.pth', map_location=torch.device('cpu')))
        self.decoder.eval()
        return        
    
    def compute_loss(self, x_pred: torch.tensor, x: torch.tensor) -> torch.tensor:
        """
        Computes the mean squared loss.
        
        Inputs:
            x_pred: Predicted Image; expected shape (batch size, 3, 64, 64)
            x: Ground truth Image; expected shape (batch size, 3, 64, 64)
        """
        
        loss= torch.mean(((x-x_pred)**2))        
        return loss 
    
    def validation(self):
        """Compute validation loss."""
        
        self.encoder.eval()  
        self.decoder.eval()
        
        val_loss=0.0
        count=0
        for batch_idx, (x, _) in enumerate(self.val_dataset):
            
            with torch.no_grad():
                                
                x= x.to(self.device)                
                z_pred= self.encoder(x)                    
                x_pred= self.decoder(z_pred)
                loss= self.compute_loss(x_pred, x)
                    
                val_loss+= loss.item()
                count+=1            
        
        return val_loss/count
        
    def train(self):
        """Train the network and evaluate the performance on the test set after a certain interval."""
        
        for epoch in range(self.args.num_epochs):            
            
            train_loss=0.0
            count=0
            
            #LR Scheduler
            self.scheduler.step()    
            print(self.scheduler.get_last_lr())

            #Compute identification metrics
            if epoch % 250 == 0:
                self.eval_identification()
            
            #Training            
            self.encoder.train()
            self.decoder.train()
            
            for batch_idx, (x, _) in enumerate(self.train_dataset):
                
                self.opt.zero_grad()
                
                #Forward Pass
                x= x.to(self.device)
                z_pred= self.encoder(x)
                x_pred= self.decoder(z_pred)
                
                #Compute Reconstruction Loss
                loss= self.compute_loss(x_pred, x)
                    
                #Backward Pass
                loss.backward()
                
                self.opt.step()
                    
                train_loss+= loss.item()
                count+=1                
            
            val_loss= self.validation()  
                                
            print('\n')
            print('Done Training for Epoch: ', epoch)
            #print('Training Loss: ', train_loss/count)   
            #print('Validation Loss: ', val_loss)
            print('Best Epoch: ', self.validation_helper.best_epoch)   
            
            if self.args.wandb_log:
                wandb.log({'train_loss': train_loss/count})
                wandb.log({'val_loss': val_loss})
                        
            if self.validation_helper.save_model(val_loss, epoch):
                self.save_model()
            elif self.validation_helper.early_stop(val_loss):
                print('Early Stopping')
                break
            
        return
    
    
    def get_latent_regression_dataset(self):
        """Get the true and predicted latents for the train split that will be used for training regression function in MCC block computation.
        
            Returns:
                pred_latent, true_latent; both of shape (batch size, latent dim)
        """
        reg_feat= []
        reg_target= []
        for batch_idx, (x, z) in enumerate(self.train_dataset):
            with torch.no_grad():
                #Forward Pass
                x= x.to(self.device)
                z_pred= self.encoder(x)

                reg_feat.append(z_pred.cpu().detach().numpy())
                reg_target.append(z.cpu().detach().numpy())                

        reg_feat= np.concatenate(reg_feat, axis=0)
        reg_target= np.concatenate(reg_target, axis=0)
        
        return reg_feat, reg_target
    
    def eval_identification(self, seed: int=-1, save_dir: str= 'plots/', plot: bool= False) -> dict:
        """
        Compute reconstruction and latent identification metrics on the test set.

        Inputs:
            seed: Random seed used to initilaise the network parameters
            save_dir: Directory to store the results
            plot: Whether to plot the latent block renderings or not        
        """
        self.encoder.eval()  
        self.decoder.eval()

        recon_rmse= []
        mcc_pearson= []
        mcc_spearman= []
        mcc_block= []
        pred_z_arr= []
        true_z_arr= []
        reg_model_dict= {}
           
        for batch_idx, (x, z) in enumerate(self.test_dataset):
            with torch.no_grad():
                
                #Forward Pass
                x= x.to(self.device)
                pred= self.encoder(x)
                x_pred= self.decoder(pred)

                #Reconstruction Metrics
                for idx in range(x.shape[0]):
                    x[idx]= plot_transform(x[idx], input_normalization= self.args.input_normalization)

                for idx in range(x_pred.shape[0]):
                    x_pred[idx]= plot_transform(x_pred[idx], input_normalization= self.args.input_normalization)
                
                loss=  torch.sqrt( self.compute_loss(x_pred, x) )
                recon_rmse.append( loss.item() )
                
                #Latent Identification Metrics
                pred_z= pred.cpu().detach().numpy()
                true_z= z.cpu().detach().numpy()
                
                pred_z_arr.append(pred_z)
                true_z_arr.append(true_z)                
                
                #Block Rendering
                if plot and batch_idx == 0:                
                    width= 64
                    height= 64
                    padding= 10
                    final_image= Image.new('RGB', (3*padding + (width+padding)*10, 3*padding + (height+padding)*4))

                    for idx in range(10):                        
                        data= x[idx].cpu()
                        data= F.to_pil_image(data)            
                        final_image.paste(data, ( 2*padding +(width+padding) * idx, 2*padding + (height+padding)*0))

                        data= x_pred[idx].cpu()
                        data= F.to_pil_image(data)            
                        final_image.paste(data,  (2*padding + (width+padding) * idx, 2*padding + (height+padding)*1))
                        
                        total_blocks= 2
                        for block_idx in range(total_blocks):
                            x_pred_block= self.decoder.render_block(pred, block_idx)
                            data= x_pred_block[idx].cpu()
#                             data= plot_transform(data, input_normalization= self.args.input_normalization)
                            data= (data - data.min())/(data.max() - data.min())
                            data= F.to_pil_image(data)            
                            final_image.paste(data,  (2*padding + (width+padding) * idx, 2*padding + (height+padding)*(2+block_idx)))
                
                    final_image.save(save_dir + 'block_rendering_seed_' + str(seed) + '.jpg')
                    

                #MCC Pearson/Spearman/Block Computation for Latent Identification
                #Scalar Latent Case
                if self.args.latent_case in ['balls_supp_l_shape', 'balls_supp_extrapolate', 'balls_supp_disconnected', 'diff_balls_supp_l_shape', 'diff_balls_supp_extrapolate']:
                    mcc_pearson.append( get_cross_correlation(z_hat= copy.deepcopy(pred_z), z= copy.deepcopy(true_z), corr_case='pearson') )
                    mcc_spearman.append( get_cross_correlation(z_hat= copy.deepcopy(pred_z), z= copy.deepcopy(true_z), corr_case='spearman') ) 
                    
                #Block Latent Case
                else:
                    
                    if self.args.method_type in ['ae_base']:
                        partitions= [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]]                     
                    else:
                        partitions= [[0, 1, 2, 3]]
                    
                    mcc_partition= []
                    for p_idx, partition in enumerate(partitions):       
                        
                        #Regression funciton for computing the MCC
                        if str(p_idx) not in reg_model_dict.keys():
                            # print(batch_idx, 'Computing regression function')
                            reg_feat, reg_target= self.get_latent_regression_dataset()
                            reg_model_dict[str(p_idx)]= mcc_block_reg_model(
                                                            z_hat= reg_feat, 
                                                            z= reg_target,
                                                            partition= partition
                                                        )
                        
                        #MCC block with the current partition of predicted latents
                        mcc_partition.append( 
                                                get_block_mcc(
                                                            z_hat= copy.deepcopy(pred_z), 
                                                            z= copy.deepcopy(true_z),
                                                            total_blocks= self.args.total_blocks, 
                                                            partition= partition,
                                                            reg_model= reg_model_dict[str(p_idx)]
                                                            ) 
                                            )
                     
                    #Take the best score over partitions
                    mcc_block.append( np.max(mcc_partition) ) 

        recon_rmse= np.mean(recon_rmse)
        mcc_pearson= np.mean(mcc_pearson)
        mcc_spearman= np.mean(mcc_spearman)
        mcc_block= np.mean(mcc_block)
        pred_z_arr = np.concatenate(pred_z_arr, axis=0)
        true_z_arr = np.concatenate(true_z_arr, axis=0)

        if self.args.wandb_log:
            wandb.log({'test_loss': recon_rmse})
            wandb.log({'mcc_pearson': mcc_pearson})
            wandb.log({'mcc_spearman': mcc_spearman})
            wandb.log({'mcc_block': mcc_block})
        
        res={}
        res['recon_rmse']= recon_rmse
        res['mcc_pearson']= mcc_pearson
        res['mcc_spearman']= mcc_spearman
        res['mcc_block']= mcc_block
        print(res)
        res['pred_z']= pred_z_arr
        res['true_z']= true_z_arr
        print(pred_z_arr.shape, true_z_arr.shape)
        
        return res  
