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

from utils.helper import ValidationHelper
from utils.metrics import *

import wandb

class AE():
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset, seed=0, device= None):
        
        self.args= args
        self.seed= seed
        self.device= device
        self.train_dataset= train_dataset
        self.val_dataset= val_dataset
        self.test_dataset= test_dataset
                
        self.encoder= Encoder(self.args.latent_dim).to(self.device)
        self.decoder= Decoder(self.args.latent_dim, device=self.device).to(self.device)
        
        self.opt, self.scheduler= self.get_optimizer()        
        self.validation_helper= ValidationHelper()
        
        self.res_dir= os.environ['SCRATCH'] + '/Additive_Decoder_Logs/'
        self.save_path= self.res_dir + str(self.args.method_type) + '/' + str(self.args.latent_case) + '/' + \
                      'num_balls_' + str(self.args.total_blocks)  + '/'  + \
                      'seed_' + str(seed) + '/' 
        
        if self.args.wandb_log:
            wandb.init(project="additive-decoder-identification", reinit=True)
            wandb.run.name=  'ae/' + self.save_path
            
    def get_optimizer(self):        
                    
        opt= optim.Adam([
                    {'params': filter(lambda p: p.requires_grad, list(self.encoder.parameters()) + list(self.decoder.parameters()) )}, 
                    ], lr= self.args.lr, weight_decay= self.args.weight_decay )
        
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)        
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
    
    def validation(self):
        
        self.encoder.eval()  
        self.decoder.eval()
        
        val_loss=0.0
        count=0
        for batch_idx, (x, _) in enumerate(self.val_dataset):
            
            with torch.no_grad():
                                
                x= x.to(self.device)                
                z_pred= self.encoder(x)                    
                out= self.decoder(z_pred)
                loss= torch.mean(((out-x)**2)) 
                    
                val_loss+= loss.item()
                count+=1            
        
        return val_loss/count
        
    def train(self):
        
        for epoch in range(self.args.num_epochs):            
                
            train_loss=0.0
            count=0
            
            #LR Scheduler
            self.scheduler.step()    
            print(self.scheduler.get_last_lr())
                                                    
            #Training            
            self.encoder.train()
            self.decoder.train()
            
            #Compute identification metrics
            if epoch % 10 == 0:
                self.eval_identification(epoch= epoch)
            
            for batch_idx, (x, _) in enumerate(self.train_dataset):
                
                self.opt.zero_grad()
                
                #Forward Pass
                x= x.to(self.device)
                z_pred= self.encoder(x)
                x_pred= self.decoder(z_pred)
                
                #Compute Reconstruction Loss
                loss= self.compute_loss(z_pred, x_pred, x)
                    
                #Backward Pass
                loss.backward()
                
                self.opt.step()
                    
                train_loss+= loss.item()
                count+=1                
            
            val_loss= self.validation()  
                                
            print('\n')
            print('Done Training for Epoch: ', epoch)
#             print('Training Loss: ', train_loss/count)   
#             print('Validation Loss: ', val_loss)
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
    
    def compute_loss(self, z_pred, x_pred, x):
        
        loss= torch.mean(((x-x_pred)**2))        
        return loss 
        
    
    def eval_identification(self, epoch= -1):
        
        self.encoder.eval()  
        self.decoder.eval()

        save_dir= 'results/plots/'+ self.args.latent_case + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)        
        
        res= get_predictions(self.encoder, self.decoder, self.train_dataset, self.val_dataset, self.test_dataset, device= self.device, save_dir= save_dir, plot=False)
        true_z= res['true_z']
        pred_z= res['pred_z']
        recon_err= res['recon_loss']        

        #Latent Prediction Error
        rmse, r2= get_indirect_prediction_error(pred_z, true_z)
#         rmse_mlp, r2_mlp= get_indirect_prediction_error(pred_z, true_z, model= 'mlp')

        #MCC
        mcc_pearson= -1
        mcc_spearman= -1
        mcc_block= -1
        if self.args.latent_case in ['balls_supp_l_shape', 'balls_supp_extrapolate', 'balls_supp_disconnected']:
            mcc_pearson= get_cross_correlation(copy.deepcopy(pred_z), copy.deepcopy(true_z), corr_case='pearson')
            mcc_spearman= get_cross_correlation(copy.deepcopy(pred_z), copy.deepcopy(true_z), corr_case='spearman')
        else:
            mcc_block= get_block_mcc(copy.deepcopy(pred_z), copy.deepcopy(true_z), total_blocks= self.args.total_blocks)
    
        if self.args.wandb_log:
            wandb.log({'test_loss': recon_err['te']})
            wandb.log({'latent_pred_rmse': rmse})
            wandb.log({'latent_pred_r2': r2})
#             wandb.log({'latent_pred_r2_mlp': r2_mlp})
            wandb.log({'mcc_pearson': mcc_pearson})
            wandb.log({'mcc_spearman': mcc_spearman})
            wandb.log({'mcc_block': mcc_block})
        
        return rmse, r2

    
    def get_final_layer_weights(self):
        
        self.load_model()
        
        for p in self.model.fc_net.parameters():
            print(p.data)
            