import os
import sys
import argparse

latent_dim= 2
total_blocks= 2
learning_rate= 1e-3
batch_size= 64
train_size= 20000
plot_case= 'extrapolate'
latent_case= 'balls_supp_l_shape'
seed_grid= range(10)
input_normalization_grid= ['none']

method_grid= ['ae_base', 'ae_additive']
eval_latent_grid= ['balls_supp_l_shape', 'balls_supp_extrapolate']

#Disconnected support experiments
# latent_case= 'balls_supp_disconnected'
#method_grid= ['ae_base']

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--case', type=str, default='train',
                   help= '')
parser.add_argument('--curr_method', type=str, default='none',
                   help= '')
parser.add_argument('--curr_seed', type=int, default=-1,
                    help='')

args = parser.parse_args()
case= args.case
curr_method= args.curr_method
curr_seed= args.curr_seed

if case == 'train':
    for method in method_grid:    
        if method != curr_method and curr_method!='none':
            continue

        for seed in seed_grid:            
            if seed != curr_seed and curr_seed!=-1:
                continue
            
            for input_normalization in input_normalization_grid:
                script= 'sbatch scripts/slurm_launcher.sh' \
                        + ' ' + str(latent_case) \
                        + ' ' + method \
                        + ' ' + str(latent_dim) \
                        + ' ' + str(total_blocks) \
                        + ' ' + str(train_size) \
                        + ' ' + str(seed) \
                        + ' ' + str(input_normalization) \
                        + ' ' + str(batch_size) \
                        + ' ' + str(learning_rate) 
                os.system(script)
    
elif case == 'eval':
    for method in method_grid:    
        if method != curr_method and curr_method!='none':
            continue

        for eval_latent in eval_latent_grid:

            for input_normalization in input_normalization_grid:
                
                script= 'sbatch scripts/slurm_launcher_eval.sh' \
                        + ' ' + str(latent_case) \
                        + ' ' + method \
                        + ' ' + str(latent_dim) \
                        + ' ' + str(total_blocks) \
                        + ' ' + str(train_size) \
                        + ' ' + str(input_normalization) \
                        + ' ' + str(batch_size) \
                        + ' ' + str(learning_rate) \
                        + ' ' + eval_latent \
                        + ' ' + plot_case
                os.system(script)
