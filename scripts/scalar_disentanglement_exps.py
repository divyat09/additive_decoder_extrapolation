import os
import sys
import argparse

latent_dim= 2
total_blocks= 2
latent_case= 'balls_supp_l_shape'
# latent_case= 'balls_supp_disconnected'

seed_grid= range(10)
train_size_grid= [1000]
method_grid= ['ae_base', 'ae_additive']

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--case', type=str, default='train',
                   help= '')
parser.add_argument('--curr_method', type=str, default='none',
                   help= '')
parser.add_argument('--curr_train_size', type=int, default= -1,
                    help='')
parser.add_argument('--curr_seed', type=int, default=-1,
                    help='')

args = parser.parse_args()
case= args.case
curr_method= args.curr_method
curr_train_size= args.curr_train_size
curr_seed= args.curr_seed

if case == 'train':
    for method in method_grid:    
        if method != curr_method and curr_method!='none':
            continue

        for train_size in train_size_grid:        
            if train_size != curr_train_size and curr_train_size!=-1:
                continue

            for seed in seed_grid:            
                if seed != curr_seed and curr_seed!=-1:
                    continue

                script= 'sbatch scripts/slurm_launcher.sh' + ' ' + str(latent_case) + ' ' + method + ' ' + str(latent_dim) + ' ' + str(total_blocks) + ' ' + str(train_size) + ' ' + str(seed)

                os.system(script)
    
elif case == 'eval':
    print('Not Implemented Yet')
    
elif case == 'log':
    print('Not Implemented Yet')
    