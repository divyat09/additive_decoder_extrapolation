import os
import sys
import argparse

latent_dim= 4
total_blocks= 2
train_size= 50000

seed_grid= range(10)
method_grid= ['ae_base', 'ae_additive']
latent_grid= ['balls_supp_iid_no_occ', 'balls_supp_scm_linear']

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--case', type=str, default='train',
                   help= '')
parser.add_argument('--curr_method', type=str, default='none',
                   help= '')
parser.add_argument('--curr_latent', type=str, default='none',
                   help= '')
parser.add_argument('--curr_seed', type=int, default=-1,
                    help='')

args = parser.parse_args()
case= args.case
curr_method= args.curr_method
curr_seed= args.curr_seed
curr_latent= args.curr_latent

if case == 'train':
    for method in method_grid:    
        if method != curr_method and curr_method!='none':
            continue
        
        for latent_case in latent_grid:
            if latent_case != curr_latent and curr_latent!='none':
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
    