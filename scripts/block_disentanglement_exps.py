import os
import sys
import argparse

latent_dim= 4
total_blocks= 2
learning_rate= 1e-3
batch_size= 1024
train_size= 50000
plot_case= 'latent_traversal'
seed_grid= range(10)
input_normalization_grid= ['none']

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
curr_latent= args.curr_latent
curr_seed= args.curr_seed

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

        for latent in latent_grid:
            if latent != curr_latent and curr_latent!='none':
                continue

            for seed in seed_grid:            
                if seed != curr_seed and curr_seed!=-1:
                    continue
                
                for input_normalization in input_normalization_grid:
                    script= 'sbatch scripts/slurm_launcher.sh' \
                            + ' ' + str(latent) \
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

        for latent in latent_grid:
            if latent != curr_latent and curr_latent!='none':
                continue

            for input_normalization in input_normalization_grid:
                
                script= 'sbatch scripts/slurm_launcher_eval.sh' \
                        + ' ' + latent \
                        + ' ' + method \
                        + ' ' + str(latent_dim) \
                        + ' ' + str(total_blocks) \
                        + ' ' + str(train_size) \
                        + ' ' + str(input_normalization) \
                        + ' ' + str(batch_size) \
                        + ' ' + str(learning_rate) \
                        + ' ' + latent \
                        + ' ' + plot_case
                os.system(script)
