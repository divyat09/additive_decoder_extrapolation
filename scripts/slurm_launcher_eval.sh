#!/bin/sh
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1

#Force
slurm output
export PYTHONUNBUFFERED=1

#Load module
module load python/3.8

#Load python environment
source ~/clear_rep_env/bin/activate

python test.py --latent_case $1 --method_type $2 --latent_dim $3 --total_blocks $4 --train_size $5 --input_normalization $6 --batch_size $7 --lr $8 --eval_latent_case $9 --plot_case $10 --total_blocks 2 --cuda_device 0 --wandb_log 0
