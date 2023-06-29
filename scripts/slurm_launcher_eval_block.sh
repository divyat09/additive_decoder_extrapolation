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

python test.py --eval_latent_case $1 --latent_case $1 --method_type $2 --latent_dim $3  --total_blocks 2 --train_size 50000 --lr 5e-4 --batch_size 64 --cuda_device 0  --wandb_log 0 --plot_case latent_traversal