#!/bin/sh
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1

#Force
slurm output
export PYTHONUNBUFFERED=1

#Load module
module load python/3.9

#Load python environment
source ~/additive_env/bin/activate

python train.py --latent_case $1 --method_type $2 --latent_dim $3 --total_blocks $4 --train_size $5 --seed $6 --input_normalization $7 --batch_size $8 --lr $9 --num_epochs 6000 --cuda_device 0  --wandb_log 1
