import os
import sys
import random
import argparse
import torch
import numpy as np
import pygame
from pygame import gfxdraw, init
from typing import Callable, Optional
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

from scipy.stats import bernoulli
# range where the bump functions are evaluated
x_min, x_max = 0, 5
y_min, y_max = 0, 5

# Image size will be num_pixels x num_pixels
num_pixels = 64

# points where the bump functions will be evaluated
X_evals, Y_evals = np.meshgrid(np.linspace(x_min, x_max, num_pixels), np.linspace(y_min, y_max, num_pixels))

# colors
color1 = np.array([69,154,145])
color2 = np.array([203,107,102])
white = np.array([255,255,255])

def bump2D(x_center, y_center, x_eval, y_eval, temp=1):
    r2 = (x_eval - x_center)**2 + (y_eval - y_center)**2
    return np.where(np.sqrt(r2) < 1, np.exp(temp) * np.exp(-temp / (1 - r2)), 0)

def render_img(x1, y1, x2, y2):
    bump1 = bump2D(x1, y1, X_evals, Y_evals)
    bump2 = bump2D(x2, y2, X_evals, Y_evals)
    img = (1 - bump1[:,:, None] - bump2[:,:, None]) * white + bump1[:,:, None] * color1 + bump2[:, :, None] * color2
    return img.astype(np.int32)

def sample_L_distribution():
    # rejection sampling
    while True:
        y1 = np.random.uniform(0, 5)
        y2 = np.random.uniform(0, 5)
        x1= 5*0.25
        x2= 5*0.75
        if y1 <= 2.5 or y2 <= 2.5:
            return render_img(x1, 5 - y1, x2, 5 - y2), np.array([x1, y1, x2, y2])

def sample_extrapolate_distribution():
    # rejection sampling
    while True:
        y1 = np.random.uniform(0, 5)
        y2 = np.random.uniform(0, 5)
        x1= 5*0.25
        x2= 5*0.75
        if y1 > 2.5 and y2 > 2.5:
            return render_img(x1, 5 - y1, x2, 5 - y2), np.array([x1, y1, x2, y2])    
    
#Make scatter plot
def make_latent_support_plot(z, base_dir, data_case):
    
    f= base_dir + data_case + '_' + 'latent' + '.jpg'

    plt.scatter(z[:, 1], z[:, 3])
    plt.savefig(f)
    plt.clf()
    
    return

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100000,
                    help='')
parser.add_argument('--train_size', type=int, default=50000,
                    help='')
parser.add_argument('--test_size', type=int, default=50000,
                    help='')
parser.add_argument('--num_balls', type=int, default=2,
                    help='')
parser.add_argument('--latent_case', type=str, default='supp_l_shape', 
                   help='supp_extrapolate; supp_scm_linear; latent_traversal')

args = parser.parse_args()
seed= args.seed
train_size= args.train_size
test_size= args.test_size
latent_case= args.latent_case
num_balls= args.num_balls

base_dir= 'data/datasets/diff_balls_' + latent_case + '/' + 'num_balls_' + str(num_balls) + '/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

#Random Seed
random.seed(seed*10)
np.random.seed(seed*10) 
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed*10)

for data_case in ['train', 'val', 'test']: 
    print('Data Case: ', data_case)

    if data_case == 'train':
        dataset_size= args.train_size
    if data_case == 'val':
        dataset_size= int(args.train_size/4)
    elif data_case == 'test':
        dataset_size= args.test_size

    count=0
    final_z= []
    final_x= []
    for idx in range(dataset_size):
        
        #Sample image
        if latent_case == 'supp_l_shape':
            x, z = sample_L_distribution()
        elif latent_case == 'supp_extrapolate':
            x, z = sample_extrapolate_distribution()                
        
        z= np.expand_dims(z/5, axis= 0)
        x= np.expand_dims(x, axis= 0)            

        final_z.append(z)
        final_x.append(x)

        count+=1        
        if count >= dataset_size:
            break

    final_z= np.concatenate(final_z, axis=0)
    final_x= np.concatenate(final_x, axis=0)

    print(final_z.shape, final_x.shape)
    print(final_z[:5])
    
    #Make plot
    make_latent_support_plot(final_z, base_dir, data_case)    
    
    f= base_dir + data_case + '_' + 'x' + '.npy'
    np.save(f, final_x)

    f= base_dir + data_case + '_' + 'z' + '.npy'
    np.save(f, final_z)        
