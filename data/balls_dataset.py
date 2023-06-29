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

if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    

COLOURS_ = [
    [2, 156, 154],
    [222, 100, 100],
    [149, 59, 123],
    [74, 114, 179],
    [27, 159, 119],
    [218, 95, 2],
    [117, 112, 180],
    [232, 41, 139],
    [102, 167, 30],
    [231, 172, 2],
    [167, 118, 29],
    [102, 102, 102],
    ]

SCREEN_DIM = 64

def circle(
    x_,
    y_,
    surf,
    color=(204, 204, 0),
    radius=0.1,
    screen_width=SCREEN_DIM,
    y_shift=0.0,
    offset=None,
):
    if offset is None:
        offset = screen_width / 2
    scale = screen_width
    x = scale * x_ + offset
    y = scale * y_ + offset

    gfxdraw.aacircle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )
    gfxdraw.filled_circle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )    


class Balls(torch.utils.data.Dataset):
    
    ball_rad = 2.0*0.08
    screen_dim = 64    

    def __init__(
        self,
        transform: Optional[Callable] = None,
        n_balls: int = 1,
    ):
        super(Balls, self).__init__()
        if transform is None:

            def transform(x):
                return x

        self.transform = transform
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.n_balls = n_balls

    def __len__(self) -> int:
        # arbitrary since examples are generated online.
        return 20000

    def draw_scene(self, z):
        self.surf.fill((255, 255, 255))
        if z.ndim == 1:
            z = z.reshape((1, 2))
        for i in range(z.shape[0]):
            circle(
                z[i, 0],
                z[i, 1],
                self.surf,
                color=COLOURS_[i],
                radius=self.ball_rad,
                screen_width=self.screen_dim,
                y_shift=0.0,
                offset=0.0,
            )
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def __getitem__(self, item):
        raise NotImplemented()


class BlockLatent(Balls):
    def __init__(
        self,
        transform: Optional[Callable] = None,
        n_balls: int = 2,
        latent_case: str = 'iid',
    ):
        super().__init__(transform=transform, n_balls=n_balls)
        self.dataset_size= 20000
        self.latent_dim= self.n_balls*2
        self.latent_case= latent_case
        
    def __getitem__(self, item):

        if self.latent_case == 'iid':
            z = self.get_observational_data_iid()
        elif 'scm' in self.latent_case:
            z = self.get_observational_data_scm()
        elif self.latent_case == 'supp_l_shape':
            z = self.get_l_support_data()
        elif self.latent_case == 'supp_extrapolate':
            z = self.get_extrapolation_data()
        elif self.latent_case == 'supp_disconnected':
            support_case= bernoulli.rvs(0.5, size=1)[0]
            z= self.get_disconnected_support_data(support_case)
        elif self.latent_case == 'supp_u_shape':
            support_case= bernoulli.rvs(0.5, size=1)[0]
            z= self.get_u_support_data(support_case)
        elif self.latent_case == 'supp_iid_no_occ':
            z= self.get_supp_iid_no_occlusion()
        else:
            print('Latent type not supported')
            sys.exit()
            
        x = self.draw_scene(z)
        x = self.transform(x)
        
        return z.flatten(), x

    def get_u_support_data(self, support_case):
        
        x1= np.random.uniform(0.25, 0.25, size=1)[0]
        x2= np.random.uniform(0.75, 0.75, size=1)[0]
        
        y1= np.random.uniform(0.1, 0.9, size=1)[0]  
        
        # y1 less than 0.1 + (0.9-0.1)/4 -> 0.3
        if y1 < 0.3:
            y2= np.random.uniform(0.1, 0.9, size=1)[0]
        else:
            if support_case:
                y2= np.random.uniform(0.1, 0.34, size=1)[0]
            else:
                y2= np.random.uniform(0.65, 0.9, size=1)[0]

        z= np.array([[x1, y1], [x2, y2]])
        return z    
    
    def get_disconnected_support_data(self, support_case):
        
        x1= np.random.uniform(0.25, 0.25, size=1)[0]
        x2= np.random.uniform(0.75, 0.75, size=1)[0]
        
        y1= np.random.uniform(0.1, 0.9, size=1)[0]
        
        if support_case:
            y2= np.random.uniform(0.1, 0.25, size=1)[0]
        else:
            y2= np.random.uniform(0.75, 0.9, size=1)[0]

        z= np.array([[x1, y1], [x2, y2]])
        return z    
    
    def get_l_support_data(self):
        
        x1= np.random.uniform(0.25, 0.25, size=1)[0]
        x2= np.random.uniform(0.75, 0.75, size=1)[0]
        
        y1= np.random.uniform(0.1, 0.9, size=1)[0]  
        
        if y1 < 0.5:
            y2= np.random.uniform(0.1, 0.9, size=1)[0]
        else:
            y2= np.random.uniform(0.1, 0.50, size=1)[0]

        z= np.array([[x1, y1], [x2, y2]])
        return z    
    
    def get_extrapolation_data(self):
        
        x1= np.random.uniform(0.25, 0.25, size=1)[0]
        x2= np.random.uniform(0.75, 0.75, size=1)[0]
        
        y1= np.random.uniform(0.5, 0.9, size=1)[0]  
        y2= np.random.uniform(0.5, 0.9, size=1)[0]

        z= np.array([[x1, y1], [x2, y2]])
        return z
        
    def get_supp_iid_no_occlusion(self):
        
        dist= 0        
        while(dist < (2*self.ball_rad)**2):
            z= np.random.uniform(0.1, 0.9, size=(self.n_balls, 2))
            dist= (z[0, 0] - z[1, 0])**2 + (z[0, 1] - z[1, 1])**2
        
        return z

    def get_supp_iid(self):
        
        z= np.random.uniform(0.1, 0.9, size=(self.n_balls, 2))
        
        return z
    
    def get_observational_data_scm(self):
        
        dist= 0        
        while(dist < (2*self.ball_rad)**2):
            
            print('yes')
        
            x1= np.random.uniform(0.1, 0.9, size=1)[0]        
            y1= np.random.uniform(0.1, 0.9, size=1)[0]

            if self.latent_case == 'supp_scm_linear':
                constraint= x1+y1
            elif self.latent_case == 'supp_scm_non_linear':
                constraint= 1.25 * (x1**2 + y1**2)

            if constraint >= 1.0:
                x2= np.random.uniform(0.1, 0.5, size=1)[0]
                y2= np.random.uniform(0.5, 0.9, size=1)[0]
            else:
                x2= np.random.uniform(0.5, 0.9, size=1)[0]
                y2= np.random.uniform(0.1, 0.5, size=1)[0]

            z= np.array([[x1, y1], [x2, y2]])
            dist= (z[0, 0] - z[1, 0])**2 + (z[0, 1] - z[1, 1])**2
        
        return z

    def sample_supp_grid(self, movement_axis= 'x_axis'):
        
        final_z= []
        final_x= []
        
        for i in range(1, 10):            
            x1= np.random.uniform(0.25, 0.25, size=1)[0]
            x2= np.random.uniform(0.75, 0.75, size=1)[0]
            
            y1= np.random.uniform(i*0.1, i*0.1, size=1)[0]
            for j in range(1, 10):
                y2= np.random.uniform( j*0.1, j*0.1, size=1)[0]
                
                if movement_axis == 'x_axis':
                    z= np.array([[y1, x1], [y2, x2]])
                elif movement_axis == 'y_axis':
                    z= np.array([[x1, y1], [x2, y2]])       
                
                x= self.draw_scene(z)
                x= self.transform(x)
                
                final_z.append( torch.tensor(z.flatten()) )                
                final_x.append( torch.tensor(x) )
        
        final_z= torch.stack(final_z).float()        
        final_x= torch.stack(final_x).float()  
        final_x= final_x.permute(0, 3, 1, 2)        
        data_transform=  transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])  
        
        final_x= data_transform(final_x)        
        print(final_z)
        
        return final_z, final_x    
    
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

base_dir= 'data/datasets/balls_' + latent_case + '/' + 'num_balls_' + str(num_balls) + '/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

#Random Seed
random.seed(seed*10)
np.random.seed(seed*10) 

data_obj= BlockLatent(latent_case= latent_case, n_balls= num_balls)

if latent_case == 'latent_traversal':

    base_dir= 'data/datasets/balls_latent_traversal/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    z, x= data_obj.sample_supp_grid(movement_axis= 'x_axis')
    f= base_dir + 'supp_grid_x_axis_' + 'x' + '.pt'
    torch.save(x, f)
    f= base_dir + 'supp_grid_x_axis_' + 'z' + '.pt'
    torch.save(z, f)

    z, x= data_obj.sample_supp_grid(movement_axis= 'y_axis')
    f= base_dir + 'supp_grid_y_axis_' + 'x' + '.pt'
    torch.save(x, f)
    f= base_dir + 'supp_grid_y_axis_' + 'z' + '.pt'
    torch.save(z, f)

    sys.exit()

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
    for batch_idx, (z, x) in enumerate(data_obj):

        z= np.expand_dims(z, axis= 0)
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
