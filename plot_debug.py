import torch
import torchvision.transforms.functional as F
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def invert_imagenet_transform(data: torch.Tensor) -> torch.Tensor:
    """
    Inverts the imagenet normalization transform.

    Inputs:
        data: Image to be transformed; Expected shape: (3, 64, 64)
    Returns:
        Transformed image; Expected shape: (3, 64, 64)        
    """

    mean_norm= [0.485, 0.456, 0.406]
    sem_norm= [0.229, 0.224, 0.225]
    for idx in range(3):
        data[idx, :, : ] = data[idx, :, : ] * sem_norm[idx] + mean_norm[idx]

    return data

def plot_transform(img: torch.Tensor, input_normalization: str='none') -> torch.Tensor:
    """
    Converts each image to [0, 1] range for plotting.

    Inputs:
        data: img to be transformed; Expected shape: (3, 64, 64)
    Returns:
        Transformed image; Expected shape: (3, 64, 64)        
    """

    if input_normalization == 'none':
         img= img/255

    elif input_normalization == 'imagenet':
        img= invert_imagenet_transform(img)
        img= img/255
 
    elif input_normalization == 'tensor_uniform':
        img= (img+1)/2

    elif input_normalization == 'tensor_imagenet': 
        img= invert_imagenet_transform(img)

    img= torch.clamp(img, 0, 1)
    return img


def train_transform(img: torch.tensor, input_normalization: str='none') -> torch.tensor:

    if input_normalization == 'none':
        img= torch.tensor(img, dtype=torch.float)
        img= img.permute(2, 0, 1)

    elif input_normalization == 'imagenet':
        img= torch.tensor(img, dtype=torch.float)
        img= img.permute(2, 0, 1)
        data_transform=  transforms.Compose([
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        img= data_transform(img)

    elif input_normalization == 'tensor':
        data_transform=  transforms.Compose([
                            transforms.ToTensor(),
                        ])
        img= data_transform(img)

    elif input_normalization == 'tensor_uniform':
        data_transform=  transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])
        img= data_transform(img)


    elif input_normalization == 'tensor_imagenet':
        data_transform=  transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        img= data_transform(img)

    return img

#Main Code
for input_normalization in ['none', 'imagenet', 'tensor', 'tensor_uniform', 'tensor_imagenet']:

    #Load Image
    x= np.load('data/datasets/balls_supp_l_shape/num_balls_2/' + 'train' +  '_' + 'x' + '.npy')
    img= x[0]

    #Process Image
    img_train= train_transform(img, input_normalization=input_normalization)
    img_plot= plot_transform(img_train, input_normalization= input_normalization)

    #Save Image
    img_pl= F.to_pil_image(img_plot)
    img_pl.save('plots/'+ input_normalization +'.png')
