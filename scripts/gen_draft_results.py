#Common imports
import sys
import os
import argparse
import random
import copy
import pickle

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import matplotlib.colors as mcolors

import torch
import torch.utils.data as data_utils
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from PIL import Image

import colorsys

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

def gen_line_plot(res, eval_latent_case, train_size_grid):
    
    fontsize=40
    fontsize_lgd= fontsize/1.5
    marker_list = ['o', '^', 's']
    x_ticks= train_size_grid

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].tick_params(labelsize=fontsize)
    ax[0].set_ylabel('Recontruction Error', fontsize=fontsize)
    ax[0].set_xlabel('Train Size', fontsize=fontsize)

    ax[0].errorbar( x_ticks, 
                    res[eval_latent_case]['ae_base']['recon_err']['mean'] ,
                    yerr= res[eval_latent_case]['ae_base']['recon_err']['sem'], 
                    marker= marker_list[0], markersize= fontsize_lgd, 
                    linewidth=4, fmt='o--', 
                    label='Non-Additive Decoder'
                    )

    ax[0].errorbar( x_ticks, 
                    res[eval_latent_case]['ae_additive']['recon_err']['mean'] , 
                    yerr= res[eval_latent_case]['ae_additive']['recon_err']['sem'], 
                    marker= marker_list[1], markersize= fontsize_lgd, 
                    linewidth=4, fmt='o--', 
                    label='Additive Decoder'
                    )

    ax[1].tick_params(labelsize=fontsize)
    ax[1].set_ylabel('MCC', fontsize=fontsize)
    ax[1].set_xlabel('Train Size', fontsize=fontsize)
    ax[1].set(ylim=(50, 100))
    ax[1].errorbar( x_ticks, 
                   res[eval_latent_case]['ae_base']['mcc']['mean'], 
                   yerr= res[eval_latent_case]['ae_base']['mcc']['sem'] , 
                   marker= marker_list[0], markersize= fontsize_lgd, 
                   linewidth=4, fmt='o--', 
                   label='Non-Additive Decoder'
                   )
    
    ax[1].errorbar( x_ticks, 
                   res[eval_latent_case]['ae_additive']['mcc']['mean'], 
                   yerr=res[eval_latent_case]['ae_additive']['mcc']['sem'], 
                   marker= marker_list[1], markersize= fontsize_lgd, 
                   linewidth=4, fmt='o--', 
                   label='Additive Decoder'
                   )

    lines, labels = fig.axes[-1].get_legend_handles_labels()    
    lgd= fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=fontsize_lgd, ncol=2)
    plt.tight_layout()

    if eval_latent_case == 'balls_supp_l_shape':
        title= 'In Support'
        save_name= 'Non_Block_In_Support'
    elif eval_latent_case == 'balls_supp_extrapolate':
        title= 'OOD Support'
        save_name= 'Non_Block_OOD_Support'

    plt.title(title)
    plt.savefig( 'results/' + save_name  +  '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight',  dpi=600)

    return

def gen_violin_plot(res, eval_case_grid, labels, save_name, ylabel):

    fontsize=40
    fontsize_lgd= fontsize/1.5
    marker_list = ['o', '^', 's', '*']

    data= {
            'recon_err': [],
            'mcc': []
        }   

    metrics= data.keys()
    methods= ['ae_additive', 'ae_base']

    # create a color palette with two different colors
    colors = [ (1, 0.6, 0.6,),
               (0.75, 0.75, 1,), 
               (1, 0.3, 0.3,),
               (0.3, 0.3, 1),
             ]

    for metric in metrics:
        for eval_case in eval_case_grid:
            for method in methods:
                data[metric].append( res[method][eval_case][metric] )

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].set_yscale('log')
    ax[0].tick_params(labelsize=fontsize)
    ax[0].grid(True)
    ax[0].set_title('Reconstruction MSE (Log Scale)', fontsize=fontsize)
    ax[0].set_ylim( pow(10, -3), 3 * pow(10, -1) )
#     ax[0].set_xticks(np.arange(1, len(labels) + 1), labels)
#     ax[0].set_xticklabels(labels, rotation = 45, fontsize= fontsize_lgd)
    
    plot_data= np.array(data['recon_err'])
    plot_data= np.reshape( plot_data, (plot_data.shape[0], plot_data.shape[2]) ).T
    sns.stripplot(data=plot_data, color='black', jitter=0.0, ax=ax[0], size= 20)
    sns.boxplot(data=plot_data, ax=ax[0], palette= colors, showfliers=False, linewidth=8, width=0.45, medianprops=dict(color="black", alpha=0.7))
    ax[0].set_xticks([])
    

    ax[1].tick_params(labelsize=fontsize)
    ax[1].grid(True)
    ax[1].set_title(ylabel, fontsize=fontsize)
    ax[1].set(ylim=(45, 105))
#     ax[1].set_xticks(np.arange(1, len(labels) + 1), labels)
#     ax[1].set_xticklabels(labels, rotation = 45, fontsize= fontsize_lgd)

    plot_data= np.array(data['mcc'])
    plot_data= np.reshape( plot_data, (plot_data.shape[0], plot_data.shape[2]) ).T
    sns.stripplot(data=plot_data, color='black', jitter=0.0, ax=ax[1], size= 20)
    violin= sns.boxplot(data=plot_data, ax=ax[1], palette= colors, showfliers=False, linewidth=8, width=0.45, medianprops=dict(color="black", alpha=0.7))
    for i, label in enumerate(labels):
        violin.collections[i].set_label(label)
    ax[1].set_xticks([])

    handles, labels = fig.axes[-1].get_legend_handles_labels()
#     plt.tight_layout()
#     plt.savefig( 'results/' + save_name  +  '.pdf', dpi=600)
    
    handles = [mpatches.Patch(color=colors[idx]) for idx, label in enumerate(labels)]    
    lgd= fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize= fontsize_lgd)
    
    plt.savefig( 'results/' + save_name  +  '.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)

    return 


def gen_mini_violin_plot(res, eval_case_grid, labels, save_name, ylabel):

    fontsize=40
    fontsize_lgd= fontsize/1.5
    marker_list = ['o', '^', 's', '*']

    data= {
            'recon_err': [],
            'mcc': []
        }   

    metrics= data.keys()
    methods= ['ae_additive', 'ae_base']

    # create a color palette with two different colors
    colors = [ (1, 0.6, 0.6,),
               (0.75, 0.75, 1,), 
             ]

    for metric in metrics:
        for eval_case in eval_case_grid:
            for method in methods:
                data[metric].append( res[method][eval_case][metric] )

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].set_yscale('log')
    ax[0].tick_params(labelsize=fontsize)
    ax[0].grid(True)
    ax[0].set_title('Reconstruction MSE (Log Scale)', fontsize=fontsize)
    ax[0].set_ylim( pow(10, -3), 3 * pow(10, -1) )
#     ax[0].set_xticks(np.arange(1, len(labels) + 1), labels)
#     ax[0].set_xticklabels(labels, rotation = 45, fontsize= fontsize_lgd)
    
    plot_data= np.array(data['recon_err'])
    plot_data= np.reshape( plot_data, (plot_data.shape[0], plot_data.shape[2]) ).T
    sns.stripplot(data=plot_data, color='black', jitter=0.0, ax=ax[0], size= 20)
    sns.boxplot(data=plot_data, ax=ax[0], palette= colors, showfliers=False, linewidth=8, width=0.45, medianprops=dict(color="black", alpha=0.7))
    ax[0].set_xticks([])
    

    ax[1].tick_params(labelsize=fontsize)
    ax[1].grid(True)
    ax[1].set_title(ylabel, fontsize=fontsize)
    ax[1].set(ylim=(45, 105))
#     ax[1].set_xticks(np.arange(1, len(labels) + 1), labels)
#     ax[1].set_xticklabels(labels, rotation = 45, fontsize= fontsize_lgd)

    plot_data= np.array(data['mcc'])
    plot_data= np.reshape( plot_data, (plot_data.shape[0], plot_data.shape[2]) ).T
    sns.stripplot(data=plot_data, color='black', jitter=0.0, ax=ax[1], size= 20)
    violin= sns.boxplot(data=plot_data, ax=ax[1], palette= colors, showfliers=False, linewidth=8, width=0.45, medianprops=dict(color="black", alpha=0.7))
    for i, label in enumerate(labels):
        violin.collections[i].set_label(label)
    ax[1].set_xticks([])

    handles, labels = fig.axes[-1].get_legend_handles_labels()
#     plt.tight_layout()
#     plt.savefig( 'results/' + save_name  +  '.pdf', dpi=600)
    
    handles = [mpatches.Patch(color=colors[idx]) for idx, label in enumerate(labels)]    
    lgd= fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize= fontsize_lgd)
    
    plt.savefig( 'results/' + save_name  +  '.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)

    return 

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--results_case', type=str, default='violin_plot',
                   help= 'violin_plot, mcc_table, extrapolation, block_rendering, latent_traversal')
parser.add_argument('--method_type', type=str, default='ae_additive',
                   help= 'ae, ae_poly')
parser.add_argument('--latent_case', type=str, default='balls_supp_l_shape',
                    help='laplace; uniform')
parser.add_argument('--eval_latent_case', type=str, default='balls_supp_l_shape',
                    help='laplace; uniform')
parser.add_argument('--train_size', type=int, default= 50000,
                    help='')

args = parser.parse_args()
results_case= args.results_case

if results_case == 'violin_plot_2d':
    
    method_grid= ['ae_base', 'ae_additive']
    eval_cases= ['In Support', 'OOD Support']
    latent_grid= [
                    {'train' : 'balls_supp_l_shape' , 'eval':  'balls_supp_l_shape', 'legend':  eval_cases[0]},
                    {'train' : 'balls_supp_l_shape', 'eval': 'balls_supp_extrapolate', 'legend': eval_cases[1] },                    
                ]    
    train_size= 10000
    
    res= {}
    
    for method_type in method_grid:
        res[method_type]= {}
        for latent_case in latent_grid:
            res[method_type][latent_case['legend']] = {
                                                        'recon_err' : [], 
                                                        'mcc': []
                                                    } 
            save_dir= 'results/' +  latent_case['train'] + '_eval_latent_' +  latent_case['eval'] + \
         '/' + method_type + '/' +  str(train_size) + '/'
            data= pickle.load( open(save_dir + 'logs.pickle', 'rb') )

            curr_data= data['recon_rmse']
            res[method_type][latent_case['legend']]['recon_err'].append( curr_data ) 
            print(curr_data)
            
            curr_data= data['mcc_spearman']
            print(curr_data)
            res[method_type][latent_case['legend']]['mcc'].append( curr_data ) 

    df= {
            'Case': [],
            'Method': [],
            'Recon Error': [],
            'MCC': [],
        }
    
    #Table
    for latent_case in latent_grid:
        for method_type in method_grid:

            df['Case'].append( latent_case['legend'] )
            if method_type == 'ae_base':
                df['Method'].append( 'Non-Additive Decoder' )
            else:
                df['Method'].append( 'Additive Decoder' )

            curr_data= res[method_type][latent_case['legend']]['recon_err'] 
            mean= round( np.mean(curr_data), 2)
            sem= round( np.std(curr_data)/np.sqrt(len(curr_data)), 3)
            df['Recon Error'].append( str(mean) + ' ( ' +  str(sem) +  ' ) ' )

            curr_data= res[method_type][latent_case['legend']]['mcc'] 
            mean= round( np.mean(curr_data), 1)
            sem= round( np.std(curr_data)/np.sqrt(len(curr_data)), 2)
            df['MCC'].append( str(mean) + ' ( ' +  str(sem) +  ' ) ' )

    df= pd.DataFrame.from_dict(df)
    print(df.to_latex(index=False))

    #Violin Plot
    labels= [
                'Additive (In Support)',
                'Non-Additive (In Support)',
                'Additive (Out of Support)',
                'Non-Additive (Out of Support)',
            ]

    gen_violin_plot(res, eval_cases, labels, 'violin_plot_2d', 'LMS-Spearman (MCC)')


elif results_case == 'violin_plot_4d':
    
    method_grid= ['ae_base', 'ae_additive']
    eval_cases= ['Independent Latent', 'Dependent Latent']
    latent_grid= [
                    {'train' : 'balls_supp_iid_no_occ' , 'eval':  'balls_supp_iid_no_occ', 'legend':  eval_cases[0]},
                    {'train' : 'balls_supp_scm_linear', 'eval': 'balls_supp_scm_linear', 'legend': eval_cases[1] },                    
                ]    
    train_size= 50000
    
    res= {}
    for method_type in method_grid:
        res[method_type]= {}
        for latent_case in latent_grid:
            res[method_type][latent_case['legend']] = {
                                                        'recon_err' : [], 
                                                        'mcc': []
                                                    } 
            save_dir= 'results/' +  latent_case['train'] + '_eval_latent_' +  latent_case['eval'] + \
         '/' + method_type + '/' +  str(train_size) + '/'
            data= pickle.load( open(save_dir + 'logs.pickle', 'rb') )

            curr_data= data['recon_rmse']
            res[method_type][latent_case['legend']]['recon_err'].append( curr_data ) 
            
            curr_data= data['mcc_block']
            res[method_type][latent_case['legend']]['mcc'].append( curr_data ) 

    df= {
            'Case': [],
            'Method': [],
            'Recon Error': [],
            'MCC': [],
        }
    
    #Table
    for latent_case in latent_grid:
        for method_type in method_grid:

            df['Case'].append( latent_case['legend'] )
            if method_type == 'ae_base':
                df['Method'].append( 'Non-Additive Decoder' )
            else:
                df['Method'].append( 'Additive Decoder' )

            curr_data= res[method_type][latent_case['legend']]['recon_err'] 
            mean= round( np.mean(curr_data), 2)
            sem= round( np.std(curr_data)/np.sqrt(len(curr_data)), 3)
            df['Recon Error'].append( str(mean) + ' ( ' +  str(sem) +  ' ) ' )

            curr_data= res[method_type][latent_case['legend']]['mcc'] 
            mean= round( np.mean(curr_data), 1)
            sem= round( np.std(curr_data)/np.sqrt(len(curr_data)), 2)
            df['MCC'].append( str(mean) + ' ( ' +  str(sem) +  ' ) ' )

    df= pd.DataFrame.from_dict(df)
    print(df.to_latex(index=False))

    #Violin Plot
    labels= [
                'Additive (IID Latent)',
                'Non-Additive (IID Latent)',
                'Additive (Corr Latent)',
                'Non-Additive (Corr Latent)',
            ]

    gen_violin_plot(res, eval_cases, labels, 'violin_plot_4d', 'LMS-Tree')
    
    #Mini Violin Plots
    labels= [
                'Additive',
                'Non-Additive',
            ]
    
    gen_mini_violin_plot(res, [eval_cases[0]], labels, 'violin_plot_4d_indep', 'LMS-Tree')    
    gen_mini_violin_plot(res, [eval_cases[1]], labels, 'violin_plot_4d_dep', 'LMS-Tree')


elif results_case == 'sample_complexity_analysis':

    method_grid= ['ae_base', 'ae_additive']
    latent_case= 'balls_supp_l_shape'
    eval_latent_grid= ['balls_supp_l_shape', 'balls_supp_extrapolate']
    train_size_grid= [1000, 10000, 50000]     

    res= {}
    for eval_latent_case in eval_latent_grid:   
        res[eval_latent_case]= {}         
        for method_type in method_grid:
            res[eval_latent_case][method_type]= { 
                                'recon_err' : { 'mean': [], 'sem': [] }, 
                                'mcc': { 'mean': [], 'sem': [] }
                                }
            for train_size in train_size_grid:
                save_dir= 'results/' +  latent_case + '_eval_latent_' +  eval_latent_case + \
             '/' + method_type + '/' +  str(train_size) + '/'
                data= pickle.load( open(save_dir + 'logs.pickle', 'rb') )
                
                curr_data= data['recon_rmse']
                res[eval_latent_case][method_type]['recon_err']['mean'].append( np.mean(curr_data)  ) 
                res[eval_latent_case][method_type]['recon_err']['sem'].append(  np.std(curr_data)/np.sqrt(len(curr_data))  ) 
                
                curr_data= data['mcc_spearman']
                res[eval_latent_case][method_type]['mcc']['mean'].append( np.mean(curr_data)  ) 
                res[eval_latent_case][method_type]['mcc']['sem'].append(  np.std(curr_data)/np.sqrt(len(curr_data))  ) 
    
    for eval_latent_case in eval_latent_grid:
        gen_line_plot(res, eval_latent_case, train_size_grid)    


    
elif results_case == 'specific_plots':

    method_type= args.method_type
    latent_case= args.latent_case
    eval_latent_case= args.eval_latent_case
    train_size= args.train_size

    save_dir= 'results/' +  latent_case + '_eval_latent_' +  eval_latent_case + \
    '/' + method_type + '/' +  str(train_size) + '/'
    data= pickle.load( open(save_dir + 'logs.pickle', 'rb') )    

    mcc_arr= data['mcc_spearman']
#     mcc_arr= data['mcc_block']
    sorted_indices= np.argsort(mcc_arr)
    
    print('MCC Array: ', mcc_arr)
    print('Min Seed: ', sorted_indices[0])
    print('Median Seed', sorted_indices[ int(len(mcc_arr)/2)  ] )
    print('Max Seed: ', sorted_indices[-1] )
