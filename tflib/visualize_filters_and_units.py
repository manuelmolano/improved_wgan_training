#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:20:24 2017

@author: manuel
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import time

left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots

def plot_filters(filters, sess, config):
    for ind_layer in range(1):#range(len(filters)):
        filter_temp = filters[ind_layer].eval(session=sess)[0,:,:,:]
        my_cmap = plt.cm.gray
        num_filters = filter_temp.shape[2]
        num_rows = int(np.ceil(np.sqrt(num_filters)))
        num_cols = int(np.ceil(np.sqrt(num_filters)))
        #print(np.corrcoef(all_filters.T))
        
        f,sbplt = plt.subplots(num_rows,num_cols,figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)   
        for ind_f in range(num_filters):
          filter_aux = filter_temp[:,:,ind_f]
          filter_aux = filter_aux[:,:].T
          filter_aux = filter_aux/np.max(np.abs(filter_aux))
          sbplt[int(np.floor(ind_f/num_rows))][ind_f%num_cols].imshow(filter_aux, interpolation='nearest', cmap = my_cmap)
          sbplt[int(np.floor(ind_f/num_rows))][ind_f%num_cols].axis('off')
          
          
        f.savefig(config.sample_dir+'filters_layer_' + str(ind_layer) + '.svg',dpi=600, bbox_inches='tight')
        plt.close(f)  
        
        f,sbplt = plt.subplots(num_rows,num_cols,figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)   
        #all_filters = np.empty(shape=(num_filters,FLAGS.num_neurons))
        for ind_f in range(num_filters):
          filter_aux = filter_temp[:,:,ind_f]
          filter_aux = np.mean(filter_aux[:,:],axis=0)
          #all_filters[ind_f,:] = filter_aux
          #filter_aux = filter_aux/np.max(np.abs(filter_aux))
          sbplt[int(np.floor(ind_f/num_rows))][ind_f%num_cols].plot(filter_aux)
    #      sbplt[int(np.floor(ind_f/num_rows))][ind_f%num_cols].axis('off')
          
        f.savefig(config.sample_dir+'filters_neurons_dim_layer_' + str(ind_layer) + '.svg',dpi=600, bbox_inches='tight')
        plt.close(f)    
        
    

def plot_untis_rf(activations, outputs, sess, config, portion=0.05):
    num_layers = len(activations)
    critics_decision = outputs.eval(session=sess)
    critics_decision = critics_decision.reshape(1,len(critics_decision))
    num_rows = int(np.ceil(np.sqrt(num_layers)))
    num_cols = int(np.ceil(np.sqrt(num_layers)))
    f,sbplt = plt.subplots(num_rows,num_cols,figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)  
   
    for ind_f in range(num_layers):
        act_temp = activations[ind_f][:,:,0,:].eval(session=sess)  
        act_shape = act_temp.shape
        num_features = act_shape[1]
        num_bins = act_shape[2]
        num_units = num_features*num_bins  
        corr_with_decision = np.zeros((num_units,))
        counter = 0
        for ind_feature in range(num_features):
            for ind_bin in range(num_bins):
                act_aux = act_temp[:,ind_feature,ind_bin].reshape(1,act_temp.shape[0])
                aux = np.corrcoef(np.concatenate((act_aux,critics_decision),axis=0))
                corr_with_decision[counter] = aux[1,0]
                counter += 1
            
        
        sbplt[int(np.floor(ind_f/num_rows))][ind_f%num_cols].hist(corr_with_decision)  
        sbplt[int(np.floor(ind_f/num_rows))][ind_f%num_cols].set_title(str(np.mean(np.abs(corr_with_decision))))
    f.savefig(config.sample_dir+'correlations.svg',dpi=600, bbox_inches='tight')
    plt.close(f)  
    