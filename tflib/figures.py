#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:13:56 2017

@author: manuel
"""
#import sys, os
#sys.path.append('/home/manuel/improved_wgan_training/')
#import glob

import numpy as np
from tflib import  retinal_data, analysis
import matplotlib.pyplot as plt
import matplotlib
#parameters for figure
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots

def figure_1():
    #plot real samples
    print('to do')
    
    
def figure_2_4(num_samples, num_neurons, num_bins, folder, folder_fc, fig_2_or_4):
    original_data = np.load(folder + '/stats_real.npz')   
    mean_spike_count_real, autocorrelogram_mat_real, firing_average_time_course_real, cov_mat_real, k_probs_real, lag_cov_mat_real = \
    [original_data["mean"], original_data["acf"], original_data["firing_average_time_course"], original_data["cov_mat"], original_data["k_probs"], original_data["lag_cov_mat"]]
    
    #load conv information
    conv_data = np.load(folder + '/samples_fake.npz')['samples']
    conv_data_bin = (conv_data > np.random.random(conv_data.shape)).astype(float)   
    cov_mat_conv, k_probs_conv, mean_spike_count_conv, autocorrelogram_mat_conv, firing_average_time_course_conv, lag_cov_mat_conv = \
        analysis.get_stats_aux(conv_data_bin, num_neurons, num_bins)
    #load fc information
    if fig_2_or_4==2:
        fc_data = np.load(folder_fc + '/samples_fake.npz')['samples']
        fc_data_bin = (fc_data > np.random.random(fc_data.shape)).astype(float)   
        cov_mat_comp, k_probs_comp, mean_spike_count_comp, autocorrelogram_mat_comp, firing_average_time_course_comp, lag_cov_mat_comp = \
            analysis.get_stats_aux(fc_data_bin, num_neurons, num_bins)
    elif fig_2_or_4==4:
        k_pairwise_samples = retinal_data.load_samples_from_k_pairwise_model(num_samples=num_samples, num_bins=num_bins, num_neurons=num_neurons, instance='1')    
        cov_mat_comp, k_probs_comp, mean_spike_count_comp, autocorrelogram_mat_comp, firing_average_time_course_comp, lag_cov_mat_comp = \
            analysis.get_stats_aux(k_pairwise_samples, num_neurons, num_bins)
    
    
    only_cov_mat_conv = cov_mat_conv.copy()
    only_cov_mat_conv[np.diag_indices(num_neurons)] = np.nan
    only_cov_mat_comp = cov_mat_comp.copy()
    only_cov_mat_comp[np.diag_indices(num_neurons)] = np.nan

    #PLOT
    index = np.linspace(-10,10,2*10+1)
    #figure for all training error across epochs (supp. figure 2)
    f = plt.figure(figsize=(10, 10),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    plt.subplot(2,3,5)
    #plot autocorrelogram(s)
    plt.plot(index, autocorrelogram_mat_conv,'r')
    plt.plot(index, autocorrelogram_mat_comp,'g')
    plt.plot(index, autocorrelogram_mat_real,'b')
    plt.title('Autocorrelogram')
    plt.xlabel('time (ms)')
    plt.ylabel('number of spikes')
    
    #plot mean firing rates
    mean_spike_count_real = mean_spike_count_real*1000/num_bins
    mean_spike_count_conv = mean_spike_count_conv*1000/num_bins
    mean_spike_count_comp = mean_spike_count_comp*1000/num_bins
    plt.subplot(2,3,1)
    plt.plot([np.min(mean_spike_count_real),np.max(mean_spike_count_real)],[np.min(mean_spike_count_real),np.max(mean_spike_count_real)],'k')
    plt.plot(mean_spike_count_real,mean_spike_count_conv,'.r')
    plt.plot(mean_spike_count_real,mean_spike_count_comp,'.g')
    plt.xlabel('mean firing rate expt (Hz)')
    plt.ylabel('mean firing rate models (Hz)')   
    plt.title('mean firing rates')

    #plot covariances
    plt.subplot(2,3,2)
    only_cov_mat_real = cov_mat_real.copy()
    only_cov_mat_real[np.diag_indices(num_neurons)] = np.nan
    plt.plot([np.nanmin(only_cov_mat_real.flatten()),np.nanmax(only_cov_mat_real.flatten())],\
                    [np.nanmin(only_cov_mat_real.flatten()),np.nanmax(only_cov_mat_real.flatten())],'k')
    plt.plot(only_cov_mat_real.flatten(),only_cov_mat_conv.flatten(),'.r')
    plt.plot(only_cov_mat_real.flatten(),only_cov_mat_comp.flatten(),'.g')
    plt.title('pairwise covariances')
    plt.xlabel('covariances expt')
    plt.ylabel('covariances models')
  
        
        
    #plot k-statistics
    plt.subplot(2,3,3)
    plt.plot([0,np.max(k_probs_real)],[0,np.max(k_probs_real)],'k')        
    plt.plot(k_probs_real,k_probs_conv,'.r')        
    plt.plot(k_probs_real,k_probs_comp,'.g')  
    plt.xlabel('k-probs expt')
    plt.ylabel('k-probs models')
    plt.title('k statistics')        
      
    #plot average time course
    #firing_average_time_course[firing_average_time_course>0.048] = 0.048
    plt.subplot(6,3,10)
    firing_average_time_course_real = firing_average_time_course_real*1000/num_bins
    firing_average_time_course_conv = firing_average_time_course_conv*1000/num_bins
    firing_average_time_course_comp = firing_average_time_course_comp*1000/num_bins
    maximo = np.max(firing_average_time_course_real.flatten())
    minimo = np.min(firing_average_time_course_real.flatten())
    plt.imshow(firing_average_time_course_real,interpolation='nearest')
   
    plt.title('Real time course (Hz)')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6,3,13)
    plt.imshow(firing_average_time_course_conv,interpolation='nearest', clim=(minimo,maximo))
    plt.title('Spike-GAN time course (Hz)')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6,3,16)
    plt.imshow(firing_average_time_course_comp,interpolation='nearest', clim=(minimo,maximo))#map_aux = 
    plt.title('FC time course (Hz)')
    plt.xlabel('time (ms)')
    plt.ylabel('neuron')
    #f.colorbar(map_aux,orientation='horizontal')
    plt.xticks([])
    plt.yticks([])
        
    #plot lag covariance
    plt.subplot(2,3,6)
    plt.plot([np.min(lag_cov_mat_real.flatten()),np.max(lag_cov_mat_real.flatten())],\
                        [np.min(lag_cov_mat_real.flatten()),np.max(lag_cov_mat_real.flatten())],'k')        
    plt.plot(lag_cov_mat_real,lag_cov_mat_conv,'.r')
    plt.plot(lag_cov_mat_real,lag_cov_mat_comp,'.g')
    plt.xlabel('lag cov real')
    plt.ylabel('lag cov models')
    plt.title('lag covarainces')
    f.savefig(folder+'figure_2.svg',dpi=600, bbox_inches='tight')
    plt.close(f)
    
    
if __name__ == '__main__':
    #FIGURE 4
    dataset = 'retina'
    num_samples = '8192'
    num_neurons = '50'
    num_bins = '32'
    critic_iters = '5'
    lambd = '10' 
    num_layers = '2'
    num_features = '128'
    kernel = '5'
    iteration = '21'
    sample_dir = '/home/manuel/improved_wgan_training/samples conv/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
          + '_critic_iters_' + critic_iters + '_lambda_' + lambd +\
          '_num_layers_' + num_layers + '_num_features_' + num_features + '_kernel_' + kernel +\
          '_iteration_' + iteration + '/'
    figure_2_4(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir,folder_fc='', fig_2_or_4=4)
    
    #FIGURE 2
    dataset = 'uniform'
    num_samples = '8192'
    num_neurons = '32'
    num_bins = '64'
    ref_period = '2'
    firing_rate = '0.25'
    correlation = '0.3'
    group_size = '2'
    critic_iters = '5'
    lambd = '10' 
    num_layers = '2'
    num_features = '128'
    kernel = '5'
    iteration = '20'
    num_units = '400'
    sample_dir = '/home/manuel/improved_wgan_training/samples conv/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
          + '_ref_period_' + ref_period + '_firing_rate_' + firing_rate + '_correlation_' + correlation +\
          '_group_size_' + group_size + '_critic_iters_' + critic_iters + '_lambda_' + lambd +\
          '_num_layers_' + num_layers + '_num_features_' + num_features + '_kernel_' + kernel +\
          '_iteration_' + iteration + '/'
    sample_dir_fc = '/home/manuel/improved_wgan_training/samples fc/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
          + '_ref_period_' + ref_period + '_firing_rate_' + firing_rate + '_correlation_' + correlation +\
          '_group_size_' + group_size + '_critic_iters_' + critic_iters + '_lambda_' + lambd + '_num_units_' + num_units +\
          '_iteration_' + iteration + '/'
          
    figure_2_4(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir, folder_fc=sample_dir_fc, fig_2_or_4=2)
    
    
    
    
    