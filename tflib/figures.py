#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:13:56 2017

@author: manuel
"""
import sys, os
sys.path.append('/home/manuel/improved_wgan_training/')
import glob

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
    
    
def figure_2(num_neurons, num_bins, folder):
    original_data = np.load(folder + '/stats_real.npz')   
    mean_spike_count_real, autocorrelogram_mat_real, firing_average_time_course_real, cov_mat_real, k_probs_real, lag_cov_mat_real = \
    [original_data["mean"], original_data["acf"], original_data["firing_average_time_course"], original_data["cov_mat"], original_data["k_probs"], original_data["lag_cov_mat"]]
    
    #load conv information
    if os.path.exists(folder + '/stats_fake_II.npz'):
        conv_data = np.load(folder + '/stats_fake_II.npz') 
    else:
        files = glob.glob(folder+'/stats_fake*.npz')
        latest_file = 'stats_fake'+str(analysis.find_latest_file(files,'stats_fake'))+'.npz'
        conv_data = np.load(folder +'/'+latest_file)  
        print(folder +'/'+latest_file)
    mean_spike_count_conv, autocorrelogram_mat_conv, firing_average_time_course_conv, cov_mat_conv, k_probs_conv, lag_cov_mat_conv = \
    [conv_data["mean"], conv_data["acf"], conv_data["firing_average_time_course"], conv_data["cov_mat"], conv_data["k_probs"], conv_data["lag_cov_mat"]]
    #load fc information
    folder_fc = folder.copy()
    folder_fc = folder_fc[0:folder_fc.find('conv')]+'fc'+folder_fc[folder_fc.find('conv')+4:]
    if os.path.exists(folder_fc + '/stats_fake_II.npz'):
        fc_data = np.load(folder_fc + '/stats_fake_II.npz') 
    else:
        files = glob.glob(folder_fc+'/stats_fake*.npz')
        latest_file = 'errors_fake'+str(analysis.find_latest_file(files,'stats_fake'))+'.npz'
        fc_data = np.load(folder_fc +'/'+latest_file)    
    mean_spike_count_fc, autocorrelogram_mat_fc, firing_average_time_course_fc, cov_mat_fc, k_probs_fc, lag_cov_mat_fc = \
    [fc_data["mean"], fc_data["acf"], fc_data["firing_average_time_course"], fc_data["cov_mat"], fc_data["k_probs"], fc_data["lag_cov_mat"]]
    
    only_cov_mat_conv = cov_mat_conv.copy()
    only_cov_mat_conv[np.diag_indices(num_neurons)] = np.nan
    only_cov_mat_fc = cov_mat_fc.copy()
    only_cov_mat_fc[np.diag_indices(num_neurons)] = np.nan

    #PLOT
    index = np.linspace(-10,10,2*10+1)
    #figure for all training error across epochs (supp. figure 2)
    f = plt.figure(figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    plt.subplot(2,3,5)
    #plot autocorrelogram(s)
    plt.plot(index, autocorrelogram_mat_conv,'r')
    plt.plot(index, autocorrelogram_mat_fc,'g')
    plt.plot(index, autocorrelogram_mat_real,'b')
    plt.title('Autocorrelogram')
    plt.xlabel('time (ms)')
    plt.ylabel('number of spikes')
    
    #plot mean firing rates
    plt.subplot(2,3,1)
    plt.plot([np.min(mean_spike_count_real),np.max(mean_spike_count_real)],[np.min(mean_spike_count_real),np.max(mean_spike_count_real)],'k')
    plt.plot(mean_spike_count_real,mean_spike_count_conv,'.r')
    plt.plot(mean_spike_count_real,mean_spike_count_fc,'.g')
    plt.xlabel('mean firing rate expt')
    plt.ylabel('mean firing rate model')   
    plt.title('mean firing rates')

    #plot covariances
    plt.subplot(2,3,2)
    only_cov_mat_real = cov_mat_real.copy()
    only_cov_mat_real[np.diag_indices(num_neurons)] = np.nan
    plt.plot([np.nanmin(only_cov_mat_real.flatten()),np.nanmax(only_cov_mat_real.flatten())],\
                    [np.nanmin(only_cov_mat_real.flatten()),np.nanmax(only_cov_mat_real.flatten())],'k')
    plt.plot(only_cov_mat_real.flatten(),only_cov_mat_conv.flatten(),'.r')
    plt.plot(only_cov_mat_real.flatten(),only_cov_mat_fc.flatten(),'.g')
    plt.title('pairwise covariances')
    plt.xlabel('covariances expt')
    plt.ylabel('covariances model')
  
        
        
    #plot k-statistics
    plt.subplot(2,3,3)
    plt.plot([0,np.max(k_probs_real)],[0,np.max(k_probs_real)],'k')        
    plt.plot(k_probs_real,k_probs_conv,'.r')        
    plt.plot(k_probs_real,k_probs_fc,'.g')  
    plt.xlabel('k-probs expt')
    plt.ylabel('k-probs model')
    plt.title('k statistics')        
      
    #plot average time course
    #firing_average_time_course[firing_average_time_course>0.048] = 0.048
    plt.subplot(6,3,10)
    maximo = np.max(firing_average_time_course_real.flatten())
    minimo = np.max(firing_average_time_course_real.flatten())
    map_aux = plt.imshow(firing_average_time_course_real,interpolation='nearest')
    f.colorbar(map_aux)
    plt.title('Real firing time course')
    plt.xlabel('time (ms)')
    plt.ylabel('neuron')
    plt.subplot(6,3,13)
    map_aux = plt.imshow(firing_average_time_course_conv,interpolation='nearest', clim=(minimo,maximo))
    plt.title('Conv firing time course')
    plt.xlabel('time (ms)')
    plt.ylabel('neuron')
    plt.subplot(6,3,16)
    map_aux = plt.imshow(firing_average_time_course_fc,interpolation='nearest', clim=(minimo,maximo))
    plt.title('FC firing time course')
    plt.xlabel('time (ms)')
    plt.ylabel('neuron')
    
    #plot lag covariance
    plt.subplot(2,3,6)
    plt.plot([np.min(lag_cov_mat_real.flatten()),np.max(lag_cov_mat_real.flatten())],\
                        [np.min(lag_cov_mat_real.flatten()),np.max(lag_cov_mat_real.flatten())],'k')        
    plt.plot(lag_cov_mat_real,lag_cov_mat_conv,'.r')
    plt.plot(lag_cov_mat_real,lag_cov_mat_fc,'.g')
    plt.set_xlabel('lag cov real')
    plt.set_ylabel('lag cov models')
    
    f.savefig(folder+'figure_2.svg',dpi=600, bbox_inches='tight')
    plt.close(f)
    #stat-stats
    
    #dynamic stats
    
def figure_4(samples, num_neurons, num_bins, folder):
    original_data = np.load(folder + '/stats_real.npz')   
    if any(k not in original_data for k in ("mean","acf","cov_mat","k_probs","lag_cov_mat","firing_average_time_course","samples")):
        if 'samples' not in original_data:
            samples = retinal_data.get_samples(num_bins=num_bins, num_neurons=num_neurons, instance=instance)
        else:
            samples = original_data['samples']
        cov_mat_real, k_probs_real, mean_spike_count_real, autocorrelogram_mat_real, firing_average_time_course_real, lag_cov_mat_real =\
        analysis.get_stats_aux(samples, num_neurons, num_bins)

    else:
        mean_spike_count_real, autocorrelogram_mat_real, firing_average_time_course_real, cov_mat_real, k_probs_real, lag_cov_mat_real = \
        [original_data["mean"], original_data["acf"], original_data["firing_average_time_course"], original_data["cov_mat"], original_data["k_probs"], original_data["lag_cov_mat"]]
        
    k_pairwise_samples = retinal_data.load_samples_from_k_pairwise_model(num_samples=num_samples, num_bins=num_bins, num_neurons=num_neurons, instance=data_instance)    
    
    
if __name__ == '__main__':
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
    sample_dir = '/home/manuel/improved_wgan_training/samples conv/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
          + '_ref_period_' + ref_period + '_firing_rate_' + firing_rate + '_correlation_' + correlation +\
          '_group_size_' + group_size + '_critic_iters_' + critic_iters + '_lambda_' + lambd +\
          '_num_layers_' + num_layers + '_num_features_' + num_features + '_kernel_' + kernel +\
          '_iteration_' + iteration + '/'
    figure_2(num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir)
    
    
    
    