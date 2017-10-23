#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:44:19 2017

@author: manuel
"""

#get real samples

import numpy as np
from tflib import sim_pop_activity, retinal_data, analysis


def generate_spike_trains(config, recovery_dir):
    if config.dataset=='uniform':
        if recovery_dir!="":
            aux = np.load(recovery_dir+ '/stats_real.npz')
            real_samples = aux['samples']
            firing_rates_mat = aux['firing_rate_mat']
            correlations_mat = aux['correlation_mat']
            activity_peaks = aux['activity_peaks']
            shuffled_index = aux['shuffled_index']
        else:
            #we shuffle neurons to test if the network still learns the packets
            shuffled_index = np.arange(config.num_neurons)
            np.random.shuffle(shuffled_index)
            firing_rates_mat = config.firing_rate+2*(np.random.random(int(config.num_neurons/config.group_size),)-0.5)*config.firing_rate/2    
            correlations_mat = config.correlation+2*(np.random.random(int(config.num_neurons/config.group_size),)-0.5)*config.correlation/2   
            #peaks of activity
            #sequence response
            aux = np.arange(int(config.num_neurons/config.group_size))
            activity_peaks = [[x]*config.group_size for x in aux]#np.random.randint(0,high=config.num_bins,size=(1,config.num_neurons)).reshape(config.num_neurons,1)
            activity_peaks = np.asarray(activity_peaks)
            activity_peaks = activity_peaks.flatten()
            activity_peaks = activity_peaks*config.group_size*config.num_bins/config.num_neurons
            activity_peaks = activity_peaks.reshape(config.num_neurons,1)
            #peak of activity equal for all neurons 
            #activity_peaks = np.zeros((config.num_neurons,1))+config.num_bins/4
            real_samples = sim_pop_activity.get_samples(num_samples=config.num_samples, num_bins=config.num_bins,\
                                num_neurons=config.num_neurons, correlations_mat=correlations_mat, group_size=config.group_size, shuffled_index=shuffled_index,\
                                refr_per=config.ref_period,firing_rates_mat=firing_rates_mat, activity_peaks=activity_peaks, folder=config.sample_dir)
            
        #save original statistics
        analysis.get_stats(X=real_samples, num_neurons=config.num_neurons, num_bins=config.num_bins, folder=config.sample_dir, shuffled_index=shuffled_index,\
                           name='real',firing_rate_mat=firing_rates_mat, correlation_mat=correlations_mat, activity_peaks=activity_peaks)
            
        #get dev samples
        dev_samples = sim_pop_activity.get_samples(num_samples=int(config.num_samples/4), num_bins=config.num_bins,\
                       num_neurons=config.num_neurons, correlations_mat=correlations_mat, group_size=config.group_size, shuffled_index=shuffled_index,\
                       refr_per=config.ref_period,firing_rates_mat=firing_rates_mat, activity_peaks=activity_peaks)
        
    elif config.dataset=='packets':
        if recovery_dir!="":
            aux = np.load(recovery_dir+ '/stats_real.npz')
            real_samples = aux['samples']
            firing_rates_mat = aux['firing_rate_mat']
            shuffled_index = aux['shuffled_index']
        else:
            #we shuffle neurons to test if the network still learns the packets
            shuffled_index = np.arange(config.num_neurons)
            np.random.shuffle(shuffled_index)
            firing_rates_mat = config.firing_rate+2*(np.random.random(size=(config.num_neurons,1))-0.5)*config.firing_rate/2 
            real_samples = sim_pop_activity.get_samples(num_samples=config.num_samples, num_bins=config.num_bins, refr_per=config.ref_period,\
                                 num_neurons=config.num_neurons, group_size=config.group_size, firing_rates_mat=firing_rates_mat, packets_on=True,\
                                 prob_packets=config.packet_prob, shuffled_index=shuffled_index, folder=config.sample_dir)
        #save original statistics
        analysis.get_stats(X=real_samples, num_neurons=config.num_neurons, num_bins=config.num_bins, folder=config.sample_dir, name='real',\
                       firing_rate_mat=firing_rates_mat, shuffled_index=shuffled_index)
        #get dev samples
        dev_samples = sim_pop_activity.get_samples(num_samples=int(config.num_samples/4), num_bins=config.num_bins, refr_per=config.ref_period,\
                       num_neurons=config.num_neurons, group_size=config.group_size, firing_rates_mat=firing_rates_mat, packets_on=True,\
                       prob_packets=config.packet_prob,shuffled_index=shuffled_index)
        
    elif config.dataset=='retina':
        real_samples = retinal_data.get_samples(num_bins=config.num_bins, num_neurons=config.num_neurons, instance=config.data_instance)
        #save original statistics
        analysis.get_stats(X=real_samples, num_neurons=config.num_neurons, num_bins=config.num_bins, folder=config.sample_dir, name='real',instance=config.data_instance)
        dev_samples = []
    return real_samples, dev_samples





