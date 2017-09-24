# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:31:05 2017

@author: manuel
"""
import time
import numpy as np
#import matplotlib.pyplot as plt
#import time

def spike_trains_corr(num_bins=64, num_neurons=32, correlations_mat=np.zeros((16,))+0.5,\
                        group_size=2,refr_per=2,firing_rates_mat=np.zeros((32,))+0.2,activity_peaks=np.zeros((16,))+32):
    std_resp = 5
    noise = np.mean(firing_rates_mat)/5
    X = np.zeros((num_neurons,num_bins)) 
    
    for ind in range(int(num_neurons/group_size)):
        q = np.sqrt(correlations_mat[ind])
        spike_trains = (np.zeros((group_size,num_bins)) + firing_rates_mat[ind]) > np.random.random((group_size,num_bins))
        reference = (np.zeros((1,num_bins)) + firing_rates_mat[ind]) > np.random.random((1,num_bins))
        reference = refractory_period(refr_per,reference,firing_rates_mat[ind])
        reference = np.tile(reference,(group_size,1))
        same_state = (np.zeros((group_size,num_bins)) + q) > np.random.random((group_size,num_bins))
        spike_trains[same_state] = reference[same_state]
        spike_trains = refractory_period(refr_per,spike_trains,firing_rates_mat[ind])
        X[ind*group_size:(ind+1)*group_size,:] = spike_trains

    #here we use the activity peaks to modulate the firing of neurons    
    t = np.arange(num_bins).reshape(1,num_bins)
    prob_firing = np.exp(-(t-activity_peaks)**2/std_resp**2) + noise
    X = X*prob_firing
    X = X > np.random.random(X.shape)
    assert np.sum(np.isnan(X.flatten()))==0
    return X.astype(float)

def get_samples(num_samples=2**13,num_bins=64, num_neurons=32, correlations_mat=np.zeros((16,))+0.5,\
                        group_size=2,refr_per=2,firing_rates_mat=np.zeros((16,))+0.2,activity_peaks=np.zeros((16,))+32):                        
    X = np.zeros((num_neurons*num_bins,num_samples))
    
    for ind in range(num_samples):
        sample = spike_trains_corr(num_neurons=num_neurons,num_bins=num_bins, correlations_mat=correlations_mat,\
                    group_size=group_size, firing_rates_mat=firing_rates_mat, refr_per=refr_per, activity_peaks=activity_peaks)
        X[:,ind] = sample.reshape((num_neurons*num_bins,-1))[:,0]
     
    return X

def get_aproximate_probs(num_samples=2**13,num_bins=64, num_neurons=32, correlations_mat=np.zeros((16,))+0.5,\
                        group_size=2,refr_per=2,firing_rates_mat=np.zeros((16,))+0.2):
    X = np.zeros((num_neurons*num_bins,num_samples))
    start_time = time.time()
    for ind in range(num_samples):
        if ind%10000==0:
            print(str(ind) + ' time ' + str(time.time() - start_time))
        sample = spike_trains_corr(num_neurons=num_neurons,num_bins=num_bins, correlations_mat=correlations_mat,\
                    group_size=group_size, firing_rates_mat=firing_rates_mat, refr_per=refr_per)
        X[:,ind] = sample.reshape((num_neurons*num_bins,-1))[:,0]
    
    
    r_unique = np.unique(X,axis=1,return_counts=True)
   
    
    assert abs(np.sum(r_unique[1])-num_samples)<0.00000001
    
    return r_unique
    
    
def refractory_period(refr_per, r, firing_rate):
    #print('imposing refractory period of ' + str(refr_per))    
    margin_length = 2*np.shape(r)[1]
    for ind_tr in range(int(np.shape(r)[0])):
        r_aux = r[ind_tr,:]
        margin1 = np.random.poisson(np.zeros((margin_length,))+firing_rate)
        margin1[margin1>0] = 1
        r_aux = np.hstack((margin1,r_aux))
        spiketimes = np.nonzero(r_aux>0)
        spiketimes = np.sort(spiketimes)
        isis = np.diff(spiketimes)
        too_close = np.nonzero(isis<=refr_per)
        while len(too_close[0])>0:
            spiketimes = np.delete(spiketimes,too_close[0][0]+1)
            isis = np.diff(spiketimes)
            too_close = np.nonzero(isis<=refr_per)
        
        r_aux = np.zeros(r_aux.shape)
        r_aux[spiketimes] = 1
            
        r[ind_tr,:] = r_aux[margin_length:]
    return r


    
if __name__ == '__main__':
    num_tr = 1000
    num_bins = 64
    num_neurons = 4
    firing_rate = 0.5
    group_size = 2
    ref_period = 0
    a,b,c = get_samples()
    firing_rates_mat = firing_rate+2*(np.random.random(int(num_neurons/group_size),)-0.5)*firing_rate/2
    print(firing_rates_mat)
    spike_count = np.zeros((num_neurons,))
    corr_mat = np.zeros((num_neurons,num_neurons))
    for ind in range(num_tr):
        sample = spike_trains_corr(num_neurons=num_neurons,num_bins=num_bins,\
                    group_size=group_size,firing_rates_mat=firing_rates_mat,refr_per=ref_period)
        spike_count += np.sum(sample,axis=1)
        corr_mat += np.corrcoef(sample)
    spike_count = spike_count/(num_bins*num_tr)
    corr_mat = corr_mat/num_tr
    print(spike_count)
    print(corr_mat)
