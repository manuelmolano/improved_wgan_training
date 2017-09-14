# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:31:05 2017

@author: manuel
"""

import numpy as np
#import matplotlib.pyplot as plt
#import time

def make_generator(n_samples, batch_size, dim, num_neurons, corr, group_size, refr_per, firing_rates_mat):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, num_neurons*dim), dtype='int32')
        samples = list(range(n_samples))
        epoch_count[0] += 1
        for n, i in enumerate(samples):
            image = spike_trains_corr(num_bins=dim, num_neurons=num_neurons,\
            correlation=corr,group_size=group_size,refr_per=refr_per,firing_rates_mat=firing_rates_mat)
            images[n % batch_size,:] = image.flatten()
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

def load(num_samples=2**13, batch_size=64, dim=32,num_neurons=32,corr=0.5,group_size=2,refr_per=2,firing_rates_mat=np.zeros((32,))+0.2):
    return (
        make_generator(num_samples, batch_size,dim,num_neurons,corr,group_size,refr_per,firing_rates_mat),
        make_generator(int(num_samples/4), batch_size,dim,num_neurons,corr,group_size,refr_per,firing_rates_mat)
    )


def spike_trains_corr(num_bins=64, num_neurons=32, correlation=0.5,\
                        group_size=2,refr_per=2,firing_rates_mat=np.zeros((32,))+0.2):
    X = np.zeros((num_neurons,num_bins)) 
    q = np.sqrt(correlation)
    for ind in range(int(num_neurons/group_size)):
        spike_trains = (np.zeros((group_size,num_bins)) + firing_rates_mat[ind]) > np.random.random((group_size,num_bins))
        reference = (np.zeros((1,num_bins)) + firing_rates_mat[ind]) > np.random.random((1,num_bins))
        reference = refractory_period(refr_per,reference,firing_rates_mat[ind])
        reference = np.tile(reference,(group_size,1))
        same_state = (np.zeros((group_size,num_bins)) + q) > np.random.random((group_size,num_bins))
        spike_trains[same_state] = reference[same_state]
        spike_trains = refractory_period(refr_per,spike_trains,firing_rates_mat[ind])
        X[ind*group_size:(ind+1)*group_size,:] = spike_trains
    
    X = X.astype(float)
    assert np.sum(np.isnan(X.flatten()))==0
    return X

def get_stats(num_samples=2**13,num_bins=64, num_neurons=32, correlation=0.5,\
                        group_size=2,refr_per=2,firing_rates_mat=np.zeros((32,))+0.2):                        
    lag = 10
    spike_count = np.zeros((num_neurons,num_samples))
    corr_mat = np.zeros((num_neurons,num_neurons))
    autocorrelogram_mat = np.zeros(2*lag+1)
    
    for ind in range(num_samples):
        sample = spike_trains_corr(num_neurons=num_neurons,num_bins=num_bins,\
                    group_size=group_size,firing_rates_mat=firing_rates_mat,refr_per=refr_per)
        spike_count[:,ind] = np.sum(sample,axis=1)
        corr_aux =  np.corrcoef(sample)
        corr_aux[np.isnan(corr_aux)] = 0
        corr_mat += corr_aux
        autocorrelogram_mat += autocorrelogram(sample,lag=lag)
    
    mean_spike_count = np.mean(spike_count,axis=1)
    std_spike_count = np.std(spike_count,axis=1)
    autocorrelogram_mat = autocorrelogram_mat/np.max(autocorrelogram_mat)
    autocorrelogram_mat[lag] = 0
    corr_mat = corr_mat/num_samples
    return mean_spike_count, std_spike_count, corr_mat, autocorrelogram_mat

  
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

def autocorrelogram(r,lag):
    #get autocorrelogram
    margin = np.zeros((r.shape[0],lag))
    #concatenate margins to then flatten the trials matrix
    r = np.hstack((margin,np.hstack((r,margin))))
    r_flat = r.flatten()
    spiketimes = np.nonzero(r_flat>0)
    ac = np.zeros(2*lag+1)
    for ind_spk in range(len(spiketimes[0])):
        spike = spiketimes[0][ind_spk]
        ac = ac + r_flat[spike-lag:spike+lag+1]
        
    return ac    
    
if __name__ == '__main__':
    num_tr = 1000
    num_bins = 64
    num_neurons = 4
    firing_rate = 0.5
    group_size = 2
    ref_period = 0
    a,b,c = get_stats()
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
