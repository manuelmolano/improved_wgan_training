# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:31:05 2017

@author: manuel
"""

import numpy as np
import matplotlib.pyplot as plt
#import time

def make_generator(n_samples, batch_size,dim,num_neurons,corr,group_size,refr_per):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, num_neurons*dim), dtype='int32')
        samples = list(range(n_samples))
        epoch_count[0] += 1
        for n, i in enumerate(samples):
            image = spike_trains_corr(firing_rate=0.5, num_bins=dim, num_neurons=num_neurons,\
            correlation=corr,group_size=group_size,refr_per=refr_per)
            images[n % batch_size,:] = image.flatten()
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

def load(num_samples=2**13, batch_size=64, dim=32,num_neurons=32,corr=0.5,group_size=2,refr_per=2):
    return (
        make_generator(num_samples, batch_size,dim,num_neurons,corr,group_size,refr_per),
        make_generator(int(num_samples/4), batch_size,dim,num_neurons,corr,group_size,refr_per)
    )


def spike_trains_corr(firing_rate=0.2, num_bins=64, num_neurons=32, correlation=0.5,group_size=2,refr_per=2):
    X = (np.zeros((num_neurons,num_bins)) + firing_rate) > np.random.random((num_neurons,num_bins))
    X = refractory_period(refr_per,X,firing_rate)
    q = np.sqrt(correlation)
    for ind in range(int(num_neurons/group_size)):
        reference = (np.zeros((1,num_bins)) + firing_rate) > np.random.random((1,num_bins))
        reference = refractory_period(refr_per,reference,firing_rate)
        reference = np.tile(reference,(group_size,1))
        same_state = (np.zeros((group_size,num_bins)) + q) > np.random.random((group_size,num_bins))
        aux = X[ind*group_size:(ind+1)*group_size,:]
        aux[same_state] = reference[same_state]
        X[ind*group_size:(ind+1)*group_size,:] = aux
    
    X = X.astype(float)
    return X

def spike_trains_gaussian_seq(firing_rate=0.1, num_bins=32, num_neurons=8):
    noise = 0.01*firing_rate
    margin = 14 #num bins from the middle one that the response peaks will span (see line 389)
    std_resp = 4 #std of the gaussian defining the firing rates
        
    t = np.arange(num_bins)
    
    peaks1 = np.linspace(int(num_bins/2)-margin,int(num_bins/2)+margin,num_neurons)
   
    sample =np.zeros((num_neurons,num_bins))
    for ind in range(num_neurons):
        fr = firing_rate*np.exp(-(t-peaks1[ind])**2/std_resp**2) + np.random.normal(0,noise,(1,num_bins))
        fr[fr<0] = 0
        r = fr > np.random.random(fr.shape)
        r = r.astype(float)
        r[r>0] = 1
            
        sample[ind,:] = r
        
    return sample
    
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

def autocorrelogram(r,lag,folder=None):
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
        
    plt.figure()
    index = np.linspace(-lag,lag,2*lag+1)
    plt.plot(index, ac)
    #    plt.title('mean spk-count = ' + str(round(mean_spk_count,3)) + ' (' + str(round(std_spk_count,3)) + ')')
    #    f.savefig(folder + '/autocorrelogram' + name + '.svg', bbox_inches='tight')
    #plt.show()
    #plt.close(f)

if __name__ == '__main__':
    a = spike_trains_corr(num_neurons=32,num_bins=10000,group_size=4)
    autocorrelogram(a,lag=32)
    print(np.corrcoef(a))
#    print('__name__==__main__')
#    train_gen, valid_gen = load(64)
#    t0 = time.time()
#    for i, batch in enumerate(train_gen(), start=1):
#        print( "{}\t{}".format(str(time.time() - t0), np.shape(batch)))
#        if i == 1000:
#            break
#        t0 = time.time()