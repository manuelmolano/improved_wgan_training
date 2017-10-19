# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:31:05 2017

@author: manuel
"""
import time
import numpy as np
import matplotlib.pyplot as plt
#import time
import matplotlib

#parameters for figure
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots


def spike_trains_corr(num_bins=64, num_neurons=32, correlations_mat=np.zeros((16,))+0.5,\
                        group_size=2,refr_per=4,firing_rates_mat=np.zeros((32,))+0.2,activity_peaks=np.zeros((32,1))+32):
    std_resp = 5
    noise = np.mean(firing_rates_mat)/2
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
    prob_firing = np.exp(-(t-activity_peaks)**2/std_resp**2) + noise #+ np.exp(-(t-activity_peaks*2)**2/std_resp**2)/2 + np.exp(-(t-activity_peaks*3.5)**2/std_resp**2)/1.5 
    X = X*prob_firing
    X = X > np.random.random(X.shape)
    assert np.sum(np.isnan(X.flatten()))==0
    return X.astype(float)


def spike_train_packets(num_bins=32, num_neurons=16, group_size=4, prob_packets=0.02, firing_rates_mat=np.zeros((16,1))+0.25, refr_per=2):
    X = (np.zeros((num_neurons,num_bins)) + firing_rates_mat) > np.random.random((num_neurons,num_bins))
    for ind_n in range(num_neurons):
        X[ind_n,:] = refractory_period(refr_per, X[ind_n,:].reshape(1,num_bins), firing_rates_mat[ind_n])
    packets_activity = np.zeros((num_neurons,num_bins))
    packet = np.eye(group_size)
    for ind_p in range(int(num_neurons/group_size)):
        aux = np.zeros((1,num_bins)) + prob_packets > np.random.random((1,num_bins))
        aux[:,-group_size:] = 0
        aux = refractory_period_hard(group_size, aux, prob_packets)
        packets_timing = np.nonzero(aux[0])[0]
        assert all(np.diff(packets_timing)>group_size)
        for ind_t in range(len(packets_timing)):
            packets_activity[ind_p*group_size:(ind_p+1)*group_size,packets_timing[ind_t]:packets_timing[ind_t]+group_size] = packet
                    
    result = X + packets_activity
    result[result>1] = 1
    return result
   
def plot_samples(X, num_neurons, folder, name):
    my_cmap = plt.cm.gray
    num1 = 8
    num2 = 8
    f,sbplt = plt.subplots(8,8,figsize=(num1, num2),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    num_samples = num1*num2
    
    for ind in range(num_samples):
        sample = X[:,ind].reshape((num_neurons,-1))
        sbplt[int(np.floor(ind/num1))][ind%num2].imshow(sample,interpolation='nearest', cmap = my_cmap)  
        sbplt[int(np.floor(ind/num1))][ind%num2].axis('off')  
         
    f.savefig(folder + name + '_samples.svg',dpi=600, bbox_inches='tight')
    plt.close(f) 
    
def get_samples(num_samples=2**13,num_bins=64, num_neurons=32, correlations_mat=np.zeros((16,))+0.5, packets_on=False,\
                group_size=2,refr_per=2,firing_rates_mat=np.zeros((16,))+0.2,activity_peaks=np.zeros((16,))+32, prob_packets=0.05, shuffled_index=np.arange(32), folder=''):                        
    X = np.zeros((num_neurons*num_bins,num_samples))
    for ind in range(num_samples):
        if not packets_on:
            sample = spike_trains_corr(num_neurons=num_neurons,num_bins=num_bins, correlations_mat=correlations_mat,\
                    group_size=group_size, firing_rates_mat=firing_rates_mat, refr_per=refr_per, activity_peaks=activity_peaks)
        else:
            sample = spike_train_packets(num_bins=num_bins, num_neurons=num_neurons, group_size=group_size, prob_packets=prob_packets, firing_rates_mat=firing_rates_mat, refr_per=refr_per,)
        
        sample = sample[shuffled_index,:]
        
        X[:,ind] = sample.reshape((num_neurons*num_bins,-1))[:,0]
    if folder!='':
        plot_samples(X, num_neurons, folder, 'real')
    return X

def get_aproximate_probs(num_samples=2**13,num_bins=64, num_neurons=32, correlations_mat=np.zeros((16,))+0.5,\
                        group_size=2,refr_per=2,firing_rates_mat=np.zeros((16,))+0.2, activity_peaks=np.zeros((16,))+32):
    X = np.zeros((num_neurons*num_bins,num_samples))
    start_time = time.time()
    for ind in range(num_samples):
        if ind%10000==0:
            print(str(ind) + ' time ' + str(time.time() - start_time))
        sample = spike_trains_corr(num_neurons=num_neurons,num_bins=num_bins, correlations_mat=correlations_mat,\
                    group_size=group_size, firing_rates_mat=firing_rates_mat, refr_per=refr_per, activity_peaks=activity_peaks)
        X[:,ind] = sample.reshape((num_neurons*num_bins,-1))[:,0]
    
    
    r_unique = np.unique(X,axis=1,return_counts=True)
   
    
    assert abs(np.sum(r_unique[1])-num_samples)<0.00000001
    
    return r_unique
 
def refractory_period_hard(refr_per, r, firing_rate):
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
    
def refractory_period(refr_per, r, firing_rate):
    sigma = refr_per/1.5#np.sqrt(refr_per)
    margin_length = 2*r.shape[1]
    for ind_tr in range(int(np.shape(r)[0])):
        r_aux = r[ind_tr,:]
        margin1 = np.random.poisson(np.zeros((margin_length,))+firing_rate)
        margin1[margin1>0] = 1
        r_aux = np.hstack((margin1,r_aux))
        spiketimes = np.nonzero(r_aux>0)
        spiketimes = np.sort(spiketimes)
        isis = np.diff(spiketimes)
        prob_of_removing = np.exp(-(isis/sigma**2))
        too_close = np.nonzero(np.random.random(size=prob_of_removing.shape)<prob_of_removing)
        while len(too_close[0])>0:
            spiketimes = np.delete(spiketimes,too_close[0]+1)
            isis = np.diff(spiketimes)
            prob_of_removing = np.exp(-(isis/sigma**2))
            what_to_remove = np.random.random(size=prob_of_removing.shape)<prob_of_removing           
            too_close = np.nonzero(what_to_remove)
         
        r_aux = np.zeros(r_aux.shape)
        r_aux[spiketimes] = 1
            
        r[ind_tr,:] = r_aux[margin_length:]
    return r


    
if __name__ == '__main__':
    plt.close('all')
    sample = spike_train_packets()
    sample[sample==3] = 2
    f = plt.figure()
    plt.imshow(sample,interpolation='nearest')
    plt.colorbar()
    
    
#    asdsadds
#    
#    import analysis
#    num_tr = 1000
#    num_bins = 64
#    num_neurons = 32
#    lag = 10
#    refr_per_mat = [0.1,0.8,1,1.3,1.6]
#    f = plt.figure()
#    for ind_rp in range(len(refr_per_mat)):
#        X = np.zeros((num_neurons*num_bins,num_tr))
#        autocorrelogram_mat = np.zeros(2*lag+1)
#        for ind in range(num_tr):
#            sample = spike_train_packets(refr_per=refr_per_mat[ind_rp])
#            X[:,ind] = sample.reshape((num_neurons*num_bins,-1))[:,0]
#            autocorrelogram_mat += analysis.autocorrelogram(sample,lag=lag)
#        autocorrelogram_mat = autocorrelogram_mat/np.max(autocorrelogram_mat)
#        autocorrelogram_mat[lag] = 0 
#        plt.plot(autocorrelogram_mat)
#


