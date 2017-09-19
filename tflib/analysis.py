# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:41:38 2017

@author: manuel
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#parameters for figure
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots


def get_stats(X, num_neurons, folder, name): 
    '''
    compute spike trains spikes: spk-count mean and std, autocorrelogram and correlation mat
    if name!='real' then it compares the above stats with the original ones 
    
    '''
    if name!='real':
        original_data = np.load(folder + '/stats_real.npz')    
    lag = 10
    num_samples = X.shape[1]
    spike_count = np.zeros((num_neurons,num_samples))
    corr_mat = np.zeros((num_neurons,num_neurons))
    autocorrelogram_mat = np.zeros(2*lag+1)
    
    for ind in range(num_samples):
        sample = X[:,ind].reshape((num_neurons,-1))
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
    index = np.linspace(-10,10,2*10+1)
    #figure for all training error across epochs (supp. figure 2)
    f,sbplt = plt.subplots(2,2,figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    sbplt[0][0].plot(index, autocorrelogram_mat)
    if name!='real':
        sbplt[0][0].plot(index, original_data['acf'])
        acf_error = np.sum(np.abs(autocorrelogram_mat-original_data['acf']))
    sbplt[0][0].set_title('Autocorrelogram')
    sbplt[0][0].set_xlabel('time (ms)')
    sbplt[0][0].set_ylabel('number of spikes')
    sbplt[0][1].plot(mean_spike_count)
    if name!='real':
        sbplt[0][1].plot(original_data['mean'])
        mean_error = np.sum(np.abs(mean_spike_count-original_data['mean']))
        std_error = np.sum(np.abs(std_spike_count-original_data['std']))
    sbplt[0][1].set_title('spk-counts')
    sbplt[0][1].set_xlabel('neuron')
    sbplt[0][1].set_ylabel('firing probability')
    map_aux = sbplt[1][0].imshow(corr_mat,interpolation='nearest')
    f.colorbar(map_aux,ax=sbplt[1][0])
    sbplt[1][0].set_title('sim. correlation mat')
    sbplt[1][0].set_xlabel('neuron')
    sbplt[1][0].set_ylabel('neuron')
    if name!='real':
        map_aux = sbplt[1][1].imshow(original_data['corr_mat'],interpolation='nearest')
        f.colorbar(map_aux,ax=sbplt[1][1])
        sbplt[1][1].set_title('real correlation mat')
        sbplt[1][1].set_xlabel('neuron')
        sbplt[1][1].set_ylabel('neuron')
        corr_error = np.sum(np.abs(corr_mat-original_data['corr_mat']).flatten())
    else:       
        sample = X[:,ind].reshape((num_neurons,-1))
        sbplt[1][1].imshow(sample,interpolation='nearest')
        sbplt[1][1].set_title('sample')
        sbplt[1][1].set_xlabel('time (ms)')
        sbplt[1][1].set_ylabel('neurons')
    f.savefig(folder+'stats_'+name+'.svg',dpi=600, bbox_inches='tight')
    plt.close(f)
    if name=='real':
        data = {'mean':mean_spike_count,'std':std_spike_count,'acf':autocorrelogram_mat,'corr_mat':corr_mat,'samples':X}
        np.savez(folder + '/stats_'+name+'.npz', **data)    
    else:
        data = {'mean':mean_spike_count,'std':std_spike_count,'acf':autocorrelogram_mat}
        np.savez(folder + '/stats_'+name+'.npz', **data)            
        return acf_error, mean_error, std_error, corr_error
    
    
    
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
    
    
    