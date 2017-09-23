# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:41:38 2017

@author: manuel
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tflib import sim_pop_activity

#parameters for figure
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots


def get_stats(X, num_neurons, folder, name, firing_rate_mat=None,correlation_mat=None): 
    '''
    compute spike trains spikes: spk-count mean and std, autocorrelogram and correlation mat
    if name!='real' then it compares the above stats with the original ones 
    
    '''
    if name!='real':
        original_data = np.load(folder + '/stats_real.npz')    
    lag = 10
    num_samples = X.shape[1]
    spike_count = np.zeros((num_neurons,num_samples))
    corr_mat = np.zeros((1,num_neurons**2))
    #this is to count the number of times we are adding a value to each corr_mat value (sometimes we get a NaN value and we don't add anything)
    counting_mat = np.zeros((1,num_neurons**2))
    autocorrelogram_mat = np.zeros(2*lag+1)
    
    for ind in range(num_samples):
        sample = X[:,ind].reshape((num_neurons,-1))
        spike_count[:,ind] = np.sum(sample,axis=1)
        corr_aux =  np.corrcoef(sample).flatten().reshape((1,num_neurons**2))
        corr_mat = np.nansum(np.concatenate((corr_mat,corr_aux),axis=0),axis=0).reshape(1,num_neurons**2)
        counting_mat = np.nansum(np.concatenate((counting_mat,~np.isnan(corr_aux)),axis=0),axis=0).reshape(1,num_neurons**2)
        autocorrelogram_mat += autocorrelogram(sample,lag=lag)

    corr_mat = corr_mat/counting_mat    
    corr_mat = corr_mat.reshape(num_neurons,num_neurons)
    mean_spike_count = np.mean(spike_count,axis=1)
    std_spike_count = np.std(spike_count,axis=1)
    autocorrelogram_mat = autocorrelogram_mat/np.max(autocorrelogram_mat)
    autocorrelogram_mat[lag] = 0
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
        data = {'mean':mean_spike_count,'std':std_spike_count,'acf':autocorrelogram_mat,'corr_mat':corr_mat,'samples':X,'firing_rate_mat':firing_rate_mat,'correlation_mat':correlation_mat}
        np.savez(folder + '/stats_'+name+'.npz', **data)    
    else:
        data = {'mean':mean_spike_count,'std':std_spike_count,'acf':autocorrelogram_mat}
        np.savez(folder + '/stats_'+name+'.npz', **data)            
        return acf_error, mean_error, std_error, corr_error
    
    
def evaluate_approx_distribution(X, folder, num_samples_theoretical_distr=2**15,num_bins=10, num_neurons=4, correlations_mat=np.zeros((4,))+0.3,\
                        group_size=2,refr_per=2,firing_rates_mat=np.zeros((16,))+0.25): 
    '''
    compute spike trains spikes: spk-count mean and std, autocorrelogram and correlation mat
    if name!='real' then it compares the above stats with the original ones 
    
    '''
    #get freqs of real samples
    original_data = np.load(folder + '/stats_real.npz')        
    real_samples = original_data['samples']
    aux = np.unique(real_samples,axis=1,return_counts=True)
    real_samples_probs = aux[1]/np.sum(aux[1])
    real_samples_unique = aux[0]
    
    #get freqs of simulated samples
    aux = np.unique(X,axis=1,return_counts=True)
    sim_samples_probs = aux[1]/np.sum(aux[1])
    sim_samples_unique = aux[0]
    
    #get numerical probabilities
    if os.path.exists(folder + '/numerical_probs.npz'):
        num_probs = np.load(folder + '/numerical_probs.npz')        
        num_probs = num_probs['num_probs']
    else:
        num_probs = sim_pop_activity.get_aproximate_probs(num_samples=num_samples_theoretical_distr,num_bins=num_bins, num_neurons=num_neurons, correlations_mat=original_data['correlation_mat'],\
                        group_size=group_size,refr_per=refr_per,firing_rates_mat=original_data['firing_rate_mat'])
        numerical_probs = {'num_probs':num_probs}
        np.savez(folder + '/numerical_probs.npz',**numerical_probs)
    
    samples_theoretical_probs = num_probs[0]
    #probabilites obtain from a large dataset    
    theoretical_probs = num_probs[1]/np.sum(num_probs[1])
    #generated samples that are not in the original training dataset
    #if zero, the generated samples was not present in the original dataset; 
    #if different from zero it stores the frequency with which the sample occurs in the original dataset
    prob_in_training_dataset = np.zeros((sim_samples_unique.shape[1],)) 
    #generated samples that are not in the large dataset and thus have theoretical prob = 0
    #if zero, the generated samples was not present in the large dataset; 
    #if different from zero it stores the frequency with which the sample occurs 
    numerical_prob = np.zeros((sim_samples_unique.shape[1],))
    for ind_s in range(sim_samples_unique.shape[1]):
        sample = sim_samples_unique[:,ind_s].reshape((sim_samples_unique.shape[0],1))
        #get numerical prob
        compare_mat = np.sum(np.abs(samples_theoretical_probs-sample),axis=0)
        if np.count_nonzero(compare_mat==0)==1:
            numerical_prob[ind_s] = theoretical_probs[np.nonzero(compare_mat==0)]
        else: 
            assert np.count_nonzero(compare_mat==0)==0 
        #check whether the sample was in the original dataset
        compare_mat = np.sum(np.abs(real_samples_unique-sample),axis=0)
        if np.count_nonzero(compare_mat==0)==1:
            prob_in_training_dataset[ind_s] = real_samples_probs[np.nonzero(compare_mat==0)]
        else: 
            assert np.count_nonzero(compare_mat==0)==0 
    
    f,sbplt = plt.subplots(2,2,figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)  
    print((numerical_prob!=0) & (prob_in_training_dataset!=0))
    sbplt[0][0].loglog(sim_samples_probs[(numerical_prob!=0) & (prob_in_training_dataset!=0)],numerical_prob[(numerical_prob!=0) & (prob_in_training_dataset!=0)],'xr',basex=10)
    sbplt[0][0].loglog(prob_in_training_dataset[(numerical_prob!=0) & (prob_in_training_dataset!=0)],numerical_prob[(numerical_prob!=0) & (prob_in_training_dataset!=0)],'+b',basex=10)
    equal_line =   np.linspace(0.0005,0.05,10000)
    sbplt[0][0].loglog(equal_line,equal_line,basex=10)
    sbplt[0][0].set_xlabel('frequencies of samples in real dataset')
    sbplt[0][0].set_ylabel('theoretical probabilities')
    sbplt[0][0].set_title(str(np.sum(sim_samples_probs[prob_in_training_dataset!=0])))    
    
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
    
    
    