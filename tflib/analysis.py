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
import time

#parameters for figure
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots


def get_stats(X, num_neurons, num_bins, folder, name, firing_rate_mat=[],correlation_mat=[], activity_peaks=[]): 
    '''
    compute spike trains spikes: spk-count mean and std, autocorrelogram and correlation mat
    if name!='real' then it compares the above stats with the original ones 
    
    '''
    if name!='real':
        original_data = np.load(folder + '/stats_real.npz')    
    lag = 10
    num_samples = X.shape[1]
    spike_count = np.zeros((num_neurons,num_samples))
    X_continuous = np.zeros((num_neurons,num_bins*num_samples))
    autocorrelogram_mat = np.zeros(2*lag+1)
    firing_average_time_course = np.zeros((num_neurons,num_bins))
    
    for ind in range(num_samples):
        sample = X[:,ind].reshape((num_neurons,-1))
        spike_count[:,ind] = np.sum(sample,axis=1)        
        X_continuous[:,ind*num_bins:(ind+1)*num_bins] = sample
        autocorrelogram_mat += autocorrelogram(sample,lag=lag)
        firing_average_time_course += sample

    corr_mat =  np.cov(X_continuous)
    aux = np.histogram(np.sum(X_continuous,axis=0),bins=np.arange(num_neurons+2)-0.5)[0]
    k_probs = aux/X_continuous.shape[1]
    mean_spike_count = np.mean(spike_count,axis=1)
    std_spike_count = np.std(spike_count,axis=1)
    autocorrelogram_mat = autocorrelogram_mat/np.max(autocorrelogram_mat)
    autocorrelogram_mat[lag] = 0
    firing_average_time_course = firing_average_time_course/num_samples
    index = np.linspace(-10,10,2*10+1)
    #figure for all training error across epochs (supp. figure 2)
    f,sbplt = plt.subplots(2,3,figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    
    #plot autocorrelogram(s)
    sbplt[1][1].plot(index, autocorrelogram_mat)
    if name!='real':
        sbplt[1][1].plot(index, original_data['acf'])
        acf_error = np.sum(np.abs(autocorrelogram_mat-original_data['acf']))
    sbplt[1][1].set_title('Autocorrelogram')
    sbplt[1][1].set_xlabel('time (ms)')
    sbplt[1][1].set_ylabel('number of spikes')
    
    #plot mean firing rates
    if name!='real':
        sbplt[0][0].plot(original_data['mean'],mean_spike_count,'.')
        sbplt[0][0].plot([0,np.max(original_data['mean'])],[0,np.max(original_data['mean'])])
        mean_error = np.sum(np.abs(mean_spike_count-original_data['mean']))
        std_error = np.sum(np.abs(std_spike_count-original_data['std']))
        sbplt[0][0].set_xlabel('mean firing rate expt')
        sbplt[0][0].set_ylabel('mean firing rate model')
    else:
        sbplt[0][0].plot(mean_spike_count)
        sbplt[0][0].set_xlabel('neuron')
        sbplt[0][0].set_ylabel('firing probability')
        
    sbplt[0][0].set_title('mean firing rates')

    #plot correlations
    if name!='real':
        sbplt[0][1].plot(original_data['corr_mat'].flatten(),corr_mat.flatten(),'.')
        sbplt[0][1].plot([np.min(original_data['corr_mat'].flatten()),np.max(original_data['corr_mat'].flatten())],\
                        [np.min(original_data['corr_mat'].flatten()),np.max(original_data['corr_mat'].flatten())])
        sbplt[0][1].set_title('pairwise correlations')
        sbplt[0][1].set_xlabel('correlations expt')
        sbplt[0][1].set_ylabel('correlations model')
        corr_error = np.sum(np.abs(corr_mat-original_data['corr_mat']).flatten())
    else:       
        map_aux = sbplt[0][1].imshow(corr_mat,interpolation='nearest')
        f.colorbar(map_aux,ax=sbplt[0][1])
        sbplt[0][1].set_title('correlation mat')
        sbplt[0][1].set_xlabel('neuron')
        sbplt[0][1].set_ylabel('neuron')
        
        
    #plot k-statistics
    if name!='real':
        sbplt[1][0].plot(original_data['k_probs'],k_probs,'.')
        sbplt[1][0].plot([0,np.max(original_data['k_probs'])],[0,np.max(original_data['k_probs'])])
        k_probs_error = np.sum(np.abs(k_probs-original_data['k_probs']))
        sbplt[1][0].set_xlabel('k-probs expt')
        sbplt[1][0].set_ylabel('k-probs model')
    else:
        sbplt[1][0].plot(k_probs)
        sbplt[1][0].set_xlabel('K')
        sbplt[1][0].set_ylabel('probability')
        
    sbplt[1][0].set_title('k statistics')        
        
    
    map_aux = sbplt[0][2].imshow(firing_average_time_course,interpolation='nearest')
    f.colorbar(map_aux,ax=sbplt[0][2])
    sbplt[0][2].set_title('sim firing time course')
    sbplt[0][2].set_xlabel('neuron')
    sbplt[0][2].set_ylabel('time (ms)')
    if name!='real':
        map_aux = sbplt[1][2].imshow(original_data['firing_average_time_course'],interpolation='nearest')
        f.colorbar(map_aux,ax=sbplt[1][2])
        sbplt[1][2].set_title('real firing time course')
        sbplt[1][2].set_xlabel('neuron')
        sbplt[1][2].set_ylabel('time (ms)')
        time_course_error = np.sum(np.abs(firing_average_time_course-original_data['firing_average_time_course']).flatten())    
    
    f.savefig(folder+'stats_'+name+'.svg',dpi=600, bbox_inches='tight')
    plt.close(f)
    if name=='real' and len(firing_rate_mat)>0:
        data = {'mean':mean_spike_count, 'std':std_spike_count, 'acf':autocorrelogram_mat, 'corr_mat':corr_mat, 'samples':X, 'k_probs':k_probs,\
        'firing_rate_mat':firing_rate_mat, 'correlation_mat':correlation_mat, 'activity_peaks':activity_peaks, 'firing_average_time_course':firing_average_time_course}
        np.savez(folder + '/stats_'+name+'.npz', **data)    
    else:
        data = {'mean':mean_spike_count, 'std':std_spike_count, 'acf':autocorrelogram_mat, 'corr_mat':corr_mat, 'k_probs':k_probs, 'firing_average_time_course':firing_average_time_course}
        np.savez(folder + '/stats_'+name+'.npz', **data)    
        if name!='real':        
            return acf_error, mean_error, std_error, corr_error, time_course_error, k_probs_error
    
    
def evaluate_approx_distribution(X, folder, num_samples_theoretical_distr=2**15,num_bins=10, num_neurons=4,\
                        group_size=2,refr_per=2): 
    '''
    compute spike trains spikes: spk-count mean and std, autocorrelogram and correlation mat
    if name!='real' then it compares the above stats with the original ones 
    
    '''
    #get freqs of real samples
    original_data = np.load(folder + '/stats_real.npz')        
    real_samples = original_data['samples']    
    
    if os.path.exists(folder+'/probs_ns_' + str(X.shape[1]) + '_ns_gt_' + str(num_samples_theoretical_distr) + '.npz'):
        probs = np.load(folder+'/probs_ns_' + str(X.shape[1]) + '_ns_gt_' + str(num_samples_theoretical_distr) + '.npz')
        sim_samples_freqs = probs['sim_samples_freqs']        
        numerical_prob = probs['numerical_prob']
        freq_in_training_dataset = probs['freq_in_training_dataset']
        num_impossible_samples = probs['num_impossible_samples']
        surr_samples_freqs = probs['surr_samples_freqs']
        freq_in_training_dataset_surrogates = probs['freq_in_training_dataset_surrogates']
        numerical_prob_surrogates = probs['numerical_prob_surrogates']
        num_impossible_samples_surrogates = probs['num_impossible_samples_surrogates']
        #num_impossible_samples_original = probs['num_impossible_samples_original']
    else:
        #get numerical probabilities
        if os.path.exists(folder + '/numerical_probs_ns_'+str(num_samples_theoretical_distr)+'.npz'):
            num_probs = np.load(folder + '/numerical_probs_ns_'+str(num_samples_theoretical_distr)+'.npz')        
            num_probs = num_probs['num_probs']
        else:
            num_probs = sim_pop_activity.get_aproximate_probs(num_samples=num_samples_theoretical_distr,num_bins=num_bins, num_neurons=num_neurons, correlations_mat=original_data['correlation_mat'],\
                            group_size=group_size,refr_per=refr_per,firing_rates_mat=original_data['firing_rate_mat'], activity_peaks=original_data['activity_peaks'])
            numerical_probs = {'num_probs':num_probs}
            np.savez(folder + '/numerical_probs_ns_'+str(num_samples_theoretical_distr)+'.npz',**numerical_probs)
        
        #samples
        samples_theoretical_probs = num_probs[0]
        #probabilites obtain from a large dataset    
        theoretical_probs = num_probs[1]/np.sum(num_probs[1])
        #get the freq of simulated samples in the original dataset, in the ground truth dataset and in the simulated dataset itself
        freq_in_training_dataset, numerical_prob, sim_samples_freqs = comparison_to_original_and_gt_datasets(samples=X, real_samples=real_samples,\
                ground_truth_samples=samples_theoretical_probs, ground_truth_probs=theoretical_probs)
        #'impossible' samples: samples for which the theoretical prob is 0
        num_impossible_samples = np.count_nonzero(numerical_prob==0)
        #we will now perform the same calculation for several datasets extracted from the ground truth distribution        
        num_surr = 1000  
        freq_in_training_dataset_surrogates = np.zeros((num_surr*X.shape[1],)) 
        numerical_prob_surrogates = np.zeros((num_surr*X.shape[1],))
        surr_samples_freqs = np.zeros((num_surr*X.shape[1],))
        num_impossible_samples_surrogates =  np.zeros((num_surr, ))
        counter = 0
        for ind_surr in range(num_surr+1):
            if ind_surr%10==0:
                print(ind_surr)
            if ind_surr==num_surr:
                #as a control, the last surrogate is the original dataset itself
                surrogate= real_samples
                np.random.shuffle(surrogate.T)
                surrogate = surrogate[:,0:np.min((X.shape[1],surrogate.shape[1]))]
            else:
                surrogate = sim_pop_activity.get_samples(num_samples=X.shape[1], num_bins=num_bins,\
                    num_neurons=num_neurons, correlations_mat=original_data['correlation_mat'], group_size=group_size, refr_per=refr_per,\
                    firing_rates_mat=original_data['firing_rate_mat'], activity_peaks=original_data['activity_peaks'])
                
            freq_in_training_dataset_aux, numerical_prob_aux, samples_freqs_aux = comparison_to_original_and_gt_datasets(samples=surrogate, real_samples=real_samples,\
                ground_truth_samples=samples_theoretical_probs, ground_truth_probs=theoretical_probs)
            
            if ind_surr==num_surr:
                num_impossible_samples_original = np.count_nonzero(numerical_prob_aux==0)
                assert not any(freq_in_training_dataset_aux==0)
            else:
                freq_in_training_dataset_surrogates[counter:counter+len(freq_in_training_dataset_aux)] = freq_in_training_dataset_aux
                numerical_prob_surrogates[counter:counter+len(freq_in_training_dataset_aux)] = numerical_prob_aux
                surr_samples_freqs[counter:counter+len(freq_in_training_dataset_aux)] = samples_freqs_aux
                num_impossible_samples_surrogates[ind_surr] = np.count_nonzero(numerical_prob_aux==0)
                counter += len(freq_in_training_dataset_aux)
        
        freq_in_training_dataset_surrogates = freq_in_training_dataset_surrogates[0:counter]
        numerical_prob_surrogates = numerical_prob_surrogates[0:counter]
        surr_samples_freqs = surr_samples_freqs[0:counter]
        probs = {'sim_samples_freqs':sim_samples_freqs, 'freq_in_training_dataset':freq_in_training_dataset, 'numerical_prob':numerical_prob, 'num_impossible_samples':num_impossible_samples,\
                'surr_samples_freqs':surr_samples_freqs, 'freq_in_training_dataset_surrogates':freq_in_training_dataset_surrogates, 'numerical_prob_surrogates': numerical_prob_surrogates,\
                'num_impossible_samples_surrogates': num_impossible_samples_surrogates, 'num_impossible_samples_original':num_impossible_samples_original}
        
        np.savez(folder+'/probs_ns_' + str(X.shape[1]) + '_ns_gt_' + str(num_samples_theoretical_distr) + '.npz',**probs)
        
        
    
    f,sbplt = plt.subplots(2,2,figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)  
    sbplt[0][0].loglog(sim_samples_freqs[(numerical_prob!=0) & (freq_in_training_dataset!=0)],numerical_prob[(numerical_prob!=0) & (freq_in_training_dataset!=0)],'xr',basex=10)
    sbplt[0][0].loglog(freq_in_training_dataset[(numerical_prob!=0) & (freq_in_training_dataset!=0)],numerical_prob[(numerical_prob!=0) & (freq_in_training_dataset!=0)],'+b',basex=10)
    equal_line = np.linspace(0.00005,0.005,10000)
    sbplt[0][0].loglog(equal_line,equal_line,basex=10)
    sbplt[0][0].set_xlabel('frequencies of samples in real dataset')
    sbplt[0][0].set_ylabel('theoretical probabilities')
    sbplt[0][0].set_title(str(np.sum(sim_samples_freqs[freq_in_training_dataset!=0])))  
    sbplt[0][1].hist(num_impossible_samples_surrogates)
    sbplt[0][1].plot(num_impossible_samples*np.ones((10,1)),np.arange(10),'r')
    #sbplt[0][1].plot(num_impossible_samples_original*np.ones((10,1)),np.arange(10),'g')
    sbplt[0][1].set_xlabel('num of impossible samples')
    sbplt[0][1].set_ylabel('frequency')
    
    
    #get rid of samples already present in the original dataset
    surr_samples_freqs = np.delete(surr_samples_freqs,np.nonzero(freq_in_training_dataset_surrogates==0))
    numerical_prob_surrogates = np.delete(numerical_prob_surrogates,np.nonzero(freq_in_training_dataset_surrogates==0))
    
    #get rid of the freqs for which the numerical prob is zero (so we can compute the logs)
    surr_samples_freqs = np.delete(surr_samples_freqs,np.nonzero(numerical_prob_surrogates==0))
    numerical_prob_surrogates = np.delete(numerical_prob_surrogates,np.nonzero(numerical_prob_surrogates==0))
    
    #compute the logs of the probs and freq
    surr_samples_freqs_log = np.log10(surr_samples_freqs)
    numerical_prob_surrogates_log = np.log10(numerical_prob_surrogates) 
    
    #now we want to get the bins for the hist2d
    aux = np.unique(surr_samples_freqs_log)
    bin_size = 100*np.min(np.diff(aux))
    edges_x = np.unique(np.concatenate((aux-bin_size/2,aux+bin_size/2)))   
    edges_y = np.linspace(np.min(numerical_prob_surrogates_log)-0.1,np.max(numerical_prob_surrogates_log)+0.1,10)
    my_cmap = plt.cm.gray
    _,_,_,Image = sbplt[1][1].hist2d(surr_samples_freqs_log,numerical_prob_surrogates_log,bins=[edges_x, edges_y],cmap = my_cmap)#
    plt.colorbar(Image)
    
    
    
    #now we do the same as for the surrogate real datasets but for the generated dataset
    #get rid of samples already present in the original dataset
    sim_samples_freqs = sim_samples_freqs[freq_in_training_dataset==0]
    numerical_prob = numerical_prob[freq_in_training_dataset==0]
    #get rid of the freqs for which the numerical prob is zero (so we can compute the logs)
    sim_samples_freqs = sim_samples_freqs[numerical_prob!=0]
    numerical_prob = numerical_prob[numerical_prob!=0]
    #compute the logs of the probs and freq
    sim_samples_freqs_log = np.log10(sim_samples_freqs)
    numerical_prob_log = np.log10(numerical_prob)
    
    #transalate ticks to the 10^x format      
    sbplt[1][1].plot(sim_samples_freqs_log,numerical_prob_log,'+b',markersize=2)
    ticks = sbplt[1][1].get_xticks()
    labels = []
    for ind_tck in range(len(ticks)):
        labels.append('$10^{'+str(ticks[ind_tck]) +'}$')
   
    sbplt[1][1].set_xticklabels(labels)
    
    ticks = sbplt[1][1].get_yticks()
    labels = []
    for ind_tck in range(len(ticks)):
        labels.append('$10^{'+str(ticks[ind_tck]) +'}$')
   
    sbplt[1][1].set_yticklabels(labels)    
    
    
    f.savefig(folder+'probs.svg',dpi=600, bbox_inches='tight')
    plt.close(f)
    

def comparison_to_original_and_gt_datasets(samples, real_samples, ground_truth_samples, ground_truth_probs):
    #get freqs of simulated samples
    aux = np.unique(samples,axis=1,return_counts=True)
    sim_samples_probs = aux[1]/np.sum(aux[1])
    sim_samples_unique = aux[0]    
    #get freqs of original samples
    aux = np.unique(real_samples,axis=1,return_counts=True)
    original_samples_probs = aux[1]/np.sum(aux[1])
    original_samples = aux[0]
    #simulated samples that are not in the original dataset
    #if zero, the simulated sample is not present in the original dataset; 
    #if different from zero it stores the frequency with which the sample occurs in the original dataset
    prob_in_training_dataset = np.zeros((sim_samples_unique.shape[1],)) 
    #generated samples that are not in the ground truth dataset and thus have theoretical prob = 0
    #if zero, the simulated sample is not present in the ground truth dataset; 
    #if different from zero it stores the frequency with which the sample occurs 
    numerical_prob = np.zeros((sim_samples_unique.shape[1],))
    start_time = time.time()
    for ind_s in range(sim_samples_unique.shape[1]):
        if ind_s%1000==0:
            print(str(ind_s) + ' time ' + str(time.time() - start_time))
        #get sample
        sample = sim_samples_unique[:,ind_s].reshape(sim_samples_unique.shape[0],1)
        #check whether the sample is in the ground truth dataset and if so get prob
        looking_for_sample = np.equal(ground_truth_samples.T,sample.T).all(1)
        if any(looking_for_sample):
            numerical_prob[ind_s] = ground_truth_probs[looking_for_sample]
        
        #check whether the sample is in the original dataset and if so get prob
        looking_for_sample = np.equal(original_samples.T,sample.T).all(1)
        if any(looking_for_sample):
            prob_in_training_dataset[ind_s] = original_samples_probs[looking_for_sample]
       
            
    return prob_in_training_dataset, numerical_prob, sim_samples_probs        
    
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
    
    
    