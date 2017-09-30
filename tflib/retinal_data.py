# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:25:22 2017

@author: manuel
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:31:05 2017

@author: manuel
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
#import time

def get_samples(num_bins=27, num_neurons=10, instance='1'):                        
    
    
    mat_contents = sio.loadmat('/home/manuel/generative-neural-models-master/k_pairwise/results/data_num_neurons_' + str(num_neurons) + '_' + instance + '.mat')    
    data = mat_contents['data']
    
    if num_bins!=1:
        data = data[0:data.shape[0]-data.shape[0]%num_bins,:]
        X = np.zeros((num_bins*data.shape[1],int(data.shape[0]/num_bins)))
        for ind_s in range(int(data.shape[0]/num_bins)):
            sample = data[ind_s*num_bins:(ind_s+1)*num_bins,:].T
            X[:,ind_s] = sample.reshape((num_neurons*num_bins,-1))[:,0] 
    else:
        X = data.T
    return X
        
        
def load_samples_from_k_pairwise_model(num_samples=2**13, num_bins=27, num_neurons=10, instance='1'):
    mat_contents = sio.loadmat('/home/manuel/generative-neural-models-master/k_pairwise/results/simulated_samples_num_neurons_' + str(num_neurons) + '_' + instance + '.mat')    
    data = mat_contents['samples_batch_all']
    
    if num_bins!=1:
        data = data[0:data.shape[0]-data.shape[0]%num_bins,:]
        X = np.zeros((num_bins*data.shape[1],int(data.shape[0]/num_bins)))
        for ind_s in range(int(data.shape[0]/num_bins)):
            sample = data[ind_s*num_bins:(ind_s+1)*num_bins,:].T
            X[:,ind_s] = sample.reshape((num_neurons*num_bins,-1))[:,0] 
    
    else:
        X = data.T


    assert num_samples<X.shape[1]
    np.random.shuffle(X.T)
    X = X[:,0:num_samples]
    return X
   


    
if __name__ == '__main__':
    X = get_samples()
    plt.imshow(X[:,1:1000])
    
    
    
    
#    mat_contents = sio.loadmat('/home/manuel/generative-neural-models-master/bint_fishmovie32_100.mat')   
#    data = mat_contents['bint']
#    data_rearranged = np.transpose(data,(0,2,1))
#    data_all = data_rearranged.reshape(data_rearranged.shape[0]*data_rearranged.shape[1],data_rearranged.shape[2])


#    test = np.zeros((data_rearranged.shape[0]*data_rearranged.shape[1],data_rearranged.shape[2]))
#    for ind_trial in range(data.shape[0]):
#        test[ind_trial*data.shape[2]:(ind_trial+1)*data.shape[2]] = data[ind_trial,:,:].T
#     
#    assert np.all(data_all==test)
