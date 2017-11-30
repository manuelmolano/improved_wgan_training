

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:13:56 2017

@author: manuel
"""
import sys#, os
sys.path.append('/home/manuel/improved_wgan_training/')
#import glob

import numpy as np
from tflib import  retinal_data, analysis
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap

#parameters for figure
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots
font_size = 14
margin = 0.02
     
    
def figure_2_3(num_samples, num_neurons, num_bins, folder, folder_fc, fig_2_or_4):
    original_data = np.load(folder + '/stats_real.npz')   
    mean_spike_count_real, autocorrelogram_mat_real, firing_average_time_course_real, cov_mat_real, k_probs_real, lag_cov_mat_real = \
    [original_data["mean"], original_data["acf"], original_data["firing_average_time_course"], original_data["cov_mat"], original_data["k_probs"], original_data["lag_cov_mat"]]
    
    #load conv information
    conv_data = np.load(folder + '/samples_fake.npz')['samples']
    conv_data_bin = (conv_data > np.random.random(conv_data.shape)).astype(float)   
    cov_mat_conv, k_probs_conv, mean_spike_count_conv, autocorrelogram_mat_conv, firing_average_time_course_conv, lag_cov_mat_conv = \
        analysis.get_stats_aux(conv_data_bin, num_neurons, num_bins)
    #load fc information
    if fig_2_or_4==2:
        fc_data = np.load(folder_fc + '/samples_fake.npz')['samples']
        fc_data_bin = (fc_data > np.random.random(fc_data.shape)).astype(float)   
        cov_mat_comp, k_probs_comp, mean_spike_count_comp, autocorrelogram_mat_comp, firing_average_time_course_comp, lag_cov_mat_comp = \
            analysis.get_stats_aux(fc_data_bin, num_neurons, num_bins)
    elif fig_2_or_4==4:
        k_pairwise_samples = retinal_data.load_samples_from_k_pairwise_model(num_samples=num_samples, num_bins=num_bins, num_neurons=num_neurons, instance='1')    
        cov_mat_comp, k_probs_comp, mean_spike_count_comp, autocorrelogram_mat_comp, firing_average_time_course_comp, lag_cov_mat_comp = \
            analysis.get_stats_aux(k_pairwise_samples, num_neurons, num_bins)
    
    
    only_cov_mat_conv = cov_mat_conv.copy()
    only_cov_mat_conv[np.diag_indices(num_neurons)] = np.nan
    only_cov_mat_comp = cov_mat_comp.copy()
    only_cov_mat_comp[np.diag_indices(num_neurons)] = np.nan

    #PLOT
    
    index = np.linspace(-10,10,2*10+1)
    #figure for all training error across epochs (supp. figure 2)
    if fig_2_or_4==2:
        f = plt.figure(figsize=(8, 10),dpi=250)
    else:
        f = plt.figure(figsize=(10, 6),dpi=250)
    
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    
    
    if fig_2_or_4==2:
        plt.subplot(3,1,1)
        num_rows = 1
        num_cols = 1
        for ind_s in range(num_rows*num_cols):
            sample = conv_data[:,ind_s].reshape((num_neurons,-1))
            sample_binnarized = conv_data_bin[:,ind_s].reshape((num_neurons,-1))
            for ind_n in range(num_neurons):
                plt.plot(sample[ind_n,:]+4*ind_n,'k')
                spks = np.nonzero(sample_binnarized[ind_n,:])[0]
                for ind_spk in range(len(spks)):
                    plt.plot(np.ones((2,))*spks[ind_spk],4*ind_n+np.array([2.2,3.2]),'r')
            #sbplt.axis('off')
            plt.axis('off')
            plt.xlim(0,num_bins)
            plt.ylim(-1,65)
            ax = plt.gca()
            points = ax.get_position().get_points()
            plt.text(points[0][0]-margin,points[1][1]+margin, 'A', fontsize=font_size, transform=plt.gcf().transFigure)
            
            
            
    if fig_2_or_4==2:
        plt.subplot(3,3,8)
    else:
        plt.subplot(2,3,5)
    #plot autocorrelogram(s)
    plt.plot(index, autocorrelogram_mat_conv,'r')
    plt.plot(index, autocorrelogram_mat_comp,'g')
    plt.plot(index, autocorrelogram_mat_real,'b')
    plt.title('Autocorrelogram')
    plt.xlabel('time (ms)')
    plt.ylabel('number of spikes')
    ax = plt.gca()
    points = ax.get_position().get_points()
    if fig_2_or_4==2:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'F', fontsize=font_size, transform=plt.gcf().transFigure)
    else:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'E', fontsize=font_size, transform=plt.gcf().transFigure)
               
    
    #plot mean firing rates
    mean_spike_count_real = mean_spike_count_real*1000/num_bins
    mean_spike_count_conv = mean_spike_count_conv*1000/num_bins
    mean_spike_count_comp = mean_spike_count_comp*1000/num_bins
    maximo = np.max(np.array([115,np.max(mean_spike_count_real),np.max(mean_spike_count_conv),np.max(mean_spike_count_comp)]))
    minimo = np.min(np.array([75,np.min(mean_spike_count_real),np.min(mean_spike_count_conv),np.min(mean_spike_count_comp)]))
    if fig_2_or_4==2:
        plt.subplot(3,3,4)
    else:
        plt.subplot(2,3,1)
    
    axis_ticks = np.linspace(minimo,maximo,3)
    plt.plot([minimo,maximo],[minimo,maximo],'k')
    plt.plot(mean_spike_count_real,mean_spike_count_conv,'.r')
    plt.plot(mean_spike_count_real,mean_spike_count_comp,'.g')
    plt.xlabel('mean firing rate expt (Hz)')
    plt.ylabel('mean firing rate models (Hz)')   
    plt.title('mean firing rates')
    ax = plt.gca()
    points = ax.get_position().get_points()
    if fig_2_or_4==2:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'B', fontsize=font_size, transform=plt.gcf().transFigure)
        plt.annotate('Spike-GAN',xy=(minimo,maximo-(maximo-minimo)/10),fontsize=8,color='r')
        plt.annotate('MLP GAN',xy=(minimo,maximo-2*(maximo-minimo)/10),fontsize=8,color='g')
    else:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'A', fontsize=font_size, transform=plt.gcf().transFigure)
        plt.annotate('Spike-GAN',xy=(minimo,maximo-(maximo-minimo)/10),fontsize=8,color='r')
        plt.annotate('k-pairwise model',xy=(minimo,maximo-2*(maximo-minimo)/10),fontsize=8,color='g')

    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #plot covariances
    if fig_2_or_4==2:
        plt.subplot(3,3,5)
    else:
        plt.subplot(2,3,2)
    only_cov_mat_real = cov_mat_real.copy()
    only_cov_mat_real[np.diag_indices(num_neurons)] = np.nan
    axis_ticks = np.linspace(np.nanmin(only_cov_mat_real.flatten()),np.nanmax(only_cov_mat_real.flatten()),3)
    plt.plot([np.nanmin(only_cov_mat_real.flatten()),np.nanmax(only_cov_mat_real.flatten())],\
                    [np.nanmin(only_cov_mat_real.flatten()),np.nanmax(only_cov_mat_real.flatten())],'k')
    plt.plot(only_cov_mat_real.flatten(),only_cov_mat_conv.flatten(),'.r')
    plt.plot(only_cov_mat_real.flatten(),only_cov_mat_comp.flatten(),'.g')
    plt.title('pairwise covariances')
    plt.xlabel('covariances expt')
    plt.ylabel('covariances models')
    ax = plt.gca()
    points = ax.get_position().get_points()
    if fig_2_or_4==2:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'C', fontsize=font_size, transform=plt.gcf().transFigure)
    else:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'B', fontsize=font_size, transform=plt.gcf().transFigure)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
    #plot k-statistics
    if fig_2_or_4==2:
        plt.subplot(3,3,6)
    else:
        plt.subplot(2,3,3)
    axis_ticks = np.linspace(0,np.max(k_probs_real),3)
    plt.plot([0,np.max(k_probs_real)],[0,np.max(k_probs_real)],'k')        
    plt.plot(k_probs_real,k_probs_conv,'.r')        
    plt.plot(k_probs_real,k_probs_comp,'.g')  
    plt.xlabel('k-probs expt')
    plt.ylabel('k-probs models')
    plt.title('k statistics') 
    ax = plt.gca()
    points = ax.get_position().get_points()
    if fig_2_or_4==2:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'D', fontsize=font_size, transform=plt.gcf().transFigure)
    else:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'C', fontsize=font_size, transform=plt.gcf().transFigure)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    #plot average time course
    #firing_average_time_course[firing_average_time_course>0.048] = 0.048
    if fig_2_or_4==2:
        plt.subplot(9,3,19)
    else:
        plt.subplot(6,3,10)
    firing_average_time_course_real_section = firing_average_time_course_real*1000
    firing_average_time_course_conv_section = firing_average_time_course_conv*1000
    firing_average_time_course_comp_section = firing_average_time_course_comp*1000
    maximo = np.max(firing_average_time_course_real_section.flatten())
    minimo = np.min(firing_average_time_course_real_section.flatten())
    aspect_ratio = firing_average_time_course_real_section.shape[1]/(2*firing_average_time_course_real_section.shape[0])
    plt.imshow(firing_average_time_course_real_section,interpolation='nearest', cmap='viridis', aspect=aspect_ratio)
    plt.title('Real time course (Hz)')
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    points = ax.get_position().get_points()
    if fig_2_or_4==2:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'E', fontsize=font_size, transform=plt.gcf().transFigure)
    else:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'D', fontsize=font_size, transform=plt.gcf().transFigure)
    if fig_2_or_4==2:
        plt.subplot(9,3,22)
    else:
        plt.subplot(6,3,13)
    
    plt.imshow(firing_average_time_course_conv_section,interpolation='nearest', clim=(minimo,maximo), cmap='viridis',aspect=aspect_ratio)
    plt.title('Spike-GAN time course')
    plt.xticks([])
    plt.yticks([])
    if fig_2_or_4==2:
        plt.subplot(9,3,25)
    else:
        plt.subplot(6,3,16)
    
    map_aux = plt.imshow(firing_average_time_course_comp_section,interpolation='nearest', clim=(minimo,maximo), cmap='viridis',aspect=aspect_ratio)#map_aux = 
    if fig_2_or_4==2:
        plt.title('MLP GAN time course')
    else:
        plt.title('k-pairwise time course')
    plt.xlabel('time (ms)')
    plt.ylabel('neuron')
    #f.colorbar(map_aux,orientation='horizontal')
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    points_colorbar = ax.get_position().get_points()
    ticks_values = [np.floor(minimo)+1, np.floor((maximo+minimo)/2), np.floor(maximo)]
    print(ticks_values)
    if fig_2_or_4==2:
        cbaxes = f.add_axes([points_colorbar[0][0]+(points_colorbar[1][0]-points_colorbar[0][0])/6, points_colorbar[0][1]-0.03, (points_colorbar[1][0]-points_colorbar[0][0])/1.5, 0.01]) 
        plt.colorbar(map_aux, orientation='horizontal', cax = cbaxes, ticks=ticks_values)    
    else:
        cbaxes = f.add_axes([points_colorbar[0][0]+(points_colorbar[1][0]-points_colorbar[0][0])/6, points_colorbar[0][1]-0.05, (points_colorbar[1][0]-points_colorbar[0][0])/1.5, 0.01]) 
        plt.colorbar(map_aux, orientation='horizontal', cax = cbaxes, ticks=ticks_values)    
    #plot lag covariance
    if fig_2_or_4==2:
        plt.subplot(3,3,9)
    else:
        plt.subplot(2,3,6)
    axis_ticks = np.linspace(np.min(lag_cov_mat_real.flatten()),np.max(lag_cov_mat_real.flatten()),3)
    plt.plot([np.min(lag_cov_mat_real.flatten()),np.max(lag_cov_mat_real.flatten())],\
                        [np.min(lag_cov_mat_real.flatten()),np.max(lag_cov_mat_real.flatten())],'k')        
    plt.plot(lag_cov_mat_real,lag_cov_mat_conv,'.r')
    plt.plot(lag_cov_mat_real,lag_cov_mat_comp,'.g')
    plt.xlabel('lag cov real')
    plt.ylabel('lag cov models')
    plt.title('lag covariances')
    ax = plt.gca()
    points = ax.get_position().get_points()
    if fig_2_or_4==2:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'G', fontsize=font_size, transform=plt.gcf().transFigure)
    else:
        plt.text(points[0][0]-margin,points[1][1]+margin, 'F', fontsize=font_size, transform=plt.gcf().transFigure)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    
    
    f.savefig(folder+'figure_'+str(fig_2_or_4)+'.svg',dpi=600, bbox_inches='tight')
    plt.close(f)
    return points_colorbar, cbaxes, map_aux, maximo, minimo

def figure_4(num_samples, num_neurons, num_bins, folder):
    colors = np.divide(np.array([(0, 0, 0), (128, 128, 128),(166,206,227),(31,120,180),(51,160,44),(251,154,153),(178,223,138)]),256)
    cm = LinearSegmentedColormap.from_list('my_map', colors, N=7)
    
    #original_data = np.load(folder + '/stats_real.npz') 
    importance_maps = np.load(folder+'importance_vectors.npz')
    f = plt.figure(figsize=(8, 10),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    
    num_cols = 5
    num_rows = 1
    if num_rows==1 and folder.find('retina')==-1:
        packet = np.load(folder+'packet.npz')
        maximo = np.max(packet['packet'].flatten())
        cbaxes = f.add_axes([0.1,0.85,0.4,0.25]) 
        plt.imshow(packet['packet'],interpolation='nearest',clim=[0,maximo],cmap=cm)
        plt.xlabel('time (ms)')
        plt.ylabel('neurons')
        plt.title('ideal packet')
        ax = plt.gca()
        points_colorbar = ax.get_position().get_points()
        plt.text(0.04,1.075, 'A', fontsize=14, transform=plt.gcf().transFigure)
        cbaxes = f.add_axes([0.57,0.85,0.4,0.25]) 
        plt.imshow(packet['result'],interpolation='nearest',clim=[0,maximo],cmap=cm)
        plt.axis('off')
        plt.title('realistic population activity')
        plt.text(0.55,1.075, 'B', fontsize=14, transform=plt.gcf().transFigure)
        #title
        plt.text(0.53,points_colorbar[0][1]-0.005, 'Importance Maps', ha='center', fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.04,points_colorbar[0][1]-0.005, 'c', fontsize=14, transform=plt.gcf().transFigure)
    else:
        points_colorbar = np.array([[0.01,0.095],0,0])
    #index = np.argsort(original_data['shuffled_index'])
   
    num_samples = num_cols*num_rows
    if num_rows==1:
        grad_maps = importance_maps['grad_maps'][[0,1,5,6,7],:,:]
        samples = importance_maps['samples'][[0,1,5,6,7],:]
    else:
        aux = np.random.choice(np.arange(num_samples),size=(num_samples,))#range(num_samples)
        grad_maps = importance_maps['grad_maps'][aux,:,:]
        samples = importance_maps['samples'][aux,:]
        
    #panels position params
    if folder.find('retina')!=-1:
        width = 0.4
        height = width*num_bins/(3*num_neurons)
        margin = width/10
        factor_width = 0.25
        factor_height = 0
    else:
        width = 0.17
        height = width*num_bins/(3*num_neurons)
        margin = width/10
        factor_width = 0
        factor_height = 0.045
        
    for i in range(num_samples):
        pos_h = (points_colorbar[0][0]-0.03)+(i%num_cols)*(width-factor_width+margin)
        pos_v = (points_colorbar[0][1]-0.11)-2*(height-factor_height+0.005)*np.floor(i/num_cols)
        cbaxes = f.add_axes([pos_h, pos_v, width, height]) 
        sample = samples[i,:]
        sample = sample.reshape(num_neurons,num_bins)
        cbaxes.imshow(sample,interpolation='nearest',clim=[0,np.max(samples.flatten())],cmap='gray')
        cbaxes.axis('off')  
        
        pos_h = (points_colorbar[0][0]-0.03)+(i%num_cols)*(width-factor_width+margin)
        pos_v = (points_colorbar[0][1]-0.11)-2*(height-factor_height+0.005)*np.floor(i/num_cols)-height+factor_height
        cbaxes = f.add_axes([pos_h,pos_v, width, height]) 
        cbaxes.imshow(grad_maps[i,:,:],interpolation='nearest', cmap = plt.cm.hot, clim=[0,np.max(grad_maps.flatten())])  
        cbaxes.axis('off')  
        reference = (points_colorbar[0][1]-0.11)-2*(height-factor_height+0.005)*np.floor(i/num_cols)-height+factor_height
    
    if num_rows==1:
        importance_time_vector = importance_maps['time']
        importance_neuron_vector = importance_maps['neurons']
        importance_time_vector_surr = importance_maps['time_surr']
        importance_neuron_vector_surr = importance_maps['neurons_surr']
        cbaxes = f.add_axes([0.1,reference-0.3,0.4,0.25]) 
        plt.errorbar(np.arange(num_bins), np.mean(importance_time_vector,axis=0), yerr=np.std(importance_time_vector,axis=0)/np.sqrt(importance_time_vector.shape[0]))
        plt.errorbar(np.arange(num_bins), np.mean(importance_time_vector_surr,axis=0), yerr=np.std(importance_time_vector_surr,axis=0)/np.sqrt(importance_time_vector_surr.shape[0]),color=(.7,.7,.7))
        plt.ylabel('average importance (a.u.)')
        plt.xlabel('time (ms)')
        plt.title('importance of different time periods')
        plt.xlim(-1,num_bins)
        plt.text(0.04,reference-0.025, 'D', fontsize=14, transform=plt.gcf().transFigure)
        cbaxes = f.add_axes([0.57,reference-0.3,0.4,0.25]) 
        plt.bar(np.arange(num_neurons), np.mean(importance_neuron_vector,axis=0), yerr=np.std(importance_neuron_vector,axis=0)/np.sqrt(importance_neuron_vector.shape[0]))
        plt.bar(np.arange(num_neurons), np.mean(importance_neuron_vector_surr,axis=0), yerr=np.std(importance_neuron_vector_surr,axis=0)/np.sqrt(importance_neuron_vector_surr.shape[0]),color=(.7,.7,.7))
        plt.xlabel('neurons')
        plt.title('importance of different neurons')
        plt.xlim(-1,num_neurons+1)
        plt.text(0.55,reference-0.025, 'E', fontsize=14, transform=plt.gcf().transFigure)
        f.savefig(sample_dir+'figure_4_reduced.svg',dpi=600, bbox_inches='tight')
    else:
        f.savefig(sample_dir+'figure_4_many_samples.svg',dpi=600, bbox_inches='tight')
    plt.close(f)
    if folder.find('retina')!=-1:
       activity_map = importance_maps['activity_map']
       f = plt.figure(figsize=(8, 10),dpi=250)
       cbaxes = f.add_axes([0.1,0.1,0.4,0.7]) 
       plt.errorbar(np.arange(num_bins), np.mean(activity_map,axis=0), yerr=np.std(activity_map,axis=0)/np.sqrt(activity_map.shape[0]))
       plt.ylabel('average importance (a.u.)')
       plt.xlabel('time (ms)')
       plt.title('activity of different time periods')
       plt.xlim(-1,num_bins)
       plt.text(0.04,reference-0.025, 'D', fontsize=14, transform=plt.gcf().transFigure)
       cbaxes = f.add_axes([0.57,0.1,0.4,0.7]) 
       plt.bar(np.arange(num_neurons), np.mean(activity_map,axis=1), yerr=np.std(activity_map,axis=1)/np.sqrt(activity_map.shape[1]))
       plt.xlabel('neurons')
       plt.title('activity of different neurons')
       plt.xlim(-1,num_neurons+1)
       plt.text(0.55,reference-0.025, 'E', fontsize=14, transform=plt.gcf().transFigure)
       f.savefig(sample_dir+'average_activity.svg',dpi=600, bbox_inches='tight')
    
    
if __name__ == '__main__':
    plt.close('all')
    #FIGURE 2 (16*496)
#    dataset = 'uniform'
#    num_samples = '8192'
#    num_neurons = '16'
#    num_bins = '496'
#    ref_period = '2'
#    firing_rate = '0.25'
#    correlation = '0.3'
#    group_size = '2'
#    critic_iters = '5'
#    lambd = '10' 
#    num_layers = '3'
#    num_features = '64'
#    kernel = '4'
#    iteration = '20'
#    num_units = '310'
#    sample_dir = '/home/manuel/improved_wgan_training/samples conv/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
#          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
#          + '_ref_period_' + ref_period + '_firing_rate_' + firing_rate + '_correlation_' + correlation +\
#          '_group_size_' + group_size + '_critic_iters_' + critic_iters + '_lambda_' + lambd +\
#          '_num_layers_' + num_layers + '_num_features_' + num_features + '_kernel_' + kernel +\
#          '_iteration_' + iteration + '/'
#    sample_dir_fc = '/home/manuel/improved_wgan_training/samples fc/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
#          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
#          + '_ref_period_' + ref_period + '_firing_rate_' + firing_rate + '_correlation_' + correlation +\
#          '_group_size_' + group_size + '_critic_iters_' + critic_iters + '_lambda_' + lambd + '_num_units_' + num_units +\
#          '_iteration_' + iteration + '/'
#          
#    points_colorbar, cbaxes, map_aux, maximo, minimo= figure_2_3(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir, folder_fc=sample_dir_fc, fig_2_or_4=2)
#    asdasds
    
    
    #FIGURE 4
    dataset = 'packets'
    num_samples = '8192'
    num_neurons = '32'
    num_bins = '64'
    critic_iters = '5'
    lambd = '10' 
    num_layers = '2'
    num_features = '128'
    kernel = '5'
    iteration = '21'
    packet_prob = '0.1'
    firing_rate = '0.1'
    group_size = '8'

    sample_dir = '/home/manuel/improved_wgan_training/samples conv/' + 'dataset_' + dataset + '_num_samples_' + str(num_samples) +\
          '_num_neurons_' + str(num_neurons) + '_num_bins_' + str(num_bins) + '_packet_prob_' + str(packet_prob)\
          + '_firing_rate_' + str(firing_rate) + '_group_size_' + str(group_size)  + '_critic_iters_' +\
          str(critic_iters) + '_lambda_' + str(lambd) +\
          '_num_layers_' + str(num_layers)  + '_num_features_' + str(num_features) + '_kernel_' + str(kernel) +\
          '_iteration_' + iteration + '/'
    figure_4(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir)
    asdasd
    
    
    #FIGURE 4 tests with retina
    dataset = 'retina'
    num_samples = '8192'
    num_neurons = '50'
    num_bins = '32'
    critic_iters = '5'
    lambd = '10' 
    num_layers = '2'
    num_features = '128'
    kernel = '5'
    iteration = '21'
   

    sample_dir = '/home/manuel/improved_wgan_training/samples conv/' + 'dataset_' + dataset + '_num_samples_' + str(num_samples) +\
            '_num_neurons_' + str(num_neurons) + '_num_bins_' + str(num_bins)\
            + '_critic_iters_' + str(critic_iters) + '_lambda_' + str(lambd) +\
            '_num_layers_' + str(num_layers)  + '_num_features_' + str(num_features) + '_kernel_' + str(kernel) +\
            '_iteration_' + iteration + '/'
    figure_4(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir)
    asdasd
    
  
#     #FIGURE 3
#    dataset = 'retina'
#    num_samples = '8192'
#    num_neurons = '50'
#    num_bins = '32'
#    critic_iters = '5'
#    lambd = '10' 
#    num_layers = '2'
#    num_features = '128'
#    kernel = '5'
#    iteration = '21'
#    sample_dir = '/home/manuel/improved_wgan_training/samples conv/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
#          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
#          + '_critic_iters_' + critic_iters + '_lambda_' + lambd +\
#          '_num_layers_' + num_layers + '_num_features_' + num_features + '_kernel_' + kernel +\
#          '_iteration_' + iteration + '/'
#    figure_2_3(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir,folder_fc='', fig_2_or_4=4)
#    
#    
#    
#    #FIGURE 2
#    dataset = 'uniform'
#    num_samples = '8192'
#    num_neurons = '16'
#    num_bins = '128'
#    ref_period = '2'
#    firing_rate = '0.25'
#    correlation = '0.3'
#    group_size = '2'
#    critic_iters = '5'
#    lambd = '10' 
#    num_layers = '2'
#    num_features = '128'
#    kernel = '5'
#    iteration = '20'
#    num_units = '490'
#    sample_dir = '/home/manuel/improved_wgan_training/samples conv/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
#          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
#          + '_ref_period_' + ref_period + '_firing_rate_' + firing_rate + '_correlation_' + correlation +\
#          '_group_size_' + group_size + '_critic_iters_' + critic_iters + '_lambda_' + lambd +\
#          '_num_layers_' + num_layers + '_num_features_' + num_features + '_kernel_' + kernel +\
#          '_iteration_' + iteration + '/'
#    sample_dir_fc = '/home/manuel/improved_wgan_training/samples fc/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
#          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
#          + '_ref_period_' + ref_period + '_firing_rate_' + firing_rate + '_correlation_' + correlation +\
#          '_group_size_' + group_size + '_critic_iters_' + critic_iters + '_lambda_' + lambd + '_num_units_' + num_units +\
#          '_iteration_' + iteration + '/'
#          
#    points_colorbar, cbaxes, map_aux, maximo, minimo= figure_2_3(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir, folder_fc=sample_dir_fc, fig_2_or_4=2)
#    asdasd
    
    
    
    
    
    #adasdasdasd
   #FIGURE 2 (32*256)
#    dataset = 'uniform'
#    num_samples = '8192'
#    num_neurons = '32'
#    num_bins = '256'
#    ref_period = '2'
#    firing_rate = '0.25'
#    correlation = '0.3'
#    group_size = '2'
#    critic_iters = '5'
#    lambd = '10' 
#    num_layers = '2'
#    num_features = '128'
#    kernel = '5'
#    iteration = '20'
#    num_units = '310'
#    sample_dir = '/home/manuel/improved_wgan_training/samples conv/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
#          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
#          + '_ref_period_' + ref_period + '_firing_rate_' + firing_rate + '_correlation_' + correlation +\
#          '_group_size_' + group_size + '_critic_iters_' + critic_iters + '_lambda_' + lambd +\
#          '_num_layers_' + num_layers + '_num_features_' + num_features + '_kernel_' + kernel +\
#          '_iteration_' + iteration + '/'
#    sample_dir_fc = '/home/manuel/improved_wgan_training/samples fc/' + 'dataset_' + dataset + '_num_samples_' + num_samples +\
#          '_num_neurons_' + num_neurons + '_num_bins_' + num_bins\
#          + '_ref_period_' + ref_period + '_firing_rate_' + firing_rate + '_correlation_' + correlation +\
#          '_group_size_' + group_size + '_critic_iters_' + critic_iters + '_lambda_' + lambd + '_num_units_' + num_units +\
#          '_iteration_' + iteration + '/'
#          
#    points_colorbar, cbaxes, map_aux, maximo, minimo= figure_2_3(num_samples=int(num_samples), num_neurons=int(num_neurons), num_bins=int(num_bins), folder=sample_dir, folder_fc=sample_dir_fc, fig_2_or_4=2)

    
    