#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:56:25 2017

@author: manuel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 10:13:25 2017

@author: manuel
"""
import sys, os
print(os.getcwd())
sys.path.append('/home/manuel/improved_wgan_training/')
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import os
import pprint
from model_conv import WGAN_conv
from tflib import sim_pop_activity, retinal_data#, visualize_filters_and_units, sim_pop_activity
import numpy as np
#from utils import pp, get_samples_autocorrelogram, get_samples
import matplotlib.pyplot as plt
import matplotlib
import time
#parameters for figure
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.select(-10000. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))



#parameters used for (some) figures
flags = tf.app.flags
flags.DEFINE_string("architecture", "conv", "semi-conv (conv) or fully connected (fc)")
flags.DEFINE_integer("num_iter", 300000, "Epoch to train [50]")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for adam [1e-4]")
flags.DEFINE_float("beta1", 0., "Momentum term of adam [0.]")
flags.DEFINE_float("beta2", 0.9, "Momentum term of adam [0.9]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("training_step", 200, "number of batches between weigths and performance saving")
flags.DEFINE_string("training_stage", '', "stage of the training used for the GAN")
flags.DEFINE_integer("num_layers", 4, "number of convolutional layers [4]")
flags.DEFINE_integer("num_features", 4, "features in first layers [4]")
flags.DEFINE_integer("kernel_width", 4, "width of kernel [4]")

#parameter set specifiying data
flags.DEFINE_string("dataset", "uniform", "type of neural activity. It can be simulated  or retina")
flags.DEFINE_string("data_instance", "1", "if data==retina, this allows chosing the data instance")
flags.DEFINE_integer("num_samples", 2**13, "number of samples")
flags.DEFINE_integer("num_neurons", 4, "number of neurons in the population")
flags.DEFINE_float("packet_prob", 0.05, "probability of packets")
flags.DEFINE_integer("num_bins", 32, "number of bins (ms) for each sample")
flags.DEFINE_string("iteration", "0", "in case several instances are run with the same parameters")
flags.DEFINE_integer("ref_period", 2, "minimum number of ms between spikes (if < 0, no refractory period is imposed)")
flags.DEFINE_float("firing_rate", 0.25, "maximum firing rate of the simulated responses")
flags.DEFINE_float("correlation", 0.9, "correlation between neurons")
flags.DEFINE_integer("group_size", 2, "size of the correlated groups (e.g. 2 means pairwise correlations)")
flags.DEFINE_integer("critic_iters", 5, "number of times the discriminator will be updated")
flags.DEFINE_float("lambd", 10, "parameter gradient penalization")
FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()
def main(_):
    #print parameters
    pp.pprint(flags.FLAGS.__flags)
    #folders
    if FLAGS.dataset=='uniform':
        if FLAGS.architecture=='fc':
            FLAGS.sample_dir = 'samples fc/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
            '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins)\
            + '_ref_period_' + str(FLAGS.ref_period) + '_firing_rate_' + str(FLAGS.firing_rate) + '_correlation_' + str(FLAGS.correlation) +\
            '_group_size_' + str(FLAGS.group_size)  + '_critic_iters_' + str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
            '_num_units_' + str(FLAGS.num_units) +\
            '_iteration_' + FLAGS.iteration + '/'
        elif FLAGS.architecture=='conv':
            FLAGS.sample_dir = 'samples conv/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
            '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins)\
            + '_ref_period_' + str(FLAGS.ref_period) + '_firing_rate_' + str(FLAGS.firing_rate) + '_correlation_' + str(FLAGS.correlation) +\
            '_group_size_' + str(FLAGS.group_size)  + '_critic_iters_' + str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
            '_num_layers_' + str(FLAGS.num_layers)  + '_num_features_' + str(FLAGS.num_features) + '_kernel_' + str(FLAGS.kernel_width) +\
            '_iteration_' + FLAGS.iteration + '/'
    elif FLAGS.dataset=='packets':
        if FLAGS.architecture=='fc':
            FLAGS.sample_dir = 'samples fc/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
            '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins) + '_packet_prob_' + str(FLAGS.packet_prob)\
            + '_firing_rate_' + str(FLAGS.firing_rate) + '_group_size_' + str(FLAGS.group_size)  + '_critic_iters_' +\
            str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) + '_num_units_' + str(FLAGS.num_units) +\
            '_iteration_' + FLAGS.iteration + '/'
        elif FLAGS.architecture=='conv':
            FLAGS.sample_dir = 'samples conv/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
            '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins) + '_packet_prob_' + str(FLAGS.packet_prob)\
            + '_firing_rate_' + str(FLAGS.firing_rate) + '_group_size_' + str(FLAGS.group_size)  + '_critic_iters_' +\
            str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
            '_num_layers_' + str(FLAGS.num_layers)  + '_num_features_' + str(FLAGS.num_features) + '_kernel_' + str(FLAGS.kernel_width) +\
            '_iteration_' + FLAGS.iteration + '/'
    elif FLAGS.dataset=='retina':
       if FLAGS.architecture=='fc':
            FLAGS.sample_dir = 'samples fc/' + 'dataset_' + FLAGS.dataset  +\
            '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins)\
            + '_critic_iters_' + str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
            '_num_units_' + str(FLAGS.num_units) +\
            '_iteration_' + FLAGS.iteration + '/'
       elif FLAGS.architecture=='conv':
            FLAGS.sample_dir = 'samples conv/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
            '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins)\
            + '_critic_iters_' + str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
            '_num_layers_' + str(FLAGS.num_layers)  + '_num_features_' + str(FLAGS.num_features) + '_kernel_' + str(FLAGS.kernel_width) +\
            '_iteration_' + FLAGS.iteration + '/'
      
    FLAGS.checkpoint_dir = FLAGS.sample_dir + 'checkpoint/'


    with tf.Session() as sess:
        wgan = WGAN_conv(sess,
        num_neurons=FLAGS.num_neurons,
        num_bins=FLAGS.num_bins,
        num_layers=FLAGS.num_layers,
        num_features=FLAGS.num_features,
        kernel_width=FLAGS.kernel_width,
        lambd=FLAGS.lambd,
        batch_size=FLAGS.batch_size,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)
        if not wgan.load(FLAGS.training_stage):
            raise Exception("[!] Train a model first, then run test mode")      
            
        num1 = 4
        num2 = 4
        if FLAGS.dataset=='retina':
            samples = retinal_data.get_samples(num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, instance=FLAGS.data_instance).T
        else:
            original_dataset = np.load(FLAGS.sample_dir+ '/stats_real.npz')
            _ = sim_pop_activity.spike_train_transient_packets(num_samples=1000, num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, group_size=FLAGS.group_size,\
                                                                 prob_packets=FLAGS.packet_prob,firing_rates_mat=original_dataset['firing_rate_mat'], refr_per=FLAGS.ref_period,\
                                                                 shuffled_index=original_dataset['shuffled_index'], limits=[16,32], groups=[0,1,2,3], folder=FLAGS.sample_dir, save_packet=True).T
            samples = sim_pop_activity.spike_train_transient_packets(num_samples=1000, num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, group_size=FLAGS.group_size,\
                                                                 prob_packets=FLAGS.packet_prob,firing_rates_mat=original_dataset['firing_rate_mat'], refr_per=FLAGS.ref_period,\
                                                                 shuffled_index=original_dataset['shuffled_index'], limits=[16,32], groups=[0], folder=FLAGS.sample_dir, save_packet=False).T

        inputs = tf.placeholder(tf.float32, name='inputs_to_discriminator', shape=[None, FLAGS.num_neurons*FLAGS.num_bins]) 
        score = wgan.get_critics_output(inputs)
        #real samples    
        f1,sbplt1 = plt.subplots(num1,num2,figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
        f2,sbplt2 = plt.subplots(num1,num2,figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
        f3,sbplt3 = plt.subplots(num1,num2,figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
        num_samples = 2000   
        step = 2
        pattern_size = 8
        times = step*np.arange(int(FLAGS.num_bins/step))
        times = np.delete(times,np.nonzero(times>FLAGS.num_bins-pattern_size))
        #print(times)
        importance_time_vector = np.zeros((num_samples,FLAGS.num_bins))
        importance_neuron_vector = np.zeros((num_samples,FLAGS.num_neurons))
        grad_maps = np.zeros((num_samples,FLAGS.num_neurons,FLAGS.num_bins))
        activity_map = np.zeros((FLAGS.num_neurons,FLAGS.num_bins))
        samples_mat = samples[0:num_samples,:]
        for i in range(num_samples):
            sample = samples[i,:]
            time0 = time.time()
            grad_maps[i,:,:], _ = patterns_relevance(sample, FLAGS.num_neurons, score, inputs, sess, pattern_size, times)
            time1 = time.time()
            importance_time_vector[i,:] = np.mean(grad_maps[i,:,:],axis=0)#/max(np.mean(grads,axis=0))
            importance_neuron_vector[i,:]  = np.mean(grad_maps[i,:,:],axis=1)#/max(np.mean(grads,axis=1))
            sample = sample.reshape(FLAGS.num_neurons,-1)
            activity_map += sample
            print(str(i) + ' time ' + str(time1 - time0))

        
        importance_vectors = {'time':importance_time_vector,'neurons':importance_neuron_vector,'grad_maps':grad_maps,'samples':samples_mat, 'activity_map':activity_map}
        np.savez(FLAGS.sample_dir+'importance_vectors.npz',**importance_vectors)
        
def spikes_relevance(sample, wgan, sess):
    sample = sample.reshape((sample.shape[0],1))
    score = wgan.get_critics_output(np.concatenate((sample,sample),axis=1))[0].eval(session=sess)
    spikes = np.nonzero(sample)[0]
    grad = np.zeros((sample.shape[0],))
    for ind_spk in range(len(spikes)):
        aux_sample = sample.copy()
        aux_sample[spikes[ind_spk]] = 0
        aux = wgan.get_critics_output(np.concatenate((aux_sample,aux_sample),axis=1))[0].eval(session=sess) - score
        grad[spikes[ind_spk]] = aux
        
    return grad
        
def patterns_relevance(sample_original, num_neurons, score, inputs, sess, pattern_size, times):
    #start_time = time.time()
    num_sh = 5
    dim = sample_original.shape[0]
    sample = sample_original.copy()
    sample[sample>1] = 1
    sample = sample.reshape((1,sample.shape[0]))
    score_original = sess.run(score, feed_dict={inputs: np.concatenate((sample,sample),axis=0)})[0]
    #score = wgan.get_critics_output(np.concatenate((sample,sample),axis=0))[0].eval(session=sess)
    sample = sample.reshape((num_neurons,-1))
    #print('time ' + str(time.time() - start_time))
    

    samples_shuffled = np.zeros((num_sh,num_neurons*times.shape[0],dim))
    
    for ind_sh in range(num_sh):
        counter = 0
        for ind_1 in range(times.shape[0]):
            for ind_2 in range(num_neurons):
                aux_sample = sample.copy()
                aux_pattern = aux_sample[ind_2,times[ind_1]:times[ind_1]+pattern_size]
                np.random.shuffle(aux_pattern.T)
                samples_shuffled[ind_sh,counter,:] = aux_sample.flatten()
                counter += 1
        #grad_test[ind_sh,:] = np.abs(score - wgan.get_critics_output(samples_shuffled[ind_sh,:,:]).eval(session=sess))
        #aux = np.abs(score - wgan.get_critics_output(np.concatenate((samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:],\
                                                                     #samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:],\
                                                                     #samples_shuffled[ind_sh,:,:],samples_shuffled[ind_sh,:,:]),axis=0)).eval(session=sess))
        #aux2 = np.abs(score - wgan.get_critics_output(samples_shuffled[ind_sh,:,:]).eval(session=sess))
        #print(grad_test[ind_sh,:])
        #print(aux[0:4])
        #print(aux[4:8])
        #print('----')
        #assert np.all(aux[0:2]==aux2)
    aux = samples_shuffled.reshape((num_neurons*times.shape[0]*num_sh,dim))
    scores = sess.run(score, feed_dict={inputs: aux})
    grad = np.abs(score_original - scores)
    grad = grad.reshape((num_sh,num_neurons*times.shape[0]))
        
      
    grad = np.mean(grad,axis=0)
    grad_map = np.zeros(sample.shape)
    counting_map = np.zeros(sample.shape)
    counter = 0
    for ind_1 in range(times.shape[0]):
        for ind_2 in range(num_neurons):
            grad_map[ind_2,times[ind_1]:times[ind_1]+pattern_size] = \
            grad_map[ind_2,times[ind_1]:times[ind_1]+pattern_size]+grad[counter]*sample[ind_2,times[ind_1]:times[ind_1]+pattern_size]
            counting_map[ind_2,times[ind_1]:times[ind_1]+pattern_size] = counting_map[ind_2,times[ind_1]:times[ind_1]+pattern_size]+1
            counter += 1
    
    grad_map /= counting_map
    return grad_map, grad   
    
if __name__ == '__main__':
  tf.app.run()




#[ 0.11054063  0.96706271  0.09767437  0.10173178]
#[ 0.11054277  0.9670608   0.09767365  0.10172868]
#[ 0.11054277  0.9670608   0.09767365  0.10172868]




