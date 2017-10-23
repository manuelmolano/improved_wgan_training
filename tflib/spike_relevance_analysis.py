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
#from tflib import analysis, retinal_data, visualize_filters_and_units, sim_pop_activity
import numpy as np
#from utils import pp, get_samples_autocorrelogram, get_samples
import matplotlib.pyplot as plt
import matplotlib

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
        FLAGS.sample_dir = 'samples ' + FLAGS.architecture + '/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
        '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins)\
        + '_ref_period_' + str(FLAGS.ref_period) + '_firing_rate_' + str(FLAGS.firing_rate) + '_correlation_' + str(FLAGS.correlation) +\
        '_group_size_' + str(FLAGS.group_size)  + '_critic_iters_' + str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
        '_num_layers_' + str(FLAGS.num_layers)  + '_num_features_' + str(FLAGS.num_features) + '_kernel_' + str(FLAGS.kernel_width) +\
        '_iteration_' + FLAGS.iteration + '/'
    elif FLAGS.dataset=='packets':
        FLAGS.sample_dir = 'samples ' + FLAGS.architecture + '/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
        '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins) + '_packet_prob_' + str(FLAGS.packet_prob)\
        + '_firing_rate_' + str(FLAGS.firing_rate) + '_group_size_' + str(FLAGS.group_size)  + '_critic_iters_' +\
        str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
        '_num_layers_' + str(FLAGS.num_layers)  + '_num_features_' + str(FLAGS.num_features) + '_kernel_' + str(FLAGS.kernel_width) +\
        '_iteration_' + FLAGS.iteration + '/'
    elif FLAGS.dataset=='retina':
        FLAGS.sample_dir = 'samples ' + FLAGS.architecture + '/' + 'dataset_' + FLAGS.dataset + '_instance_' + FLAGS.data_instance +\
        '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins) +\
        '_critic_iters_' + str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
        '_num_layers_' + str(FLAGS.num_layers)  + '_num_features_' + str(FLAGS.num_features) + '_kernel_' + str(FLAGS.kernel_width) +\
        '_iteration_' + FLAGS.iteration + '/'
      
    FLAGS.checkpoint_dir = FLAGS.sample_dir + 'checkpoint/'

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)


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
        original_dataset = np.load(FLAGS.sample_dir+ '/stats_real.npz')
        samples = original_dataset['samples'][:,0:num1*num2].T
        
        index = np.argsort(original_dataset['shuffled_index'])
        my_cmap = plt.cm.gray
        
        f1,sbplt1 = plt.subplots(num1,num2,figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
        f2,sbplt2 = plt.subplots(num1,num2,figsize=(8, 8),dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
        for i in range(num1*num2):
            print(i)
            sample = samples[i,:]
            grads = patterns_relevance(sample, FLAGS.num_neurons, wgan, sess, [8,8])
            sample = sample.reshape(FLAGS.num_neurons,FLAGS.num_bins)
            sample = sample[index,:]
            sbplt2[int(np.floor(i/num1))][i%num2].imshow(sample,interpolation='nearest', cmap = my_cmap)  
            sbplt2[int(np.floor(i/num1))][i%num2].axis('off')  
    
            #grads = grads.reshape(FLAGS.num_neurons,FLAGS.num_bins)
            #grads = (grads[index,:])
            sbplt1[int(np.floor(i/num1))][i%num2].imshow(grads,interpolation='nearest', cmap = my_cmap)  
            sbplt1[int(np.floor(i/num1))][i%num2].axis('off')  
            
        
        f1.savefig(FLAGS.sample_dir+'spk_relevance.svg',dpi=600, bbox_inches='tight')
        plt.close(f1)  
        f2.savefig(FLAGS.sample_dir+'_samples_spk_relevance.svg',dpi=600, bbox_inches='tight')
        plt.close(f2)  
                 
 
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
        
def patterns_relevance(sample_original, num_neurons, wgan, sess, pattern_size):
    dim = sample_original.shape[0]
    sample = sample_original.copy()
    sample = sample.reshape((sample.shape[0],1))
    score = wgan.get_critics_output(np.concatenate((sample,sample),axis=1))[0].eval(session=sess)
    sample = sample.reshape((num_neurons,-1))
    num_patterns_1 = int(num_neurons/pattern_size[0])
    num_patterns_2 = int(sample.shape[1]/pattern_size[1])
    samples_shuffled = np.zeros((dim,num_patterns_1*num_patterns_2))
    counter = 0
    for ind_2 in range(num_patterns_2):
        aux_sample = sample.copy()
        aux_pattern = aux_sample[:,ind_2*pattern_size[1]:(ind_2+1)*pattern_size[1]]
        np.random.shuffle(aux_pattern.T)
        samples_shuffled[:,counter] = aux_sample.flatten()
        counter += 1
    grad = wgan.get_critics_output(samples_shuffled).eval(session=sess) - score
    counter = 0
    grad_map = np.zeros(sample.shape)
    for ind_2 in range(num_patterns_2):
        grad_map[:,ind_2*pattern_size[1]:(ind_2+1)*pattern_size[1]] = grad[counter]*sample_original
        counter += 1
        #grad[spikes[ind_spk]] = aux
        
    return grad    
    
if __name__ == '__main__':
  tf.app.run()





