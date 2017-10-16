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
    return tf.select(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))



#parameters used for (some) figures
flags = tf.app.flags
flags.DEFINE_string("architecture", "conv", "semi-conv (conv) or fully connected (fc)")
flags.DEFINE_integer("num_iter", 300000, "Epoch to train [50]")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for adam [1e-4]")
flags.DEFINE_float("beta1", 0., "Momentum term of adam [0.]")
flags.DEFINE_float("beta2", 0.9, "Momentum term of adam [0.9]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
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

  eval_graph = tf.Graph()
  with eval_graph.as_default():
    with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
        with tf.Session(graph=eval_graph) as sess:
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
            aux = np.load(FLAGS.sample_dir+ '/stats_real.npz')
            samples = aux['samples'][:,0:int(FLAGS.batch_size)].T
            #fake_samples = (np.zeros((FLAGS.num_neurons,FLAGS.num_bins)) + aux['firing_rates_mat']) > np.random.random((FLAGS.num_neurons,FLAGS.num_bins))
            
            #samples = tf.placeholder("float", [FLAGS.batch_size, FLAGS.num_neurons*FLAGS.num_bins])
            labels = tf.placeholder(tf.float32, [FLAGS.batch_size, 1])
            critics_output = tf.reshape(tf.sigmoid(wgan.disc_cost),[FLAGS.batch_size, 1])
            
            cost_all = (critics_output - labels) ** 2
            cost = tf.reduce_sum(cost_all)
            #train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

            # Get last convolutional layer gradient for generating gradCAM visualization
            #target_conv_layer = wgan.pool5
            #target_conv_layer_grad = tf.gradients(cost, target_conv_layer)[0]
    
            # Guided backpropagtion back to input layer
            gb_grad = tf.gradients(cost, wgan.inputs)[0]
            #print(gb_grad)
            # Normalizing the gradients    
            #target_conv_layer_grad_norm = tf.div(target_conv_layer_grad, tf.sqrt(tf.reduce_mean(tf.square(target_conv_layer_grad))) + tf.constant(1e-5))

            #prob = sess.run(wgan.disc_cost, feed_dict={wgan.inputs: samples})
            
            #gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad], feed_dict={wgan.inputs: samples})
            #print(samples)
            gb_grad_value, critic_values, cost_values = sess.run([gb_grad,wgan.disc_cost,cost_all], feed_dict={wgan.inputs: samples, labels: np.zeros((FLAGS.batch_size,1))})
            f,sbplt = plt.subplots(1,2,figsize=(8, 8),dpi=250)
            matplotlib.rcParams.update({'font.size': 8})
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
            sbplt[0].plot(critic_values)
            sbplt[1].plot(cost_values)
            f.savefig(FLAGS.sample_dir+'critic_values_guided_bp_test.svg',dpi=600, bbox_inches='tight')
            plt.close(f) 
            my_cmap = plt.cm.gray
            f1,sbplt1 = plt.subplots(8,8,figsize=(8, 8),dpi=250)
            matplotlib.rcParams.update({'font.size': 8})
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
            f2,sbplt2 = plt.subplots(8,8,figsize=(8, 8),dpi=250)
            matplotlib.rcParams.update({'font.size': 8})
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
            for i in range(samples.shape[0]):
                sample = gb_grad_value[i,:]
                sample = sample.reshape(FLAGS.num_neurons,FLAGS.num_bins)
                sbplt1[int(np.floor(i/8))][i%8].imshow(sample,interpolation='nearest', cmap = my_cmap)  
                sbplt1[int(np.floor(i/8))][i%8].axis('off')  
                sbplt1[int(np.floor(i/8))][i%8].set_title('min:' + str(np.min(sample.flatten())) + '  max:' + str(np.max(sample.flatten())))
                sample = samples[i,:]
                sample = sample.reshape(FLAGS.num_neurons,FLAGS.num_bins)
                sbplt2[int(np.floor(i/8))][i%8].imshow(sample,interpolation='nearest', cmap = my_cmap)  
                sbplt2[int(np.floor(i/8))][i%8].axis('off')  
                
  
            f1.savefig(FLAGS.sample_dir+'guided_bp_test.svg',dpi=600, bbox_inches='tight')
            plt.close(f1)  
            f2.savefig(FLAGS.sample_dir+'_samples_guided_bp_test.svg',dpi=600, bbox_inches='tight')
            plt.close(f2)  
            
            
    
    
if __name__ == '__main__':
  tf.app.run()





