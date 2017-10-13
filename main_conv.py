#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 11:35:51 2017

@author: manuel
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:30:47 2017

@author: manuel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:27:20 2017

@author: manuel
"""

import os
import pprint
from model_conv import WGAN_conv
from tflib import analysis, retinal_data, visualize_filters_and_units, sim_pop_activity
import numpy as np
#from utils import pp, get_samples_autocorrelogram, get_samples


import tensorflow as tf
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
      '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins) + 'packet_prob_' + str(FLAGS.packet_prob)\
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

  
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  
  with tf.Session(config=run_config) as sess:
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

    if FLAGS.is_train:
      wgan.train(FLAGS)
    else:
      if not wgan.load(FLAGS.training_stage):
        raise Exception("[!] Train a model first, then run test mode")      

    #get filters
    print('get activations -----------------------------------')
    output,units,inputs = wgan.get_units(num_samples=2**15)  
    if FLAGS.architecture=='conv':
        visualize_filters_and_units.plot_untis_rf_conv(units,output, inputs, sess, FLAGS)
    elif FLAGS.architecture=='fc':
        visualize_filters_and_units.plot_untis_rf(units, output, inputs, sess, FLAGS)
        
    print('get filters -----------------------------------')
    filters = wgan.get_filters(num_samples=64)
    visualize_filters_and_units.plot_filters(filters, sess, FLAGS)
    
    #get samples and their statistics
    fake_samples = wgan.get_samples(num_samples=FLAGS.num_samples)
    fake_samples = fake_samples.eval(session=sess)
    fake_samples = wgan.binarize(samples=fake_samples)    
    _,_,_,_,_ = analysis.get_stats(X=fake_samples.T, num_neurons=FLAGS.num_neurons, num_bins= FLAGS.num_bins, folder=FLAGS.sample_dir, name='fake', instance=FLAGS.data_instance)

    #plot samples
    sim_pop_activity.plot_samples(fake_samples.T, FLAGS.num_neurons, FLAGS.sample_dir, 'fake')
    aux = np.load(FLAGS.sample_dir+ '/stats_real.npz')
    real_samples = aux['samples']
    sim_pop_activity.plot_samples(real_samples, FLAGS.num_neurons, FLAGS.sample_dir, 'real')
    
    #compare with k-pairwise model samples
    if FLAGS.dataset=='retina':
        k_pairwise_samples = retinal_data.load_samples_from_k_pairwise_model(num_samples=FLAGS.num_samples, num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons, instance=FLAGS.data_instance)    
        print(k_pairwise_samples.shape)
        _,_,_,_ ,_ = analysis.get_stats(X=k_pairwise_samples, num_neurons=FLAGS.num_neurons, num_bins= FLAGS.num_bins, folder=FLAGS.sample_dir, name='k_pairwise', instance=FLAGS.data_instance)
    
    #evaluate high order features approximation
    if FLAGS.dataset=='uniform' and False:
        analysis.evaluate_approx_distribution(X=fake_samples.T, folder=FLAGS.sample_dir, num_samples_theoretical_distr=2**21,num_bins=FLAGS.num_bins, num_neurons=FLAGS.num_neurons,\
                            group_size=FLAGS.group_size,refr_per=FLAGS.ref_period)



    
    
if __name__ == '__main__':
  tf.app.run()
