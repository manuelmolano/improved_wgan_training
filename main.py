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
from model import WGAN
#from utils import pp, get_samples_autocorrelogram, get_samples


import tensorflow as tf
LAMBDA = 10 # Gradient penalty lambda hyperparameter
flags = tf.app.flags
flags.DEFINE_integer("num_iter", 20000, "Epoch to train [50]")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for adam [1e-4]")
flags.DEFINE_float("beta1", 0., "Momentum term of adam [0.]")
flags.DEFINE_float("beta2", 0.9, "Momentum term of adam [0.9]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("training_step", 200, "number of batches between weigths and performance saving")
flags.DEFINE_string("training_stage", '', "stage of the training used for the GAN")
#parameter set specifiying data
flags.DEFINE_string("dataset", "uniform", "type of neural activity")
flags.DEFINE_integer("num_samples", 2**13, "number of samples")
flags.DEFINE_integer("num_neurons", 4, "number of neurons in the population")
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
  
  pp.pprint(flags.FLAGS.__flags)
  
  FLAGS.sample_dir = 'samples/' + 'dataset_' + FLAGS.dataset + '_num_samples_' + str(FLAGS.num_samples) +\
  '_num_neurons_' + str(FLAGS.num_neurons) + '_num_bins_' + str(FLAGS.num_bins)\
  + '_ref_period_' + str(FLAGS.ref_period) + '_firing_rate_' + str(FLAGS.firing_rate) + '_correlation_' + str(FLAGS.correlation) +\
  '_group_size_' + str(FLAGS.group_size)  + '_critic_iters_' + str(FLAGS.critic_iters) + '_lambda_' + str(FLAGS.lambd) +\
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
    wgan = WGAN(sess,
        num_neurons=FLAGS.num_neurons,
        num_bins=FLAGS.num_bins,
        lambd=FLAGS.lambd,
        batch_size=FLAGS.batch_size,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)

    if FLAGS.is_train:
      #import or generate data
      wgan.train(FLAGS)
    else:
      if not wgan.load(FLAGS.checkpoint_dir,FLAGS.training_stage):
        raise Exception("[!] Train a model first, then run test mode")      

  
    # Below is the code for visualization
    #plot statistics
    get_samples_autocorrelogram(sess, wgan,'fake',FLAGS,0,0)
    #plot some generated samples
    get_samples(sess, wgan,FLAGS.sample_dir)
    

if __name__ == '__main__':
  tf.app.run()