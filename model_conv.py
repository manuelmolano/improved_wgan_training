#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 11:17:31 2017

@author: manuel
"""

from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from functools import wraps
import sys
sys.path.append(os.getcwd())
from tflib import plot, sim_pop_activity, params_with_name, analysis, retinal_data
from tflib.ops import linear, act_funct, conv2d_II, deconv2d
from tensorflow.python.framework import ops as options
import matplotlib.pyplot as plt

#parameters used for (some) figures
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots

#not sure this is necessary
options.reset_default_graph()

def compatibility_decorator(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    name = kwds.pop('name', None)
    return f(targets=kwds['labels'], logits=kwds['logits'], name=name)
  return wrapper
   
# compatibility for TF v<1.0
if int(tf.__version__.split('.')[0]) < 1:
  tf.concat = tf.concat_v2
  tf.nn.sigmoid_cross_entropy_with_logits = compatibility_decorator(tf.nn.sigmoid_cross_entropy_with_logits)

class WGAN_conv(object):
  def __init__(self, sess, batch_size=64, lambd=10,
               num_neurons=4, z_dim=128, num_bins=32,
               checkpoint_dir=None,
               sample_dir=None):    
    self.sess = sess   
    self.batch_size = batch_size
    self.lambd = lambd #for the gradient penalization
    self.num_neurons = num_neurons
    self.num_bins = num_bins
    self.output_dim = self.num_neurons*self.num_bins #number of bins per samples
    self.z_dim = z_dim #latent space dimension
    
    #folders
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    
    self.build_model()

  def build_model(self):
    #real samples    
    self.inputs = tf.placeholder(tf.float32, name='real_data', shape=[self.batch_size, self.num_neurons*self.num_bins])
    #fake samples
    self.sample_inputs = self.FCGenerator(self.batch_size)
    
    #discriminator output
    disc_real = self.DCGANDiscriminator(self.inputs)
    disc_fake = self.DCGANDiscriminator(self.sample_inputs)

    #generator and discriminator cost
    self.gen_cost = -tf.reduce_mean(disc_fake)
    self.disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    
    #penalize gradients
    alpha = tf.random_uniform(
        shape=[self.batch_size,1], 
        minval=0.,
        maxval=1.
    )
    differences = self.sample_inputs - self.inputs
    interpolates = self.inputs + (alpha*differences)
    gradients = tf.gradients(self.DCGANDiscriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    self.disc_cost += self.lambd*gradient_penalty

    #this is to save the networks parameters
    self.saver = tf.train.Saver(max_to_keep=1000)

  def train(self, config):
    """Train DCGAN"""
    #define optimizer
    self.g_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2).minimize(self.gen_cost,
                                      var_list=params_with_name('Generator'), colocate_gradients_with_ops=True)
    self.d_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2).minimize(self.disc_cost,
                                       var_list=params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    #initizialize variables              
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
      
    #try to load trained parameters
    self.load()
    
    #get real samples
    if config.dataset=='uniform':
        firing_rates_mat = config.firing_rate+2*(np.random.random(int(self.num_neurons/config.group_size),)-0.5)*config.firing_rate/2    
        correlations_mat = config.correlation+2*(np.random.random(int(self.num_neurons/config.group_size),)-0.5)*config.correlation/2    
        aux = np.arange(int(self.num_neurons/config.group_size))
        activity_peaks = [[x]*config.group_size for x in aux]#np.random.randint(0,high=self.num_bins,size=(1,self.num_neurons)).reshape(self.num_neurons,1)
        activity_peaks = np.asarray(activity_peaks)
        activity_peaks = activity_peaks.flatten()
        activity_peaks = activity_peaks*config.group_size*self.num_bins/self.num_neurons
        activity_peaks = activity_peaks.reshape(self.num_neurons,1)
        self.real_samples = sim_pop_activity.get_samples(num_samples=config.num_samples, num_bins=self.num_bins,\
        num_neurons=self.num_neurons, correlations_mat=correlations_mat, group_size=config.group_size, refr_per=config.ref_period,firing_rates_mat=firing_rates_mat, activity_peaks=activity_peaks)
        #get dev samples
        dev_samples = sim_pop_activity.get_samples(num_samples=int(config.num_samples/4), num_bins=self.num_bins,\
        num_neurons=self.num_neurons, correlations_mat=correlations_mat, group_size=config.group_size, refr_per=config.ref_period,firing_rates_mat=firing_rates_mat, activity_peaks=activity_peaks)
        #save original statistics
        analysis.get_stats(X=self.real_samples, num_neurons=self.num_neurons, num_bins=self.num_bins, folder=self.sample_dir, name='real',firing_rate_mat=firing_rates_mat, correlation_mat=correlations_mat, activity_peaks=activity_peaks)
    elif config.dataset=='retina':
        self.real_samples = retinal_data.get_samples(num_bins=self.num_bins, num_neurons=self.num_neurons, instance=config.data_instance)
        #save original statistics
        analysis.get_stats(X=self.real_samples, num_neurons=self.num_neurons, num_bins=self.num_bins, folder=self.sample_dir, name='real',instance=config.data_instance)
    
    
    
    #start training
    counter_batch = 0
    epoch = 0
    #fitting errors
    f,sbplt = plt.subplots(2,2,figsize=(8, 8),dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    for iteration in range(config.num_iter):
      start_time = time.time()
      # Train generator (only after the critic has been trained, at least once)
      if iteration > 0:
         _ = self.sess.run(self.g_optim)

      # Train critic
      disc_iters = config.critic_iters
      for i in range(disc_iters):
        #get batch and trained critic
        _data = self.real_samples[:,counter_batch*config.batch_size:(counter_batch+1)*config.batch_size].T
        _disc_cost, _ = self.sess.run([self.disc_cost, self.d_optim], feed_dict={self.inputs: _data})
        #if we have reached the end of the real samples set, we start over and increment the number of epochs
        if counter_batch==int(self.real_samples.shape[1]/self.batch_size)-1:
            counter_batch = 0
            epoch += 1
        else:
            counter_batch += 1
    
      #plot the  critics loss and the iteration time
      plot.plot(self.sample_dir,'train disc cost', -_disc_cost)
      plot.plot(self.sample_dir,'time', time.time() - start_time)
    
      if (iteration == 500) or iteration % 20000 == 19999:
        print('epoch ' + str(epoch))
        if config.dataset=='uniform':
            #this is to evaluate whether the discriminator has overfit 
            dev_disc_costs = []
            for ind_dev in range(int(dev_samples.shape[1]/self.batch_size)):
              images = dev_samples[:,ind_dev*config.batch_size:(ind_dev+1)*config.batch_size].T
              _dev_disc_cost = self.sess.run(self.disc_cost, feed_dict={self.inputs: images}) 
              dev_disc_costs.append(_dev_disc_cost)
            #plot the dev loss  
            plot.plot(self.sample_dir,'dev disc cost', -np.mean(dev_disc_costs))
        
        #save the network parameters
        self.save(iteration)
        
        #get simulated samples, calculate their statistics and compare them with the original ones
        fake_samples = self.get_samples(num_samples=2**13)
        fake_samples = fake_samples.eval(session=self.sess)
        fake_samples = self.binarize(samples=fake_samples)    
        acf_error, mean_error, corr_error, time_course_error,_ = analysis.get_stats(X=fake_samples.T, num_neurons=config.num_neurons,\
            num_bins=config.num_bins, folder=config.sample_dir, name='fake'+str(iteration), critic_cost=-_disc_cost,instance=config.data_instance) 
        #plot the fitting errors
        sbplt[0][0].plot(iteration,mean_error,'+b')
        sbplt[0][0].set_title('spk-count mean error')
        sbplt[0][0].set_xlabel('iterations')
        sbplt[0][0].set_ylabel('L1 error')
        sbplt[0][1].plot(iteration,time_course_error,'+b')
        sbplt[0][1].set_title('time course error')
        sbplt[0][1].set_xlabel('iterations')
        sbplt[0][1].set_ylabel('L1 error')
        sbplt[1][0].plot(iteration,acf_error,'+b')
        sbplt[1][0].set_title('AC error')
        sbplt[1][0].set_xlabel('iterations')
        sbplt[1][0].set_ylabel('L1 error')
        sbplt[1][1].plot(iteration,corr_error,'+b')
        sbplt[1][1].set_title('corr error')
        sbplt[1][1].set_xlabel('iterations')
        sbplt[1][1].set_ylabel('L1 error')
        f.savefig(self.sample_dir+'fitting_errors.svg',dpi=600, bbox_inches='tight')
        plt.close(f)
        plot.flush(self.sample_dir)
    
      plot.tick()        
        
      
       
          
  # Discriminator
  def DCGANDiscriminator(self, inputs):
    kernel_width = 2 # in the time dimension
    num_features = self.num_neurons
    #neurons are treated as different channels
    output = tf.reshape(inputs, [-1, self.num_neurons, 1, self.num_bins])
    conv2d_II.set_weights_stdev(0.02)
    deconv2d.set_weights_stdev(0.02)
    linear.set_weights_stdev(0.02)
    print((output.get_shape()))
    print('0. -------------------------------')
    output = conv2d_II.Conv2D('Discriminator.1', self.num_neurons, 2*num_features, 1, kernel_width, output, stride=1)
    output = act_funct.LeakyReLU(output)
    print((output.get_shape()))
    print('1. -------------------------------')
    output = conv2d_II.Conv2D('Discriminator.2', 2*num_features, 4*num_features, 1, kernel_width, output, stride=1)
    output = act_funct.LeakyReLU(output)
    print((output.get_shape()))
    print('2. -------------------------------')
    output = conv2d_II.Conv2D('Discriminator.3', 4*num_features, 8*num_features, 1, kernel_width, output, stride=1)
    output = act_funct.LeakyReLU(output)
    print((output.get_shape()))
    print('3. -------------------------------')
    output = conv2d_II.Conv2D('Discriminator.4', 8*num_features, 16*num_features, 1, kernel_width, output, stride=1)
    output = act_funct.LeakyReLU(output)
    print((output.get_shape()))
    print('4. -------------------------------')
    output = tf.reshape(output, [-1, 16*num_features*self.num_bins])
    output = linear.Linear('Discriminator.Output', 16*num_features*self.num_bins, 1, output)

    conv2d_II.unset_weights_stdev()
    deconv2d.unset_weights_stdev()
    linear.unset_weights_stdev()

    return tf.reshape(output, [-1])
    
  #Generator
  def FCGenerator(self, n_samples, noise=None, FC_DIM=512):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = act_funct.ReLULayer('Generator.1', 128, FC_DIM, noise)
    output = act_funct.ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
    output = act_funct.ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
    output = act_funct.ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
    output = linear.Linear('Generator.Out', FC_DIM, self.output_dim, output)
    
    output = tf.nn.sigmoid(output)
    
    return output
 
  def binarize(self, samples, threshold=None):
    '''
    Returns binarized samples by thresholding with `threshold`. If `threshold` is `None` then the
    elements of `samples` are used as probabilities for drawing Bernoulli variates.
    '''
    if threshold is not None:
      binarized_samples = samples > threshold
    else:
      #use samples as probabilities for drawing Bernoulli random variates
      binarized_samples = samples > np.random.random(samples.shape)
    return binarized_samples.astype(float)  
  
  #draw samples from the generator
  def get_samples(self, num_samples=2**13): 
    noise = tf.constant(np.random.normal(size=(num_samples, 128)).astype('float32'))
    fake_samples = self.FCGenerator(num_samples, noise=noise)
    return fake_samples  
  
  #this is to save the network parameters  
  def save(self, step=0):
    model_name = "WGAN.model"
    self.saver.save(self.sess,os.path.join(self.checkpoint_dir, model_name),global_step=step)
    
  #this is to load an existing model
  def load(self, training_stage=''):
    print(" [*] Reading checkpoints...")
    #checkpoint_dir = os.path.join(checkpoint_dir, FOLDER)
    ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if training_stage=='':
          ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      else:
          #here we select a particular checkpoint by using ckpt.all_model_checkpoint_paths
          index = ckpt.all_model_checkpoint_paths[0].find('WGAN.model')
          index = ckpt.all_model_checkpoint_paths[0].find('-',index)
          for ind_ckpt in range(len(ckpt.all_model_checkpoint_paths)):
              counter = ckpt.all_model_checkpoint_paths[ind_ckpt][index+1:]
              if counter==training_stage:
                  ckpt_name = os.path.basename(ckpt.all_model_checkpoint_paths[ind_ckpt])
                  break
    
      self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False          

 
    
 
    
 