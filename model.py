# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:25:43 2017

@author: manuel
"""

from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from six.moves import xrange
import matplotlib
matplotlib.use('Agg')
from functools import wraps
import sys
sys.path.append(os.getcwd())
from tflib import plot, save_images, sim_pop_activity, params_with_name, analysis
from tflib.ops import linear, act_funct
from tensorflow.python.framework import ops as options


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

class WGAN(object):
  def __init__(self, sess, batch_size=64, lambd=10,
               num_neurons=4, z_dim=128, num_bins=32,
               checkpoint_dir=None,
               sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      kernel_n: (optional) number of minibatch discrimination kernels. [20] Corresponds to 'B' in Salimans2016, where B=100.
      kernel_d: (optional) dimensionality of minibatch discrimination kernels. [20] Corresponds to 'C' in Salimans2016, where C=50.
    """
    self.sess = sess
    self.is_grayscale = True
    
    self.batch_size = batch_size
    self.lambd = lambd
    #dimensions' sizes
    self.num_neurons = num_neurons
    self.num_bins = num_bins
    self.output_dim = self.num_neurons*self.num_bins
    self.z_dim = z_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    #real samples    
    self.inputs = tf.placeholder(tf.float32, name='real_data', shape=[self.batch_size, self.num_neurons*self.num_bins])
    #fake samples
    self.sample_inputs = self.FCGenerator(self.batch_size)
    
    #discriminator output
    disc_real = self.FCDiscriminator(self.inputs)
    disc_fake = self.FCDiscriminator(self.sample_inputs)

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
    gradients = tf.gradients(self.FCDiscriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    self.disc_cost += self.lambd*gradient_penalty

       
    #get all variables
    t_vars = tf.trainable_variables()
    #keep D and G variables
    self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
    self.g_vars = [var for var in t_vars if 'generator' in var.name]

    #save training
    self.saver = tf.train.Saver(max_to_keep=1000)

  def train(self, config):
    """Train DCGAN"""
    #define optimizer
    #optimizer
    self.g_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2).minimize(self.gen_cost,
                                      var_list=params_with_name('Generator'), colocate_gradients_with_ops=True)
    self.d_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2).minimize(self.disc_cost,
                                       var_list=params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    
    #initizialize variables              
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

      
   
    # For generating samples
    fixed_noise = tf.constant(np.random.normal(size=(self.batch_size, 128)).astype('float32'))
    self.all_fixed_noise_samples = self.FCGenerator(self.batch_size, noise=fixed_noise)     
    
    
    #try to load trained parameters
    self.load()
    # Dataset iterator
    firing_rates_mat = config.firing_rate+2*(np.random.random(int(self.num_neurons/config.group_size),)-0.5)*config.firing_rate/2
    self.train_gen, self.dev_gen = sim_pop_activity.load(num_samples=config.num_samples, batch_size=self.batch_size, dim=self.num_bins,\
    num_neurons=self.num_neurons, corr=config.correlation, group_size=config.group_size, refr_per=config.ref_period,firing_rates_mat=firing_rates_mat)
    
    #compute pop activity statistics
    real_samples = sim_pop_activity.get_samples(num_samples=config.num_samples, num_bins=self.num_bins,\
    num_neurons=self.num_neurons, correlation=config.correlation, group_size=config.group_size, refr_per=config.ref_period,firing_rates_mat=firing_rates_mat)
    analysis.get_stats(X=real_samples, num_neurons=self.num_neurons, folder=self.sample_dir, name='real')
    
    
    # Save a batch of ground-truth samples
    _x = next(self.inf_train_gen())
    _x_r = self.sess.run(self.inputs, feed_dict={self.inputs: _x})
    _x_r = self.binarize(_x_r)
    save_images.save_images(_x_r.reshape((self.batch_size, self.num_neurons, self.num_bins)), self.sample_dir+'samples_groundtruth.png')

  
    #start training
    start_time = time.time()
    gen = self.inf_train_gen()
    for iteration in xrange(config.num_iter):
      # Train generator
      if iteration > 0:
         _ = self.sess.run(self.g_optim)

      # Train critic
      disc_iters = config.critic_iters
      for i in range(disc_iters):
        _data = next(gen)
        _disc_cost, _ = self.sess.run([self.disc_cost, self.d_optim], feed_dict={self.inputs: _data})
          
    
      plot.plot(self.sample_dir,'train disc cost', _disc_cost)
      plot.plot(self.sample_dir,'time', time.time() - start_time)
    
      if (iteration < 5) or iteration % 200 == 199:
        dev_disc_costs = []
        for (images,) in self.dev_gen():
          _dev_disc_cost = self.sess.run(self.disc_cost, feed_dict={self.inputs: images}) 
          dev_disc_costs.append(_dev_disc_cost)
        plot.plot(self.sample_dir,'dev disc cost', np.mean(dev_disc_costs))
        self.save(iteration)
            
            
      if (iteration < 5) or (iteration % 200 == 199):
        plot.flush(self.sample_dir)
    
      plot.tick()        
        
          
          
  # Discriminator
  def FCDiscriminator(self,inputs, FC_DIM=512, n_layers=3):
    output = act_funct.LeakyReLULayer('Discriminator.Input', self.output_dim, FC_DIM, inputs)
    for i in range(n_layers):
        output = act_funct.LeakyReLULayer('Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
    output = linear.Linear('Discriminator.Out', FC_DIM, 1, output)

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
  
  
  def get_samples(self, num_samples=2**13):  
    noise = tf.constant(np.random.normal(size=(num_samples, 128)).astype('float32'))
    fake_samples = self.FCGenerator(num_samples, noise=noise)
    return fake_samples  
  
  @property
  def model_dir(self):
    return "{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height)
      
  def save(self, step):
    model_name = "WGAN.model"
    self.saver.save(self.sess,os.path.join(self.checkpoint_dir, model_name),global_step=step)
    
  #this is to load an existing model
  def load(self, training_stage=''):
    print(" [*] Reading checkpoints...")
    #checkpoint_dir = os.path.join(checkpoint_dir, FOLDER)
    ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if training_stage=='':
          #it should be possible to select a particular checkpoint by using ckpt.all_model_checkpoint_paths
          ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      else:
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

  def generate_image(self,iteration):
    samples = self.sess.run(self.all_fixed_noise_samples)
    samples = self.binarize(samples)     
    save_images.save_images(samples.reshape((self.batch_size, self.num_neurons, self.num_bins)), self.sample_dir+'samples_{}.png'.format(iteration))
    
  def inf_train_gen(self):
    while True:
      for (images,) in self.train_gen():
        yield images  
    
 