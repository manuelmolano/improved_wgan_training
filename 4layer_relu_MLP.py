# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 18:45:10 2017

@author: manuel
"""

import os, sys
sys.path.append(os.getcwd())

import time
#import functools

import numpy as np
import tensorflow as tf
#import sklearn.datasets
from tflib import plot, save_images, sim_pop_activity, print_model_settings, params_with_name
from tflib.ops import linear
from tensorflow.python.framework import ops
ops.reset_default_graph()
# Download 64x64 ImageNet at http://image-net.org/small/download.php and
# fill in the path to the extracted files here!

NUM_BINS = 32 # Model dimensionality
NUM_NEURONS = 4 
GROUP_SIZE = 2
CORR = 0.5
REFR_PER = 2
NUM_SAMPLES = 2**13
CRITIC_ITERS = 5 # How many iterations to train the critic for
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
ITERS = 200000 # How many iterations to train for
LAMBDA = 9.9 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = NUM_NEURONS*NUM_BINS # Number of pixels in each image
#FOLDER = FLAGS.checkpoint_dir + '_dataset_' + FLAGS.dataset + '_num_classes_' + str(FLAGS.num_classes) + '_propClasses_' + FLAGS.classes_proportion + \
#      '_num_samples_' + str(FLAGS.num_samples) + '_num_bins_' + str(FLAGS.NUM_BINS) + '_ref_period_' + str(FLAGS.ref_period) + '_firing_rate_' + str(FLAGS.firing_rate) + '_iteration_' + FLAGS.iteration
FOLDER = '/home/manuel/improved_wgan_training/samples/samples'+ '_num_bins_' + str(NUM_BINS) +'_num_neurons_' + str(NUM_NEURONS) +'_group_size_' + str(GROUP_SIZE) +\
'_corr_' + str(CORR) +'_refrPer_' + str(REFR_PER) +'_num_samples_' + str(NUM_SAMPLES) +'_critic_iters_' + str(CRITIC_ITERS) +'_batch_size_' + str(BATCH_SIZE) +\
'_lambda_' + str(LAMBDA) + '/'
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)
CHECKPOINT_DIR = FOLDER+'checkpoint/'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

print_model_settings(locals().copy())

 


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)



def binarize(samples, threshold=None):
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


#Generator
def FCGenerator(n_samples, noise=None, FC_DIM=512):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = ReLULayer('Generator.1', 128, FC_DIM, noise)
    output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
    output = linear.Linear('Generator.Out', FC_DIM, OUTPUT_DIM, output)

    output = tf.tanh(output)
    
    return output

# Discriminator
def FCDiscriminator(inputs, FC_DIM=512, n_layers=3):
    output = LeakyReLULayer('Discriminator.Input', OUTPUT_DIM, FC_DIM, inputs)
    for i in range(n_layers):
        output = LeakyReLULayer('Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
    output = linear.Linear('Discriminator.Out', FC_DIM, 1, output)

    return tf.reshape(output, [-1])

    

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:    
    #data
    real_data = tf.placeholder(tf.float32, name='real_data', shape=[BATCH_SIZE, NUM_NEURONS*NUM_BINS])
    fake_data = FCGenerator(BATCH_SIZE)
    
    #discriminator output
    disc_real = FCDiscriminator(real_data)
    disc_fake = FCDiscriminator(fake_data)

    #generator and discriminator cost
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    
    #penalize gradients
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(FCDiscriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty


    #optimizer
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(gen_cost,
                                      var_list=params_with_name('Generator'), colocate_gradients_with_ops=True)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(disc_cost,
                                       var_list=params_with_name('Discriminator.'), colocate_gradients_with_ops=True)


    
   
    #this is to save the model during training    
    SAVER = tf.train.Saver(max_to_keep=1000)
    def save(step):
        model_name = "WGAN.model"
        SAVER.save(session,
                os.path.join(CHECKPOINT_DIR, model_name),
                global_step=step)
    
    #this is to load an existing model
    def load(training_stage=''):
        print(" [*] Reading checkpoints...")
        #checkpoint_dir = os.path.join(checkpoint_dir, FOLDER)
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
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
        
          SAVER.restore(session, os.path.join(CHECKPOINT_DIR, ckpt_name))
          print(" [*] Success to read {}".format(ckpt_name))
          return True
        else:
          print(" [*] Failed to find a checkpoint")
          return False          
    
    # For generating samples
    fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    n_samples = BATCH_SIZE
    all_fixed_noise_samples = FCGenerator(n_samples, noise=fixed_noise)     
    def generate_image(iteration):
        samples = session.run(all_fixed_noise_samples)
        samples = binarize(samples)     
        #samples = ((samples+1.)*(255.99/2)).astype('int32')
        save_images.save_images(samples.reshape((BATCH_SIZE, NUM_NEURONS, NUM_BINS)), FOLDER+'samples_{}.png'.format(iteration))


    # Dataset iterator
    train_gen, dev_gen = sim_pop_activity.load(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, dim=NUM_BINS,\
    num_neurons=NUM_NEURONS, corr=CORR, group_size=GROUP_SIZE, refr_per=REFR_PER)

    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images

    # Save a batch of ground-truth samples
    _x = next(inf_train_gen())
    _x_r = session.run(real_data, feed_dict={real_data: _x})
    _x_r = binarize(_x_r)
    #_x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
    save_images.save_images(_x_r.reshape((BATCH_SIZE, NUM_NEURONS, NUM_BINS)), FOLDER+'samples_groundtruth.png')

    #try to load trained parameters
    load()

    # Train loop
    session.run(tf.global_variables_initializer())
    gen = inf_train_gen()
    start_time = time.time()
    for iteration in range(ITERS):
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data = next(gen)
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data: _data})
          

        plot.plot(FOLDER,'train disc cost', _disc_cost)
        plot.plot(FOLDER,'time', time.time() - start_time)

        if (iteration < 5) or iteration % 200 == 199:
            t = time.time()
            dev_disc_costs = []
            for (images,) in dev_gen():
                _dev_disc_cost = session.run(disc_cost, feed_dict={real_data: images}) 
                dev_disc_costs.append(_dev_disc_cost)
            plot.plot(FOLDER,'dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration)
            save(iteration)
            
            
        if (iteration < 5) or (iteration % 200 == 199):
            plot.flush(FOLDER)

        plot.tick()

    fixed_noise = tf.constant(np.random.normal(size=(1000, 128)).astype('float32'))
    n_samples = BATCH_SIZE
    all_fixed_noise_samples = FCGenerator(n_samples, noise=fixed_noise)     
    samples = session.run(all_fixed_noise_samples)
    samples = binarize(samples)     