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

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.sim_pop_activity
import tflib.ops.layernorm
import tflib.plot

# Download 64x64 ImageNet at http://image-net.org/small/download.php and
# fill in the path to the extracted files here!
FOLDER = '/home/manuel/improved_wgan_training/samples/sim_pop_activity/'
CHECKPOINT_DIR = FOLDER+'checkpoint/'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
MODE = 'wgan-gp' # dcgan, wgan, wgan-gp, lsgan
DIM = 128 # Model dimensionality
NUM_NEURONS = 32 
GROUP_SIZE = 2
CORR = 0.5
REFR_PER = 2
NUM_SAMPLES = 2**13
CRITIC_ITERS = 5 # How many iterations to train the critic for
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
ITERS = 200000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = NUM_NEURONS*DIM # Number of pixels in each image
saver = tf.train.Saver(max_to_keep=1000)
lib.print_model_settings(locals().copy())

def GeneratorAndDiscriminator():
   
    # 512-dim 4-layer ReLU MLP G
    return FCGenerator, FCDiscriminator

  

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Normalize(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

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
    output = lib.ops.linear.Linear('Generator.Out', FC_DIM, OUTPUT_DIM, output)

    output = tf.tanh(output)
    
    return output

# Discriminator
def FCDiscriminator(inputs, FC_DIM=512, n_layers=3):
    output = LeakyReLULayer('Discriminator.Input', OUTPUT_DIM, FC_DIM, inputs)
    for i in range(n_layers):
        output = LeakyReLULayer('Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Discriminator.Out', FC_DIM, 1, output)

    return tf.reshape(output, [-1])

    

Generator, Discriminator = GeneratorAndDiscriminator()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    #this is to save the model during training    
    def save(step):
        model_name = "WGAN.model"
        saver.save(session,
                os.path.join(CHECKPOINT_DIR, model_name),
                global_step=step)
                
    #data
    real_data = tf.placeholder(tf.float32, name='real_data', shape=[BATCH_SIZE, NUM_NEURONS*DIM])
    fake_data = Generator(BATCH_SIZE)
    
    #discriminator output
    disc_real = Discriminator(real_data)
    disc_fake = Discriminator(fake_data)

    #generator and discriminator cost
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    
    #penalize gradients
    alpha = tf.random_uniform(
        shape=[int(BATCH_SIZE/len(DEVICES)),1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty


    #optimizer
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(gen_cost,
                                      var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(disc_cost,
                                       var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)


    # For generating samples
    fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    n_samples = BATCH_SIZE
    all_fixed_noise_samples = Generator(n_samples, noise=fixed_noise)
   
    def generate_image(iteration):
        samples = session.run(all_fixed_noise_samples)
        samples = binarize(samples)        
        #samples = ((samples+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, NUM_NEURONS, DIM)), FOLDER+'samples_{}.png'.format(iteration))


    # Dataset iterator
    train_gen, dev_gen = lib.sim_pop_activity.load(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, dim=DIM,\
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
    lib.save_images.save_images(_x_r.reshape((BATCH_SIZE, NUM_NEURONS, DIM)), FOLDER+'samples_groundtruth.png')


    # Train loop
    session.run(tf.global_variables_initializer())
    gen = inf_train_gen()
    for iteration in range(ITERS):

        start_time = time.time()

        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data = next(gen)
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data: _data})
          

        lib.plot.plot(FOLDER,'train disc cost', _disc_cost)
        lib.plot.plot(FOLDER,'time', time.time() - start_time)

        if iteration % 200 == 199:
            t = time.time()
            dev_disc_costs = []
            for (images,) in dev_gen():
                _dev_disc_cost = session.run(disc_cost, feed_dict={real_data: images}) 
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(FOLDER,'dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration)
            save(iteration)
            
            
        if (iteration < 5) or (iteration % 200 == 199):
            lib.plot.flush(FOLDER)

        lib.plot.tick()

