#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:56:13 2017

@author: manuel
"""

import tflib as lib
import time
import numpy as np
import tensorflow as tf

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def Conv2D(name, input_dim, output_dim, filter_size1, filter_size2, inputs, he_init=True, stride=1, save_filter=False):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, height, width)
    """
    with tf.name_scope(name):

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')


        
        if _weights_stdev is not None:
            filter_values = uniform(_weights_stdev,(filter_size1, filter_size2, input_dim, output_dim))
        else:
            fan_in = input_dim * filter_size1*filter_size2
            fan_out = output_dim * filter_size1*filter_size2 / stride

            if he_init:
                filters_stdev = np.sqrt(4./(fan_in+fan_out))
            else: # Normalized init (Glorot & Bengio)
                filters_stdev = np.sqrt(2./(fan_in+fan_out))

            filter_values = uniform(filters_stdev,(filter_size1, filter_size2, input_dim, output_dim))
        
     
        filters = lib.param(name+'.Filters', filter_values)
       
        result = tf.nn.conv2d(
            input=inputs, 
            filter=filters, 
            strides=[1, 1, stride, stride],
            padding='VALID',
            data_format='NCHW'
        )

        
        _biases = lib.param(
            name+'.Biases',
            np.zeros(output_dim, dtype='float32')
        )

        result = tf.nn.bias_add(result, _biases, data_format='NCHW')

        if save_filter:
            return result, filters
        else:
            return result
        
        
        
if __name__ == '__main__':
    
    set_weights_stdev(0.02)
    start = time.time()
    num_features = 64
    kernel_width = 5
    inputs = np.random.random(size=(64,1, num_features,16)).astype('float32')
    output = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, kernel_width-1]], "CONSTANT")
    
    aux = Conv2D('test', 1, 128, num_features, kernel_width, output, stride=2)
    print(aux.get_shape())
    aux = tf.transpose(aux, [0, 2, 1, 3])
    print(aux.get_shape())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for ind in range(10000):
        test2 = sess.run(aux)
    print(time.time()-start)
    print(test2.shape)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    