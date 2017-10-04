#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:12:26 2017

@author: manuel
"""

import tflib as lib

import numpy as np
import tensorflow as tf

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def Deconv2D(name, input_dim, output_dim, filter_size1, filter_size2, inputs, he_init=True):
    """
    inputs: tensor of shape (batch size, height, width, input_dim)
    returns: tensor of shape (batch size, 2*height, 2*width, output_dim)
    """
    with tf.name_scope(name):

      
        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        stride = 2
        fan_in = input_dim * filter_size1*filter_size2 / (stride**2)
        fan_out = output_dim * filter_size1*filter_size2


        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size1, filter_size2, input_dim, output_dim)
            )
        else:
            if he_init:
                filters_stdev = np.sqrt(4./(fan_in+fan_out))
            else: # Normalized init (Glorot & Bengio)
                filters_stdev = np.sqrt(2./(fan_in+fan_out))
            filter_values = uniform(
                filters_stdev,
                (filter_size1, filter_size2, input_dim, output_dim)
            )


        filters = lib.param(name+'.Filters', filter_values)
        inputs = tf.transpose(inputs, [0,2,3,1], name='NCHW_to_NHWC')
        input_shape = tf.shape(inputs)
        
        output_shape = tf.stack([1,  2*input_shape[2]])
        
       
      
        resized_image = tf.image.resize_images(images=inputs, size=output_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)        
       
        result = tf.nn.conv2d(input=resized_image, filter=filters, strides=[1, 1, 1, 1], padding='SAME')

  
        _biases = lib.param(
            name+'.Biases',
            np.zeros(output_dim, dtype='float32')
        )
        result = tf.nn.bias_add(result, _biases)
        result = tf.transpose(result, [0,3,1,2], name='NHWC_to_NCHW')
        return result
