# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:30:39 2017

@author: manuel
"""
import tensorflow as tf
from tflib.ops import linear
def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLU(x):
    return tf.nn.relu(x)

def ReLULayer(name, n_in, n_out, inputs):
    output = linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)
