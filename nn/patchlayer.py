#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:57:05 2018

@author: thinkpad
"""

from keras import initializers,activations
from keras.utils import conv_utils
from keras.engine import InputSpec,Layer

import numpy as np
import keras

import tensorflow as tf

class PatchLayer(Layer):
    
    def __init__(self,patch_size,
                 strides=None,
                 padding='VALID',
                 data_format=None,
                 **kwargs):
        super(PatchLayer, self).__init__(**kwargs)
        
        self.patch_size = patch_size
        self.padding = padding
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.strides = strides
        self.rank = 2
        self.input_spec = InputSpec(ndim=self.rank + 2)
        if (type(self.patch_size)==tuple):
            self.stepr,self.stepc = self.patch_size[1]
        else:
            self.stepr = self.stepc = self.patch_size
        if strides is None:
            self.strides = (self.stepr,self.stepc)
    
    def build(self, input_shape):
        
        if self.data_format == 'channels_first':
            channel_axis = 1
            self.r,self.c = tuple(np.array(input_shape[2:])//self.patch_size)
        else:
            channel_axis = -1
            self.r,self.c = tuple(np.array(input_shape[1:-1])//self.patch_size)
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        
        self.filters = input_shape[channel_axis]
        self.strides = (1,)+self.strides+(1,)
        self.kernel = self.add_weight(shape=(1,1,1,1),#,kernel_shape,
                                      name='kernel',
                                      trainable=False,
                                      initializer='zeros')
        
        self.r = input_shape[1]//self.stepr
        self.c = input_shape[2]//self.stepc
        
        self.kernel_shape = (1,self.stepr,self.stepc,1)        
        
        self.final = keras.layers.Reshape((self.r*self.c,-1))
        self.built = True

    def call(self, inputs):
                
        return self.final(tf.extract_image_patches(inputs,self.kernel_shape,self.strides,(1,1,1,1),self.padding,name='Patches'))

    def compute_output_shape(self, input_shape):
        
            return (input_shape[0],) + (self.c*self.r,) + (self.stepc*self.stepr*self.filters,)

    

