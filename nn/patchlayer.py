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

    
class PatchExtracter(Layer):
    """
    """

    def __init__(self, rank,
                 patch_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(PatchExtracter, self).__init__(**kwargs)
        self.rank = rank
        self.patch_size = conv_utils.normalize_tuple(patch_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': 2,
            'filters': self.dictionary_size,
            'kernel_size': self.patch_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate
            }
        base_config = super(PatchExtracter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
