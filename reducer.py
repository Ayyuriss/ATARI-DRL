#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:03:24 2018

@author: thinkpad
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:53:38 2018

@author: thinkpad
"""

import keras.backend as K

import tensorflow as tf
import patchlayer
import keras

class ReductionLayer(keras.models.Layer):
    
    def __init__(self, patch_size, dictionary_size,alpha,**kwargs):
        
        super(ReductionLayer, self).__init__(**kwargs)
        
        self.dict_size = dictionary_size
    
        self.patch_layer = patchlayer.PatchLayer(patch_size)
                
        self.alpha = alpha
    def build(self, input_shape):
        
        self.patch_layer.build(input_shape)
        
        self.input_shape_t = self.patch_layer.compute_output_shape(input_shape)
        
        self.dim = self.input_shape_t[-1]
        
        self.filters = self.dict_size
        
        self.strides = (1,self.dim)
        

        self.kernel_shape = (1,self.dim,self.dict_size)
        
        self.D = K.random_normal_variable((self.dim,self.dict_size),mean=0,scale = 1/min([self.dim,self.dict_size]))
        
        self.D_ols = tf.matmul(pinv(tf.matmul(self.D,self.D,transpose_a=True)+self.alpha*tf.eye(self.dict_size)),
                               self.D,transpose_b=True)
        self.kernel = K.reshape(self.D_ols, self.kernel_shape)
        #self.add_weight(shape=self.kernel_shape,
        #                              initializer='glorot_uniform',
        #                              name='kernel')
        self.D_kernel = K.reshape(tf.matmul(self.D,self.D_ols),(1,self.dim,self.dim))
    
    def call(self,inputs):
        
        
        beta = K.conv1d(self.patch_layer(inputs),
                self.D_kernel,
                strides=1,
                padding='valid',
                data_format='channels_last',
                dilation_rate=1)
        self.loss = K.mean(K.pow(beta-self.patch_layer(inputs),2))
        return K.conv1d(self.patch_layer(inputs),
                self.kernel,
                strides=1,
                padding='valid',
                data_format='channels_last',
                dilation_rate=1)
        
    def ReductionLoss(self,a,b):
        return K.mean(K.pow(a - b,2)) + 0.01*self.loss
    
    def set_D(self,D):
        K.set_value(self.D_ridge,D)
        
    def compute_output_shape(self, input_shape):
        return self.patch_layer.compute_output_shape(input_shape)[:2] + (self.dict_size,)

    def get_config(self):
        config = {
            'rank': 1,
            'filters': self.dict_size,
            'kernel_size': self.kernel_shape,
            'strides': self.strides,
            'padding': 'valid',
            'data_format': 'channels_last',
            'activation': 'linear',
            'kernel_initializer': 'personal'
        }
        base_config = super(ReductionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def pinv(A,reltol=1e-6):
    s, u, v = tf.svd(A,full_matrices=True)

    # Invert s, clear entries lower than reltol*s[0].
    atol = tf.reduce_max(s) * reltol
    s = tf.boolean_mask(s, s > atol)
    s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.shape(A)[0] - tf.size(s)])], 0))
    
    # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
    return tf.matmul(v, tf.matmul(s_inv, u,transpose_b=True),transpose_a=True)

def outer(a,b):
    return tf.einsum('ijk,ijm->ijkm',a,b)

