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
import sys,os

import keras
import keras.backend as K
import tensorflow as tf
import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))

from utils.console import Progbar
from nn import patchlayer

class ReductionLayer(keras.models.Layer):
    
    def __init__(self, patch_size, dictionary_size,alpha,**kwargs):
        
        super(ReductionLayer, self).__init__(**kwargs)
        
        self.dict_size = dictionary_size
    
        self.patch_layer = patchlayer.PatchLayer(patch_size)
                
        self.alpha = alpha
        
        self.progbar = Progbar(100,stateful_metrics=["loss"])
        
        self.old_D = 0.0
    def build(self, input_shape):
        
        self.patch_layer.build(input_shape)
        
        self.input_shape_t = self.patch_layer.compute_output_shape(input_shape)
        
        self.dim = self.input_shape_t[-1]
        
        self.filters = self.dict_size
        
        self.strides = (1,self.dim)
        

        self.kernel_shape = (1,self.dim,self.dict_size)
        
        self.D0 = K.random_normal_variable((self.dim,self.dict_size),mean=0,scale = 1)

        self.D = tf.matmul(tf.diag(1/tf.norm(self.D0,axis=1)),self.D0)

        self.D_ols = tf.matmul(tf.linalg.inv(tf.matmul(self.D,self.D,transpose_a=True)+self.alpha*tf.eye(self.dict_size)),
                               self.D,transpose_b=True)
        self.kernel = K.reshape(self.D_ols, self.kernel_shape)
        #self.add_weight(shape=self.kernel_shape,
        #                              initializer='glorot_uniform',
        #                              name='kernel')
        self.D_kernel = K.reshape(tf.matmul(self.D,self.D_ols),(1,self.dim,self.dim))
    
    def call(self,inputs):
        
        
        beta = K.conv1d(self.patch_layer(inputs),
                self.kernel,
                strides=1,
                padding='valid',
                data_format='channels_last',
                dilation_rate=1)
        
        return beta
        
    def fit(self,X,Y,batch_size=64):
        print("Fitting the reduction")
        
        n = len(X)
        self.progbar.__init__(n)
        for i in range(0,n,batch_size):
            weights = np.ones(min(n,i+batch_size)-i)
            inputs = X[i:min(i+batch_size,n)]
            targets = Y[i:min(n,i+batch_size)]
            self.fit_op([inputs,targets,weights])
            self.progbar.add(min(batch_size,n-batch_size),
                             values=[('loss',self.loss([inputs,targets,weights])[0])])
    
    def display_update(self):
        res = np.linalg.norm(K.eval(self.D)-self.old_D)
        self.old_D = K.eval(self.D)
        return res
    def set_D(self,D):
        K.set_value(self.D_ridge,D)
        
    def compile(self,model):
        
        self.optimizer = tf.train.RMSPropOptimizer(0.001)
        self.opt = self.optimizer.minimize(model.total_loss,var_list=[self.D0])
        self.fit_op = K.Function([model.input,model.targets[0],model.sample_weights[0]],[self.opt])
        self.loss = K.Function([model.input,model.targets[0],model.sample_weights[0]],[model.total_loss])
        print("Reduction Layer Compiled, batch %d"%self.patch_layer.patch_size,"\n",
              "Output shape:",self.compute_output_shape(self.input_shape_t)) 
        
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

