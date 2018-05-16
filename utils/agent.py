#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:48:51 2018

@author: thinkpad
"""

# =============================================================================
# =============================================================================
#       Keras secondary functions
#==============================================================================


from keras import backend as K
import numpy as np

def slice_2d(x, inds1):
    
    inds1 = K.cast(inds1, 'int64')
    shape = K.cast(K.shape(x), 'int64')
    ncols = shape[1]
    nrows = shape[0]
    inds0 = K.arange(nrows)
    x_flat = K.reshape(x, [-1])
    return K.gather(x_flat, inds0 * ncols + inds1)

def likelihood(a, prob):
    return slice_2d(prob,a)

def loglikelihood(a, prob):
    return K.log(likelihood(a, prob))

def kl(prob0, prob1):
    return K.sum( prob0*K.log(prob0/prob1), axis=1)

def entropy(prob0):
    return -K.sum( prob0*K.log(prob0), axis=1)

class Flattener(object):
    def __init__(self, variables):
        self.variables = variables        
        assert type(variables) == list
        self.shapes = list(map(K.int_shape, variables))
        self.get_op = K.concatenate([K.flatten(x) for x in self.variables])
        start = 0
        self.idx = []
        for s in self.shapes:
            size = np.prod(s)
            self.idx.append((start,start + size))
            start += size
        self.total_size = start
        
    def get(self):
        return K.get_value(self.get_op)
    
    def flatgrad(self,loss):
        grads = K.gradients(loss, self.variables)
        return K.concatenate([K.flatten(g) for g in grads])
    
    def set(self,theta):
        assert theta.shape == (self.total_size,)
        theta = np.array(theta,dtype='float32')
        
        for i,v in enumerate(self.variables):
            
            K.set_value(v, np.reshape(
                    theta[self.idx[i][0]:self.idx[i][1]], self.shapes[i]))
            
def choice_weighted(pi):
#    np.random.seed(np.random.randint(0,2**10))
    #print(pi.shape)
    return np.random.choice(np.arange(len(pi)), 1, p=pi)[0]                

def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

import keras
def evaluate_filters(net, layer_n, inputs):
    model = keras.models.Model(inputs=[net.input], outputs=net.layers[layer_n].output)
    return model.predict(inputs.reshape((1,)+inputs.shape))