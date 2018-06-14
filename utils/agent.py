#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:48:51 2018

@author: thinkpad
"""

# =============================================================================
#       Keras secondary functions
#==============================================================================

import keras.backend as K
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