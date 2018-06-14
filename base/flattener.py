#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 01:32:48 2018

@author: thinkpad
"""

import keras.backend as K
import numpy as np


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
        
    def get_value(self):
        return K.get_value(self.get_op)

    def set_value(self,theta):
        assert theta.shape == (self.total_size,)
        theta = np.array(theta,dtype='float32')
        
        for i,v in enumerate(self.variables):
            
            K.set_value(v, np.reshape(
                    theta[self.idx[i][0]:self.idx[i][1]], self.shapes[i]))
    def get(self):
        return self.get_op    

    def flatgrad(self,loss):
        grads = K.gradients(loss, self.variables)
        return K.concatenate([K.flatten(g) for g in grads])
    
