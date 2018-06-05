#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:26:44 2018

@author: thinkpad
"""
import numpy as np

class Discrete(object):
    
    def __init__(self,n):        
        self.n = n
        self.shape = (n,)
        self.dtype = np.int64

    def sample(self):
        return np.random.randint(self.n)
    
    def __repr__(self):
        return "Discrete(%d)" % self.n
        
    def __eq__(self,m):
        return self.n ==m

class Box(object):
    def __init__(self,low=None, high = None, shape=None, dtype = np.float32):
        
        self.shape = shape
        self.dtype = dtype
        
        self.low = low + np.zeros(shape)
        self.high = high + np.zeros(shape)
        
    def sample(self):
        np.random.uniform(low = self.low, high = self.high)
    
    def __repr__(self):
        return "Box" +str(self.shape)