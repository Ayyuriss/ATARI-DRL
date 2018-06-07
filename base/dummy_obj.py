#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:59:45 2018

@author: thinkpad
"""
import numpy as np
import collections

class Policy(object):
    
    def __init__(self, eps, actions_n):
        
        self.eps = eps
        self.n = actions_n
    
    def reset(self):
        self.eps = 1
        
    def decrement(self,tol):
        self.eps = max(self.eps-tol,0.1)
    
    def choose(self, weights):
        #w = self.eps/self.n+ (1-self.eps)*weights/np.sum(weights)
        #return np.random.choice(np.arange(self.n),p=w)
        
        if np.random.rand()<self.eps:
            return np.random.randint(self.n)
        return np.argmax(weights)
    
class ReplayMemory(object):
    
    def __init__(self, memory_size, keys):
        
        self.keys = keys
        self.memory_size = memory_size
        self.forget()
    
    def record(self,episode):
        for k in episode.keys():
            for v in episode[k]:
                self.memory[k].append(v)
                
    def forget(self):
        self.memory = {s : collections.deque([],self.memory_size) for s in self.keys}
        
    def empty_episode(self):
        return   {s :[] for s in self.keys}
    
    def sample(self,size):
        
        sample = self.empty_episode()
        sample_idx = np.random.choice(np.arange(self.size),size,replace=False)
        for k in self.memory.keys():
            sample[k] = np.concatenate([[self.memory[k][i]] for i in sample_idx])
            
        return sample
    
    @property
    def size(self):
        return len(self.memory[self.keys[0]])