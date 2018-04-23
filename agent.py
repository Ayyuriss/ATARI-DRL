#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:27:48 2018

@author: thinkpad
"""
import numpy as np


class Agent(object):
    
    def __init__(self, model, model_type, epsilon=-1):
        
        self.epsilon = epsilon
        
        assert model_type in ["Q","Policy"]
        
        self.model = model
        self.actions_n = model.actions_n
        
    def act(self,state,train=False):
        if train:
            if np.random.rand() < self.epsilon:
                return np.random.randint(0,self.actions_n)
            else:
                return self.learned(state)
        else:
            return self.learned(state)
    
    def learned(self,state):
        return self.model.devide(state)        
    
    def train(self,episode):
        
        self.model.reinforce(episode)
    
