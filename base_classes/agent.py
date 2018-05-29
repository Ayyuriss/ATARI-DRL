#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:27:48 2018

@author: thinkpad
"""
import numpy as np
import DeepFunctions
import utils.agent as utils
import collections
import dummy_obj

class Agent(object):
    
    def __init__(self, model,epsilon=0):
        
        self.model = model
        
        self.actions_n = model.actions_n
        
        self.history = collections.OrderedDict()
        
        self.params = self.model.variables

        self.Flaten = utils.Flattener(self.params)
        
        self.checkpoints = "./checkpoints/"
        
        self.policy = dummy_obj.Policy(epsilon,self.actions_n)
        
    def act(self,state,train=False):
        
        raise NotImplementedError
    
    def reinforce(self,episodes):
        
        raise NotImplementedError
        
    def save(self,name):
        print("Saving %s"%name)
        self.model.save(self.checkpoints+name)
        
    def load(self,name):
        print("Loading %s"%name)
        return self.model.load(self.checkpoints+name)
    def log(self, key,value):
        if key not in self.history.keys():    
            self.history[key] = [value]
        else:
            self.history[key] = np.concatenate([self.history[key],[value]])
    def print_log(self):
        max_l = max(list(map(len,self.history.keys())))
        for k,v in self.history.items():
            print(k+(max_l-len(k))*" ",": %f"%v[-1])


        
    
