#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:16:59 2018

@author: thinkpad
"""

import torch

class BaseDQN(object):
    
    def __init__(self, states_dim, actions_n):
        
        self.states_dim = states_dim
        self.actions_n = actions_n
        self.setup_model(states_dim, actions_n)

    def setup_model(self):
        pass
    
    def decide(self,state):
        
        return self.predict(state)
    
    def reinforce(self,episode):
        pass
    def predict(self,state):
        pass
    
class DQN(BaseDQN):
    
    def __init__(self, states_dim, actions_n):
        
        super(DQN,self).__init__(states_dim, actions_n)

        self.setup_model(states_dim, actions_n)
    
    def setup_model(self):
        pass
    
    def decide(self,state):
        
        return self.predict(state)
    
    def predict(self,state):
        pass