# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:07:46 2018

@author: gamer
"""

import torch as trc
import numpy as np




class BasePolicy(object):

    def __init__(self,states_dim, actions_n):
        
        self.states_dim = states_dim
        self.actions_n = actions_n
        self.setup_model()
        
    def setup_model(self):
        
        pass
        
    def decide(self,state):
        return self.predict(state)
    
    def reinforce(self,episode):
        pass
    
    def predict(self,state):
        pass
    
    @property
    def variables(self):
        pass
    
class DeepPolicy(BasePolicy):
    def __init__(self, states_dim, actions_n,NN):
        
        super(DeepPolicy, self).__init__(states_dim, actions_n)
        
    def setup_model(self):
        pass
        
    def predict(self,state):
        
        return self.NN.forward(state)
    
    @property
    def variables(self):
        return self.NN.parameters()
        
