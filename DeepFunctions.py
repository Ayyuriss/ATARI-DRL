# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:07:46 2018

@author: gamer
"""

import NeuralNets

class BaseDeep(object):

    def __init__(self,states_dim, actions_n,stochastic =True, network_type='FC'):
        
        self.states_dim = states_dim
        self.actions_n = actions_n
        self.network_type = network_type
        self.stochastic = stochastic
        self.setup_model()
        
    def setup_model(self):
        raise (NotImplementedError, self.network_type)
        
    def evaluate(self,state):
        return self.net.predict(state)
    
    def reinforce(self,episode):
        raise NotImplementedError

    @property
    def variables(self):
        return self.net.trainable_variables
    @property
    def input(self):
        return self.net.input
        
    @property
    def output(self):
        return self.net.output
    
class DeepPolicy(BaseDeep):
    def setup_model(self):
        if self.network_type in ['FC','CONV']:
            if self.network_type =='FC':
                self.net = NeuralNets.P_FCNet(self.states_dim, self.actions_n)
            elif self.network_type =='CONV':
                self.net = NeuralNets.P_ConvNet(self.states_dim, self.actions_n)
        else:
            raise (NotImplementedError, self.network_type)
   
        
class DeepQ(BaseDeep):
    def setup_model(self):
        if self.network_type =='FC':
            self.net = NeuralNets.Q_FCNet(self.states_dim, self.actions_n)
        elif self.network_type =='CONV':
            self.net = NeuralNets.Q_ConvNet(self.states_dim, self.actions_n)
        else:
            raise (NotImplementedError, self.network_type)
            
