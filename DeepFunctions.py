# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:07:46 2018

@author: gamer
"""

import NeuralNets
import numpy as np


# ================================================================
# Base class for Q and deep Policy
# ================================================================
class BaseDeep(object):

    def __init__(self,states_dim, actions_n, network_type='FC'):
        
        self.states_dim = states_dim
        self.actions_n = actions_n
        self.network_type = network_type
        self.setup_model()
        
    def setup_model(self):
        raise (NotImplementedError, self.network_type)
        
    def predict(self,state):
        
        return self.net.predict(state)
    
    def save(self,name):
        self.net.save(name)
        
    def load(self,name):
        self.net.load(name)
    
    def learn(self,states,target_q,batch_size):
        
        self.net.fit(states,target_q,batch_size)


    @property
    def variables(self):
        return self.net.trainable_variables
    @property
    def input(self):
        return self.net.input
        
    @property
    def output(self):
        return self.net.output

# ================================================================
# Object class for Q and policy
# ================================================================

class DeepPolicy(BaseDeep):
    
    def setup_model(self):
        
        assert self.network_type in ['FC','CNN']
            
        if self.network_type =='FC':
            self.net = NeuralNets.Policy_FCNet(self.states_dim, self.actions_n)
        else:
            self.net = NeuralNets.Policy_CNNet(self.states_dim, self.actions_n)
      
   
        
class DeepQ(BaseDeep):
    
    def setup_model(self):
        
        assert self.network_type in ['FC','CNN']
        
        if self.network_type =='FC':
            self.net = NeuralNets.Q_FCNet(self.states_dim, self.actions_n)
        else:
            self.net = NeuralNets.Q_CNNet(self.states_dim, self.actions_n)
        
        #self.net.zero_initializer()
            

            
# ================================================================
# Value Function for baseline
# ================================================================
class ValueFunction(BaseDeep):

    def setup_model(self):
        self.net = NeuralNets.SingleFCNet(self.states_dim,self.actions_n)                           
        
    def _features(self, episode):
        states = episode["state"].astype('float32')
        states = states.reshape(states.shape[0], -1)
        proba = episode["output"].astype('float32')
        n = len(episode["reward"])
        al = np.arange(n).reshape(-1, 1) / 10.0
        ret = np.concatenate([states, proba, al, np.ones((n, 1))], axis=1)
        return ret

    def fit(self, episodes):
        featmat = np.concatenate([self._features(episode) for episode in episodes])
        returns = np.concatenate([episode["returns"] for episode in episodes])
        self.net.fit(featmat,returns)

    def evaluate(self, episode):
        self.net.predict(self._features(episode))
