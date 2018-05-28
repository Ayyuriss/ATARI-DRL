#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:04:10 2018

@author: thinkpad
"""

from base_classes.agent import Agent
import DeepFunctions
import numpy as np

class DQN(Agent):
    
    def __init__(self, states_dim, actions_n, neural_type, gamma, epsilon):
        
        model = DeepFunctions.DeepQ(states_dim, actions_n, neural_type)
        super(DQN,self).__init__(model,epsilon)
        self.eps = epsilon
        self.discount = gamma

    def act(self,state):
        
        val = self.model.predict(state)
        #weights = 1.0*(np.argmax(val)==val)
        return self.policy.choose(val)
    
    def reinforce(self,rollout,batch_size=50,epochs=1):

        #t = rollout("t")

        actions = rollout["action"]
        rewards = rollout["reward"]
        not_final = np.logical_not(rollout["terminated"])
        
        old_theta = self.Flaten.get()
        old_q = self.model.predict(rollout["state"])
        
        target_q = self.model.predict(rollout["next_state"])
        max_Q_prim = np.max(target_q,axis=1)
    
        for i in range(len(actions)):
            target_q[i,actions[i]] = rewards[i]
            if not_final[i]:
                target_q[i,actions[i]] += self.discount*max_Q_prim[i]
                    
        for _ in range(epochs):
            self.model.learn(rollout["state"],target_q,batch_size)
            
        new_theta = self.Flaten.get()
        new_q = self.model.predict(rollout["state"])
        self.log("Theta MSE",np.linalg.norm(new_theta-old_theta))
        self.log("Q MSE",np.linalg.norm(new_q-old_q))

        self.log("Average reward",np.mean(rewards))
        self.log("Min reward",np.min(rewards))
        self.log("Average return",np.mean(rollout["return"]))


        self.log("Epsilon",self.eps)
        self.print_log()
        
    def set_epsilon(self,eps):
        self.eps = eps
    