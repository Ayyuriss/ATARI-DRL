#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:04:10 2018

@author: thinkpad
"""
import sys,os
import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
from base.agent import Agent
from base import dummy_obj
from nn import DeepFunctions
from utils.console import Progbar


class DQN(Agent):
    
    def __init__(self, env, neural_type, gamma, memory_max):
        
        model = DeepFunctions.DeepQ(env.observation_space, env.action_space, neural_type)
        
        super(DQN,self).__init__(model)
        self.discount = gamma
        self.env = env
        
        self.memory_max = memory_max
        self.progbar = Progbar(self.memory_max)
        self.memory = dummy_obj.Memory(self.memory_max,["t","state","action","reward","next_state","terminated"])
        self.eps = 0
    def act(self,state):
        
        if np.random.rand()<self.eps:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))
    
    def train(self,num_steps, epochs, batch_size=64):
        
        self.set_eps(1)
        
        for _ in range(epochs):
            rollout = self.rollout(num_steps)
            actions = rollout["action"]
            rewards = rollout["reward"]
            not_final = np.logical_not(rollout["terminated"])
            
            old_theta = self.Flaten.get()
            old_q = self.model.predict(rollout["state"])
            
            target_q = self.model.predict(rollout["next_state"])
            max_Q_prim = np.max(target_q,axis=1)
        
            for i in range(len(actions)):
                target_q[i,actions[i]] = rewards[i] + not_final[i]* self.discount*max_Q_prim[i]
                        
            self.model.learn(rollout["state"],target_q, batch_size)
                
            new_theta = self.Flaten.get()
            new_q = self.model.predict(rollout["state"])
            self.log("Theta MSE",np.linalg.norm(new_theta-old_theta))
            self.log("Q MSE",np.linalg.norm(new_q-old_q))
    
            self.log("Average reward",np.mean(rewards))
            self.log("Max reward",np.max(rewards))
            self.log("Min reward",np.min(rewards))
            self.log("Epsilon",self.eps)
            self.print_log()
    
    def set_eps(self,x):
        self.eps = max(x,0.1)
        
    def rollout(self,num_steps):

        collected = 0
        self.progbar.__init__(num_steps)

        
        while collected < num_steps:
            collected += self.get_episode(num_steps-collected,1/self.memory_max)
        roll = self.memory.random_sample(num_steps)        
        return roll
        
    def get_episode(self, length, eps):
        
        state = self.env.reset()        
        
        episode = self.memory.empty_episode()
        
        i = 0
        
        while i < length:
            
            self.progbar.add(1)

            # save current state

            episode["state"].append(state)
            
            # act
            action = self.act(state)   
            state, rew, done = self.env.step(action)

            episode["next_state"].append(state)            
            episode["t"].append(i)            
            episode["action"].append(action)
            episode["reward"].append(rew)        
            episode["terminated"].append(done)
            
            
            self.set_eps(self.eps-eps)
            i += 1
            
            if done:
                state = self.env.reset()
                break

        # record the episodes
        self.memory.record(episode)
        
        del(episode)
        
        return i

    def play(self,name='play'):
        eps = self.eps
        self.set_eps(0)
        state = self.env.reset()
        done = False
        
        while not done:
            
            action = self.act(state)
            
            state, _, done = self.env.step(action)
        
        self.env.draw(name)
        self.set_eps(eps)


