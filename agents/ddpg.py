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
from nn import deepfunctions
from utils.console import Progbar
import keras.backend as K

class DDPG(Agent):
    
    deep = deepfunctions.DeepQ
    
    def __init__(self, env, gamma, memory_max, batch_size, train_steps=1000000, log_freq = 1000, eps_start = 1, eps_decay = -1, eps_min = 0.1):
        
        model = self.deep(env)
        
        super(DDPG,self).__init__(model)
        self.discount = gamma
        self.env = env
        
        self.memory_max = memory_max
        self.eps = eps_start
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.done = 0
        self.log_freq = log_freq
        self.progbar = Progbar(self.memory_max)
        self.memory = dummy_obj.Memory(self.memory_max,self.batch_size,["t","state","action","reward","next_state","terminated"])

        self.eps_decay = eps_decay        
        if eps_decay == -1:            
            self.eps_decay = 1/train_steps
        self.eps_min = eps_min
        
    def act(self,state):
        
        if np.random.rand()<self.eps:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))
    
    def setup_agent():
        
        current_state = K.
        
    def train(self):
        
        to_log = 0
        self.progbar.__init__(self.batch_size*self.log_freq)
        
        while(self.done<self.train_steps):
            _ = self.env.reset()
            old_theta = self.Flaten.get()
            avg_rew = 0
            max_rew = 0
            min_rew = 0
            while to_log <self.log_freq:

                self.get_episode()
                rollout = self.memory.sample()

                actions = rollout["action"]
                rewards = rollout["reward"]
                not_final = np.logical_not(rollout["terminated"])

                avg_rew += np.mean(rewards)

                max_rew,min_rew = max(np.max(rewards),max_rew),min(min_rew,np.min(rewards))

                target_q = self.model.predict(rollout["next_state"])
                max_Q_prim = np.max(target_q,axis=1)
        
                for i in range(len(actions)):
                    target_q[i,actions[i]] = rewards[i] + not_final[i]* self.discount*max_Q_prim[i]
                
                self.model.train_on_batch(rollout["state"],target_q)
                
                to_log+=1

            new_theta = self.Flaten.get()

            self.log("Theta MSE",np.linalg.norm(new_theta-old_theta))
            self.log("Average reward",np.mean(avg_rew/self.log_freq))
            self.log("Max reward",max_rew)
            self.log("Min reward",min_rew)
            self.log("Epsilon",self.eps)
            self.log("Done",self.done)
            self.log("Total",self.train_steps)
            self.print_log()
            self.play()
            self.save(self.env.name)
            self.progbar.__init__(self.batch_size*self.log_freq)
                
            to_log = 0 
            
    def set_eps(self,x):
        self.eps = max(x,self.eps_min)
        
    def get_episode(self):
        
        episode = self.memory.empty_episode()

        state = self.env.current_state()        
        
        for i in range(self.batch_size):
            
            self.progbar.add(1)
            self.done += 1

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
            
            self.set_eps(self.eps-self.eps_decay)
            
            if done:
                state = self.env.reset()
                
        # record the episodes
        self.memory.record(episode)
        
        del(episode)

    def play(self,name='play'):
        
        name = name+self.env.name+str(self.eps)
        
        eps = self.eps
        
        self.set_eps(0)
        
        state = self.env.reset()
        #print(self.env.t,end=",")
        done = False
        
        while not done:
            
            action = self.act(state)
            
            state, _, done = self.env.step(action)
            #print(self.env.t,end=",")
        
        self.env.draw(name)
        self.set_eps(eps)

class DDPG2(DDPG):
    
    deep = deepfunctions.DeepQ2
    
