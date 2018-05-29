# -*- coding: utf-8 -*-
"""
Created on Tue May  8 08:51:00 2018

@author: gamer
"""
import dummy_obj
import numpy as np
import scipy.signal
from utils.console import Progbar
EPS = np.finfo(np.float32).tiny

class Roller(object):
    
    def __init__(self, env, agent, memory_max):
        
        self.env = env
        self.agent = agent

        self.memory_max = memory_max
        
        self.progbar = Progbar(self.memory_max)
        
        self.memory = dummy_obj.Memory(self.memory_max,["t","state","action","next_state","reward","return","terminated"])

    def rollout(self,num_steps):
        
        roll={}        
        # do the rollouts
        self.get_episodes(num_steps)
        
        # scramble
        
        # vectorize results
        
        roll = self.memory.random_sample(num_steps)        

        # calcualte advantage
        #if self.policy:
        #    self.compute_advantage()
        return roll
        

    def get_episodes(self, length):
        
        
        state = self.env.reset()
        
        self.agent.set_epsilon(1)
        
        episode = self.memory.empty_episode()
        self.progbar.__init__(length)
        for i in range(length):
            
            self.progbar.add(1)

            # save current state
            episode["state"].append(state)
            
            # act
            action = self.agent.act(state)   
            state, rew, done = self.env.step(action)
            episode["next_state"].append(state)

            
            episode["t"].append(i)            
            episode["action"].append(action)
            episode["reward"].append(rew)        
            episode["terminated"].append(done)
            
            if done:
                state = self.env.reset()
            
            self.agent.decrement_eps(1/length)
        # vectorize results)
        episode["return"] = discount(np.array(episode["reward"]), self.agent.discount)
        # record the episodes
        self.memory.record(episode)
        
        del(episode)


    def compute_advantage(self):
        # Compute baseline, advantage
        for episode in self.episodes:
            b = episode["baseline"] = self.baseline.predict(episode)
            b1 = np.append(b, 0 if episode["terminated"][-1] else b[-1])
            deltas = episode["reward"] + self.discount*b1[1:] - b1[:-1] 
            episode["advantage"] = discount(deltas, self.discount)
        alladv = np.concatenate([episode["advantage"] for episode in self.episodes])    
        # Standardize advantage
        std = alladv.std()
        mean = alladv.mean()
        for episode in self.episodes:
            episode["advantage"] = (episode["advantage"] - mean) / std

    def play(self,name='play'):
        eps = self.agent.eps
        self.agent.set_epsilon(0)
        state = self.env.reset()
        done = False
        
        while not done:
            
            action = self.agent.act(state)
            
            state, _, done = self.env.step(action)
        
        self.env.draw(name)
        self.agent.set_epsilon(eps)

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]        


def normalize(v):
    norm = np.linalg.norm(v)
    v_tmp = v-np.mean(v)
    if norm < EPS: 
       return v_tmp
    return v_tmp/np.max(np.abs(v_tmp))

