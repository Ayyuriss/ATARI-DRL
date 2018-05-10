# -*- coding: utf-8 -*-
"""
Created on Tue May  8 08:51:00 2018

@author: gamer
"""

from DeepFunctions import ValueFunction

import collections
import numpy as np
import scipy.signal
from utils.console import Progbar
EPS = np.finfo(np.float32).tiny

class Roller(object):
    
    def __init__(self,agent_type, env, agent, max_steps):
        
        self.env = env
        self.agent = agent
        self.discount = agent.discount
        self.max_steps = max_steps
        self.policy = (agent_type == 'Policy')
        
#        self.memory = {"n": [], "state":[],"action":[],"reward":[],"terminated":[],"output":[],"return":[]}
        self.memory = collections.deque([],self.max_steps)
        if self.policy:
            self.baseline = ValueFunction(self.env.states_dim, self.env.actions_n)
        self.progbar = Progbar(self.max_steps)

    def rollout(self):
        
        self.forget()
        
        while len(self.memory['t']) < self.max_steps:
            
            self.get_episode(self.max_steps-len(self.memory['t']), policy=self.policy)
            
        for k in self.memory.keys():
            self.memory[k] = np.concatenate([self.memory[k]],axis=0)        
        self.memory['state'] = normalize(self.memory['state'])

        if self.policy:
            self.compute_advantage()

        self.scramble()

        return self.memory
        

    def get_episode(self, max_step, policy=False):
        
        self.env.reset()
        state, _, done = self.env.step(0)
        
        episode = {"t": [], "state":[],"action":[],"reward":[],"terminated":[],"output":[]}
        
        for i in range(max_step):
            self.progbar.add(1)
            
            episode["state"].append(state)

            action = self.agent.act(state)

            state, rew, done = self.env.step(action)

            output = self.agent.model.evaluate(state)
            
            episode["t"].append(i)            
            episode["action"].append(action)
            episode["output"].append(output)
            episode["reward"].append(rew)        
            episode["terminated"].append(done)
            
            if done:
                #episode["reward"][-1] = -5
                break
        episode = {k:np.array(v) for (k,v) in episode.items()}
        episode["return"] = discount(episode["reward"], self.discount)
        
        self.record(episode)


    def record(self,episode):
        for k in episode.keys():
            for v in episode[k]:
                self.memory[k].append(v) 
                
    def forget(self):
        self.memory = {s : collections.deque([],self.max_steps) for s in ["t", "state","action","reward","terminated","output","return"] }
        self.progbar.__init__(self.max_steps)
    
    def scramble(self):
        n = len(self.memory['t'])
        idx = np.random.permutation(np.arange(n))
        for k,v in self.memory.items():
            self.memory[k] = v[idx]
        
    def compute_advantage(self):
        # Compute baseline, advantage
        for episode in self.episodes:
            b = episode["baseline"] = self.baseline.evaluate(episode)
            b1 = np.append(b, 0 if episode["terminated"][-1] else b[-1])
            deltas = episode["reward"] + self.discount*b1[1:] - b1[:-1] 
            episode["advantage"] = discount(deltas, self.discount)
        alladv = np.concatenate([episode["advantage"] for episode in self.episodes])    
        # Standardize advantage
        std = alladv.std()
        mean = alladv.mean()
        for episode in self.episodes:
            episode["advantage"] = (episode["advantage"] - mean) / std


def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.
    inputs
    ------
    x: ndarray
    gamma: float
    outputs
    -------
    y: ndarray with same shape as x, satisfying
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1
    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]        


def normalize(v):
    norm = np.linalg.norm(v)
    v_tmp = v-np.mean(v)
    if norm < EPS: 
       return v_tmp
    return v_tmp/np.max(np.abs(v_tmp))