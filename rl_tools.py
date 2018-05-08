#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:43:53 2018

@author: thinkpad
"""


import numpy as np
import scipy.signal
import collections
import DeepFunctions

EPS = np.finfo(np.float64).tiny

def to_categorical(Y,n):
    _Y = np.zeros((len(Y),n))
    mY = int(min(Y))
    MY = int(max(Y))
    assert(MY-mY+1) == n
    for i,y in enumerate(Y):
        _Y[i,int(y)-mY]=1
    return _Y.astype(int)

def positify(y):
    _y = y.reshape(-1,3)
    mins = _y.min(axis=0)
    maxs = _y.max(axis=0)
    y2 = (y - mins)/(maxs-mins)
    return y2

    




def write_dict(dic):

    fout = "./here.txt"
    fo = open(fout, "a+")
    fo.write('\n'+'-'*10+'\n')
    for k, v in dic.items():
        fo.write(str(k) + ' >>> '+ str(v) + '\n')
    fo.close()


def rollout(env, agent, len_episode):
    """
    Simulate the env and agent for len_episode steps
    """
    env.reset()
    state,_,start_lives = env.step(0)
    
    episode = {"state":[],"action":[],"reward":[],"terminated":[]}
    
    for _ in range(len_episode):
        episode["state"].append(state)
        action = agent.act(state)
        
        episode["action"].append(action)
        
        state, rew, done = env.step(action)


        episode["reward"].append(rew)        
        episode["terminated"].append(done)
        if done:
            break
    episode = {k:np.array(v) for (k,v) in episode.items()}
    return episode



class Roller(object):
    
    def __init__(self,env, agent, max_steps):
        
        self.env = env
        self.agent = agent
        self.discount = agent.discount
        self.max_length = max_steps
        self.policy = (agent.type == 'Policy')
        
#        self.memory = {"n": [], "state":[],"action":[],"reward":[],"terminated":[],"output":[],"return":[]}
        self.episodes = []
        if self.policy:
            self.baseline = DeepFunctions.ValueFunction(self.env.states_dim, self.env.actions_n)


    def rollout(self):
        
        current_length = 0
#        for k in self.memory:
#            self.memory = []
        
        while current_length < self.max_length:
            
            episode = self.get_episode(self.max_length-current_length, policy=self.policy)    
            current_length += len(episode['state'])  
            self.episodes.append(episode)
        
        self.compute_advantage()
        return self.episodes


    def get_episode(self, max_step, policy=False):
        
        self.env.reset()
        state, _, done = self.env.step(0)
        episode = {"n": [], "state":[],"action":[],"reward":[],"terminated":[],"output":[]}
        
        for i in range(max_step):

            output = self.agent.model.evaluate(state)
            action = self.agent.act(state)
            state, rew, done = self.env.step(action)            
            episode["n"].append(i)
            episode["state"].append(state)
            episode["action"].append(action)
            
            episode["reward"].append(rew)        
            episode["terminated"].append(done)
            episode["output"].append(output)
            
            if done:
                if policy:
                    episode["reward"][-1] = -sum(episode["reward"][:-1])
                break
        episode = {k:np.array(v) for (k,v) in episode.items()}
        episode["return"] = discount(episode["reward"], self.agent.discount)
        return episode


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

def rollouts(env, agent, num_episodes, len_episode):
    print("Starting rollouts")    
    episodes = []
    for i in range(num_episodes):
        print("Rollout %d/%d"%(i,num_episodes))
        episodes.append(rollout(env, agent, len_episode))
    states = np.concatenate([episode["state"] for episode in episodes], axis = 0)
    actions = np.concatenate([episode["action"] for episode in episodes]).astype(int)
    rewards = np.concatenate([episode["reward"] for episode in episodes])
    terminated = np.concatenate([episode["terminated"] for episode in episodes])
    return {"states":states, "actions":actions, "rewards": rewards,"terminated":terminated}


def policy_rollouts(env, agent, value_func, num_episodes, len_episode):
    print("Starting rollouts")    
    episodes = []
    for i in range(num_episodes):
        print("Rollout %d/%d"%(i,num_episodes))
        episodes.append(_policy_rollout(env, agent, len_episode))
    
    compute_advantage(value_func, episodes, agent.discount)
    
    states = np.concatenate([episode["state"] for episode in episodes], axis = 0)
    actions = np.concatenate([episode["action"] for episode in episodes]).astype(int)
    proba = np.concatenate([episode["proba"] for episode in episodes],axis=0)
    advantages = np.concatenate([episode["advantages"] for episode in episodes])
    
    return {"states":states, "actions":actions, "proba": proba,
            "advantages": advantages}