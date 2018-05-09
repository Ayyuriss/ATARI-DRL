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
            

