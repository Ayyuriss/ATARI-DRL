#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""
from ale_environment import *
from agent import *
import rl_tools
class Ayoub(object):
    
    def __init__(self):
        
        self.display()
        
    def display(self):
        print("Ayoub")
        
        
class Ghriss(Ayoub):
    
    def __ini__(self):
        
        super(Ghriss, self).__init__()
    
    def display(self):
        print("Ghriss")
game = "breakout"
env = ALE(game)
#env = ALE("seaquest.bin")
agent = DQN(env.states_dim,env.actions_n,'FC',0.99,1.0)
#env.step(0)
print(env.states_dim)
print(env.getScreenRGB().shape)

for _ in range(10):
    rollout = rl_tools.rollouts(env, agent, 10, 500)
    agent.reinforce(rollout)
    agent.save("leaneard"+game)
    agent.set_epsilon(agent.eps/2)