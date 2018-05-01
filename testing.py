#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""
from ale_environment import *
from agent import *
import rl_tools


game = "breakout"
env = ALE(game,num_steps = 4, skip_frame = 1)
#env = ALE("seaquest.bin")
agent = DQN(env.states_dim,env.actions_n,'CONV',0.99,.05)
#env.step(0)
print(env.states_dim)


for _ in range(100):
    rollout = rl_tools.rollouts(env, agent, 100, 700)
    agent.reinforce(rollout)
    agent.save("leaneard4"+game)





