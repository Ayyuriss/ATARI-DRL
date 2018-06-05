#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""

from envs import environments
from agents.dqn import DQN
from agents.trpo import TRPO
import gc
import time
gc.enable()
gc.collect()
#game = "breakout"
#env = ALE(game,num_frames = 2, skip_frames = 4, render = False)
#env = ALE("seaquest.bin")


game = "grid"
env = environments.GRID(grid_size=32,square_size=2)
agent = DQN(env,'FC',0.99,100000)
#agent.load("learned"+game+str(0.1))
agent.model.net.reducer.display_update()
#agent.model.net.reducer.compile(agent.model.net.model)
#agent = TRPO(env.states_dim, env.actions_n,'FC',0.99)

print(env.observation_space)
time.sleep(5)
#agent.model.net.zero_initializer()
for i in range(100):
    agent.train(10000,10)
    agent.save("learned"+game+str(agent.eps))
    agent.play(i)
