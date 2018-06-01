#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""

import environments
from agent_dqn import DQN
from agent_trpo import TRPO
import gc
import time
gc.enable()
gc.collect()
#game = "breakout"
#env = ALE(game,num_frames = 2, skip_frames = 4, render = False)
#env = ALE("seaquest.bin")


game = "grid"
env = environments.GRID(grid_size=32,square_size=2)
agent = DQN(env,'FC',0.99,50000)
#agent = TRPO(env.states_dim, env.actions_n,'FC',0.99)

print(env.observation_space)
time.sleep(5)
#agent.model.net.zero_initializer()
for i in range(100):
    agent.train(10000,20)
    agent.train(10000,5)
    agent.save("learned"+game+str(agent.eps))
    agent.play(i)
