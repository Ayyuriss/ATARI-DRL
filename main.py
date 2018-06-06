#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""

from envs.grid import GRID
#from agents.dqn import DQN
from agents.trpo import TRPO
import gc
import time
gc.enable()
gc.collect()
#game = "breakout"
#env = ALE(game,num_frames = 2, skip_frames = 4, render = False)
#env = ALE("seaquest.bin")


game = "grid"
env = GRID(grid_size=32,square_size=3)
agent = TRPO(env,0.99,num_steps)
#agent.load("learned"+game+str(0.1))
#agent.model.reducer.display_update()
#agent.model.net.reducer.compile(agent.model.net.model)
#agent = TRPO(env.states_dim, env.actions_n,'FC',0.99)

print(env.observation_space)
agent.train()

