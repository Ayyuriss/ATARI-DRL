#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""

from envs.grid import GRID
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
env = GRID(grid_size=36,square_size=4, stochastic = True)
time.sleep(5)

agent = DQN(env, 0.99, 100000, 32, 1000000, log_freq = 1000, eps_start = 0.1, eps_decay = 1/7e5 )
agent.load("dqnGRID")
#agent = TRPO(env,0.99,10000)

print(env.observation_space)
agent.train()
agent.play()
