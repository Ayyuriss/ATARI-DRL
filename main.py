#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""

from envs.grid import *
from agents.dqn import DQN
from agents.ddqn import *
from agents.trpo import TRPO
import gc
import time
gc.enable()
gc.collect()
#game = "breakout"
#env = ALE(game,num_frames = 2, skip_frames = 4, render = False).
#env = ALE("seaquest.bin")


game = "grid"
env = GRID(grid_size=36,square_size=4, stochastic = True)
_ = env.reset()
env.draw_frame()
time.sleep(5)

agent = DDQN4(env, 0.987, 16, memory_max=50000,train_steps= 10000000, double_update = 512, eps_start = 1, eps_decay = 5e-6)
#agent.load("dqnGRID")
agent = TRPO(env,0.99,1024)

print(env.observation_space)

for _ in range(50):
    for _ in range(50):
        agent.train()
    agent.play()

from nn import deepfunctions
