#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""

import ale_environment as environment
from agent_dqn import DQN
from rollers import Roller
import gc
import time
gc.enable()
gc.collect()
#game = "breakout"
#env = ALE(game,num_frames = 2, skip_frames = 4, render = False)
#env = ALE("seaquest.bin")


game = "grid"
env = environment.GRID(grid_size=16,square_size=2)
agent = DQN(env.states_dim,env.actions_n,'FC',0.99,1)
#agent.load("learnedgrid0.1")
print(env.states_dim)
roller = Roller(env, agent, 200000)

time.sleep(5)
#agent.model.net.zero_initializer()

start = time.time()
for i in range(100):
    print('='*80+"\n")
    print('%f'%(time.time()-start))
    rollout = roller.rollout(100000)
    for _ in range(3):
        agent.reinforce(rollout)
    del(rollout)
    agent.save("learned"+game+str(agent.eps))
    roller.play(i)
