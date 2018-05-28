#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""

import ale_environment as environment
from agent_dqn import DQN
#from rl_tools import *
from rollers import Roller
import gc
import time
gc.enable()
gc.collect()


#game = "breakout"
#env = ALE(game,num_frames = 2, skip_frames = 4, render = False)


game = "grid"
env = environment.GRID(grid_size=10)
agent = DQN(env.states_dim,env.actions_n,'FC',0.99,1)
print(env.states_dim)
roller = Roller(env, agent, 300000)

time.sleep(5)

start = time.time()
for i in range(100):
    print('='*80+"\n")
    print('%f'%(time.time()-start))
    rollout = roller.rollout(50000)
    for _ in range(5):
        agent.reinforce(rollout,32,1)
    agent.save("learned"+game+str(i))
    roller.play(i)
