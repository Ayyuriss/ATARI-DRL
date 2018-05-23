#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""
import tensorflow as tf
tf.extract_image_patches
import ale_environment as environment
import agent as agents
#from rl_tools import *
from rollers import Roller
import gc
import time
gc.enable()
gc.collect()
#game = "breakout"

#env = ALE(game,num_frames = 2, skip_frames = 4, render = False)
#env = ALE("seaquest.bin")


game = "grid"
env = environment.GRID(grid_size=16)
agent = agents.DQN(env.states_dim,env.actions_n,'FC',0.99,1)
print(env.states_dim)
roll = Roller("Q", env, agent, 700000)
#agent.model.net.reduce_weights(10)
#agent.set_epsilon(1)
#agent.load("learned"+game+str(agent.eps))
time.sleep(5)
#agent.model.net.zero_initializer()

start = time.time()
for i in range(500):
    agent.set_epsilon(max(1-i/500,0.1))
    print('='*80+"\n")
    print('%f'%(time.time()-start))
    rollout = roll.rollout()
    agent.reinforce(rollout,500,8)
    del(rollout)
    roll.forget()
    agent.save("learned"+game+str(agent.eps))
    roll.play(i)
    
"""   
if False:
    import matplotlib.pyplot as plt

    X = np.arange(-1000,1000)
    
    softplus = lambda x : np.log(1+np.exp(x))
    softlog = lambda x : np.log(1+softplus(X))
    plt.plot(X,softlog(X))
    plt.plot(X, np.log(X*(X>0)))
    plt.plot(X, np.exp(softlog(X)))
    
    import keras
    from keras import backend as K
    from keras.utils.generic_utils import get_custom_objects
    
    def softlog(x):
        return K.log(K.softplus(x))
    def exp(x):
        return K.exp(x)
    get_custom_objects().update({'custom_activation': Activation(custom_activation)})
    
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(100,5)))
    model.add(keras.layers.Dense(10,activation = softlog))
    model.add(keras.layers.Dense(1,activation = exp))
    model.compile(optimizer='sgd',loss='mse')
    model.summary()
    
    X = np.random.normal(0,1,(100000,100,5))
    Y = np.random.normal(0,1,1000)
    model.add(Activation(custom_activation))
    Y = np.abs(X[:,5,2]*X[:,40,1]*X[:,50,3]*X[:,20,1])
    
    
    
    model.fit(X,Y)
    
    Z= model.predict(X)
    
    
    plt.plot(X, )


state = rollout['state']
n_state = rollout['next_state']
rew = rollout['reward']
pos_r = np.where(rew>0)[0]

state_pos = state[pos_r]
n_state_pos = n_state[pos_r]

success_cases = []
for i in pos_r:
    success_cases.append(state[i])
    success_cases.append(n_state[i])
success_cases = np.array(success_cases)

skvideo.io.vwrite("./plays/test" + '.mp4', (success_cases*255).astype('uint8'),inputdict={'-r':'5'})

self = agent

i = 1200
a = actions[i]
old = old_q[i]
r = rew[i]
s = state[i]
max_n_Q = self.discount*np.max(n_q,axis=1)

for i in range(len(actions)):
    target_q[i,actions[i]] = rew[i]
    if not_final[i]:
        target_q[i,actions[i]] += max_n_Q[i]

new = target_q[i]
new,old

a,r,old
skimage.io.imshow(s)



d = np.std(old_q,axis=0)
e = np.max(old_q,axis=0)
f = np.min(old_q,axis=0)
g = e-f

old_qq = agent.model.net.predict(state)
np.linalg.norm(old_q-old_qq)

"""

import h5py
filename = "./checkpoints/"+"learned"+game+str(agent.eps)
f = h5py.File(filename, 'r')

print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])
