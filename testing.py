#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""
from ale_environment import *
from agent import *
from rl_tools import *
from rollers import Roller

game = "breakout"
env = ALE(game,num_frames = 4, skip_frames = 4)
#env = ALE("seaquest.bin")
agent = DQN(env.states_dim,env.actions_n,'CNN',0.99,1)
#env.step(0)
print(env.states_dim)
roll = Roller("Q", env, agent, 2000)

agent.set_epsilon(1)
#agent.load("learned4"+game+str(agent.eps))
import time
start = time.time()
for i in range(100):   
    print('='*50+'\n\n'+'='*50)
    print('%f'%(time.time()-start))
    rollout = roll.rollout()
    agent.reinforce(rollout)
    agent.save("learned4"+game+str(agent.eps))
    agent.set_epsilon(max(agent.eps*0.9,0.05))
        

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
        




"""
