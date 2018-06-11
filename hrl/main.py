# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:04:02 2018

@author: gamer
"""

import sys
sys.path.append("../")

import numpy as np
from envs.grid import GRID,GRID2
from agents.dqn import *
from agents.ddqn import *
from agents.trpo import *
from nn import deepfunctions
import keras
import tensorflow as tf

if False:
    grid_size = 36
    
    env = GRID2(grid_size=grid_size,square_size=4, stochastic = True)
    
    inputs = keras.layers.Input(shape=env.observation_space.shape)
    
    cat = deepfunctions.conv_block(inputs)
    cat = keras.layers.Flatten()(cat)
    cat = keras.layers.Dense(256, activation='tanh')(cat)
    cat = keras.layers.Dense(2, activation='linear')(cat)
    
    cat = keras.models.Model(inputs,cat)
    cat.compile('rmsprop','mse')
    cat.summary()
    mouse = deepfunctions.conv_block(inputs)
    mouse = keras.layers.Flatten()(mouse)
    mouse = keras.layers.Dense(256, activation='tanh')(mouse)
    mouse = keras.layers.Dense(2, activation='linear')(mouse)
    
    mouse = keras.models.Model(inputs,mouse)
    mouse.compile('rmsprop','mse')
    
    
    
    def rollout(n):
        states = []
        mouses = []
        cats = []
        
        state = env.reset()
        for _ in range(n):
            states.append(state)
            mouses.append(env.get_mouse())
            cats.append(env.get_cat())
            state,_,done = env.step(env.action_space.sample())
            if done:
                state = env.reset()
    
        states = np.array(states)
        mouses = np.array(mouses)
        cats = np.array(cats)
        
        return states, mouses, cats
        
        
    
    for _ in range(10):
        states, mouses, cats = rollout(5000)
        cat.fit(states,cats)
        mouse.fit(states,mouses)
    
    cat=keras.models.load_model("./cat_model")
    mouse=keras.models.load_model("./mouse_model")
    
    states, mouses, cats = rollout(5000)
    cat.evaluate(states,cats) # 0.00011150
    mouse.evaluate(states,mouses) # 0.0004631
    
    
    for l in cat.layers:
        for x in l.trainable_weights:
            l.non_trainable_weights.append(x)
        l.trainable_weights = []
    for l in mouse.layers:
        for x in l.trainable_weights:
            l.non_trainable_weights.append(x)
        l.trainable_weights = []
    
    mouse.trainable = False
    cat.trainable = False
    print(cat.summary())
    
    
    cat = keras.models.Model(cat.input,cat.output)
    cat.compile('rmsprop','mse')
    cat.summary()
    
    mouse = keras.models.Model(mouse.input,mouse.output)
    mouse.compile('rmsprop','mse')
    mouse.summary()
    
    
    cat_ready = cat.layers[1](inputs)
    for i in range(2,6):
        cat_ready = cat.layers[i](cat_ready)
        
    mouse_ready = mouse.layers[1](inputs)
    for i in range(2,6):
        mouse_ready = mouse.layers[i](mouse_ready)
        
    elements = [mouse_ready,cat_ready]
    
    concat = keras.layers.concatenate(elements)
    
    dense = keras.layers.Dense(64,activation='relu')(concat)
    dense = keras.layers.Dense(64,activation='relu')(dense)
    outputs = keras.layers.Dense(env.action_space.n)(dense)
    
    model = keras.models.Model(inputs,outputs)
    model.compile('rmsprop','mse')
    model.summary()
    
    
    model.fit(states,np.random.normal(0,1,(len(states),4)))
    
    mouse_w1 = mouse.get_weights()
    model.save("merged_model")
    model = keras.models.load_model("merged_model")
    from agents.dqn import DQN
    
    agent =  DQN2(env, 0.99, 50000, 32, 1000000, log_freq = 1000, eps_start = 1, eps_decay = 5e-5 )
    agent.model.net = model
    agent.Flaten.__init__(model.trainable_weights)
    agent.done=0;agent.train()
    agent.done=0;agent.eps_decay = 1e-5;
    agent.eps = 1
    
    agent2 =  TRPO2(env, 0.99, 10000 )
    for _ in range(10):agent2.train()
    
    grid_size = 36
    env = GRID(grid_size=grid_size,square_size=4, stochastic = True)
    agent =  DDQN(env, 0.99,32, 50000,10000,5000000, log_freq = 1000, eps_start = 0.1, eps_decay = 1e-6 )
    agent.model.load('dqnGRID')
    agent.train()
    
