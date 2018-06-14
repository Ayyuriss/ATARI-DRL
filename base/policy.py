#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 14:27:45 2018

@author: thinkpad
"""

import keras
import keras.backend as K
import numpy as np
class Policy(object):
    
    def __init__(self, env):
        self._init(env.observation_space, env.action_space)
    
    def _init(self, ob_space, ac_space):
        
        ob = K.placeholder(name="ob", shape=[None] + list(ob_space.shape))

        obscaled = ob / 255.0
        x = obscaled
        x = keras.layers.Conv2D(8, 8, strides=4, activation='relu', kernel_regularizer='l1')(x)
        x = keras.layers.Conv2D(16, 4, strides=2, activation='relu', kernel_regularizer='l2')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation='relu', kernel_initializer=normc_initializer(0.01))(x)
        self.logits = keras.layers.Dense(ac_space.shape[0], name="logits")(x)

    def mode(self):
        return K.argmax(self.logits, axis=-1)
    def neglogp(self, x):
        one_hot_actions = K.one_hot(x, self.logits.get_shape().as_list()[-1])
        return keras.backend.categorical_crossentropy(target=one_hot_actions, output=self.logits, from_logits=True)
    def kl(self, other):
        a0 = self.logits - K.max(self.logits, axis=-1, keep_dims=True)
        a1 = other.logits - K.max(other.logits, axis=-1, keep_dims=True)
        ea0 = K.exp(a0)
        ea1 = K.exp(a1)
        z0 = K.sum(ea0, axis=-1, keep_dims=True)
        z1 = K.sum(ea1, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return K.sum(p0 * (a0 - K.log(z0) - a1 + K.log(z1)), axis=-1)
    def entropy(self):
        a0 = self.logits - K.max(self.logits, axis=-1, keep_dims=True)
        ea0 = K.exp(a0)
        z0 = K.max(ea0, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return K.sum(p0 * (K.log(z0) - a0), axis=-1)
    def sample(self):
        u = K.random_uniform(self.logits.shape)
        return K.argmax(self.logits - K.log(-K.log(u)), axis=-1)

class Value(object):
    
    def __init__(self, env):
        self._init(env.observation_space, env.action_space)
    
    def _init(self, ob_space, ac_space):


        ob = K.placeholder(name="ob", shape=[None] + list(ob_space.shape))

        obscaled = ob / 255.0
        x = obscaled
        x = keras.layers.Conv2D(8, 8, strides=4, activation='relu', kernel_regularizer='l1')(x)
        x = keras.layers.Conv2D(16, 4, strides=2, activation='relu', kernel_regularizer='l2')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation='relu', kernel_initializer=normc_initializer(0.01))(x)
        self.vpred = keras.layers.Dense(1, name="value")(x)
        
            
def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return out
    return _initializer