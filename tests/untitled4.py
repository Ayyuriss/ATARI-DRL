#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:08:22 2018

@author: thinkpad
"""

import keras

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(4,4,input_shape=(16,16,3)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dense(1))
model.compile('adam','mse')
model.summary()
flat = Flattener(model.weights)
loss = K.mean((model.output-1)**2)
grads = flat.flatgrad(loss)
