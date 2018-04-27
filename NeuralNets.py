#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:28:08 2018

@author: thinkpad
"""

import keras
import numpy as np
from keras import layers
from keras import backend as K


# ================================================================
# Base structure for Neural structure
# ================================================================

class BaseNetwork(keras.Sequential):
    def __init__(self,states_dim,actions_n):
        super(BaseNetwork, self).__init__()
        
        self.input_dim = states_dim
        self.output_n = actions_n
        self.create_network()
    
    def create_network(self):
        raise NotImplementedError
    def fit(self,X,Y,epochs=10,batch_size=30):
        print("Fitting the NN:",X.shape, Y.shape)
        super(BaseNetwork, self).fit(X,Y,batch_size,epochs)
    
    @property
    def trainable_variables(self):
        return self.trainable_weights
    
    def zero_initializer(self):
        zero_weights = []
        for x in self.trainable_variables:
            zero_weights.append(np.zeros(x.shape))
        self.set_weights(zero_weights)
        
    def predict(self,image):
        if image.ndim == 4:
            return super(BaseNetwork,self).predict(image)
        elif image.ndim == 3:
            _image = image.reshape((1,)+image.shape)
            return super(BaseNetwork,self).predict(_image)[0]
        else:
            raise(NotImplementedError,"Images dimension:%d" % image.ndim)
    def save(self,name):
        self.save_weights(name)
    def load(self,name):
        self.load_weights(name, by_name=False)

# ================================================================
# Convulutional Structure
# ================================================================
        
class ConvNet(BaseNetwork):
    
    def create_network(self):
        n_filters_1 = 16
        k_size_1 = 4
        stride_1 = 2
        
        n_filters_2 = 32
        k_size_2 = 3
        stride_2 = 2
        
        self.add(layers.Conv2D(n_filters_1,k_size_1,strides=stride_1,activation='relu',input_shape=self.input_dim))
        self.add(layers.MaxPooling2D())
        self.add(layers.Conv2D(n_filters_2,k_size_2,strides=stride_2,activation='relu'))
        self.add(layers.Flatten())
        self.add(layers.Dense(256,activation='softplus'))
        self.add(layers.Dense(128,activation='softplus'))
# ================================================================
#  Fully Connected structures
# ================================================================
class FCNet(BaseNetwork):

    def create_network(self):
        self.add(layers.MaxPooling2D(input_shape = self.input_dim))
        self.add(layers.Flatten())
        self.add(layers.Dense(1024,activation='relu'))
        #self.add(layers.Dense(2048,activation='tanh'))
        self.add(layers.Dense(1024,activation='sigmoid'))
        self.add(layers.Dense(1024,activation='relu'))
        self.add(layers.Dense(512,activation='softplus'))

class SingleFCNet(BaseNetwork):
    def __init__(self,states_dim,actions_n):
        self.input_dim = np.prod(states_dim)+actions_n+2
        self.output_n = 1
        
    def create_network(self, shape):
        self.add(layers.Dense(1, input_shape=self.input_dim,activation='tanh'))
        self.compile(optimizer = 'adam',loss='mean_squared_error')


        
# ================================================================
# Neural Nets for Policy Gradient
# ================================================================
        
class P_ConvNet(ConvNet):
    def create_network(self):
        super(P_ConvNet,self).create_network()
        self.add(layers.Dense(self.output_n,activation='softmax'))
        self.compile(optimizer='rmsprop',loss='kullback_leibler_divergence')
        print(self.summary())
        
class P_FCNet(FCNet):
    def create_network(self):
        super(P_FCNet,self).create_network()
        self.add(layers.Dense(self.output_n,activation='softmax'))
        self.compile(optimizer='rmsprop',loss='kullback_leibler_divergence')
        print(self.summary())

# ================================================================
# Neural Nets for Q Learning
# ================================================================
class Q_ConvNet(ConvNet):

    def create_network(self):
        super(Q_ConvNet,self).create_network()
        self.add(layers.Dense(self.output_n,activation='linear'))
        self.compile(optimizer='rmsprop',loss='mean_squared_error')
        print(self.summary())
        
class Q_FCNet(FCNet):

    def create_network(self):
        super(Q_FCNet,self).create_network()

        self.add(layers.Dense(self.output_n,activation='linear'))
        self.compile(optimizer='rmsprop',loss='mean_squared_error')
        print(self.summary())


    

"""
images = np.random.normal(0,1,(1000,48,48,3))
Y = np.random.rand(4*1000).reshape((1000,4))
Y = (Y.T/Y.sum(axis=1)).T
states_dim = images.shape[1:]
actions_n = 4
net = Q_ConvNet(states_dim, actions_n)
"""