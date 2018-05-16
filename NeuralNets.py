#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:28:08 2018

@author: thinkpad
"""

import keras
import numpy as np
from keras import layers
import keras.backend as K
from utils.console import Progbar

# =============================================================================
# Base structure for Neural structure
# =============================================================================

class BaseNetwork(keras.Sequential):

    def __init__(self,states_dim,actions_n):

        super(BaseNetwork, self).__init__()        

        self.input_dim = states_dim
        self.output_n = actions_n
        self.create_network()
    
    def create_network(self):
        raise NotImplementedError
        
    def fit(self,X,Y,batch_size=50):
        total = len(X)
        progbar = Progbar(total)
        print("Fitting the NN:",X.shape, Y.shape)
        
        loss0 = []
        loss1 = []        
        progbar.add(batch_size)
        for i in range(batch_size,len(X),batch_size):

            loss0.append(super(BaseNetwork, self).train_on_batch(X[i-batch_size:i],Y[i-batch_size:i]))
            loss1.append(np.mean((self.predict(X[i-batch_size:i])-Y[i-batch_size:i])**2))
            
            progbar.add(batch_size)

        print("Initial loss: %f, Final loss: %f"%(sum(loss0),sum(loss1)))
        
    def zero_initializer(self):

        for x in self.trainable_variables:            
            K.set_value(x, np.zeros(x.shape))
        
    def predict(self,image):

        if image.ndim == len(self.input_dim):
            _image = image.reshape((1,)+image.shape)
            return super(BaseNetwork,self).predict(_image)[0]
        else:
            return super(BaseNetwork,self).predict(image)

    def save(self,name):
        self.save_weights(name)

    def load(self,name):
        self.load_weights(name, by_name=False)
        
    @property
    def trainable_variables(self):
        return self.trainable_weights

# =============================================================================
#  Fully Connected structures
# =============================================================================

class ValueFunction(BaseNetwork):
        
    def __init__(self,states_dim,actions_n):
        self.input_dim = (np.prod(states_dim)+actions_n+2,)
        self.output_n = 1
        super(ValueFunction,self).__init__(self.input_dim, self.output_n)

    def create_network(self):
        self.add(layers.Dense(1, input_shape = self.input_dim,activation='tanh'))
        self.compile(optimizer = 'adam',loss='mean_squared_error')
        print(self.summary())
    

# =============================================================================
# Neural Nets for Policy Gradient
# =============================================================================
        
class Policy_CNNet(BaseNetwork):
    
    def create_network(self):
        n_filters_1 = 16
        k_size_1 = 8
        stride_1 = 4
        
        n_filters_2 = 32
        k_size_2 = 4
        stride_2 = 2
        
        self.add(layers.Conv2D(n_filters_1, k_size_1, strides=stride_1,
                               activation='softplus', input_shape = self.input_dim))
        self.add(layers.Conv2D(n_filters_2, k_size_2, strides=stride_2, 
                               activation='softplus'))


        self.add(layers.Flatten())
        self.add(layers.Dense(256,activation='relu'))
        self.add(layers.Dense(self.output_n,activation='softmax'))
        self.compile(optimizer='sgd',loss='kullback_leibler_divergence')
        print(self.summary())
        
class Policy_FCNet(BaseNetwork):

    def create_network(self):
        self.add(layers.MaxPooling2D(input_shape = self.input_dim))
        self.add(layers.Flatten())
        self.add(layers.Dense(1024,activation='tanh'))
        self.add(layers.Dense(1024,activation='tanh'))
        self.add(layers.Dense(self.output_n, activation='softmax'))
        self.compile(optimizer='sgd',loss='kullback_leibler_divergence')
        print(self.summary())

# =============================================================================
# Neural Nets for Q Learning
# =============================================================================

class Q_CNNet(BaseNetwork):

    def create_network(self):
        n_filters_1 = 16
        k_size_1 = 16
        stride_1 = 4
        
        self.add(layers.Conv2D(n_filters_1, k_size_1, strides=stride_1,
                               activation='softplus', input_shape = self.input_dim))
        self.add(layers.ZeroPadding2D())
        #self.add(layers.Conv2D(n_filters_1, k_size_1, strides=stride_1,
        #                       activation='softplus'))
        #self.add(layers.MaxPooling2D((2,2),strides=(2,2)))
        
        
        n_filters_2 = 32
        k_size_2 = 4
        stride_2 = 2

        self.add(layers.Conv2D(n_filters_2, k_size_2, strides=stride_2, 
                               activation='softplus'))
        self.add(layers.ZeroPadding2D())        
        self.add(layers.Conv2D(16, 2, strides=1, 
                               activation='softplus'))
        #21self.add(layers.MaxPooling2D((2,2),strides=(2,2)))

        self.add(layers.Flatten())
        self.add(layers.Dense(256,activation='softplus'))
        self.add(layers.Dense(128,activation='softplus'))
        self.add(layers.Dense(self.output_n,activation='linear'))
        self.compile(optimizer='sgd',loss='mean_squared_error')
        print(self.summary())
    
    
class Q_FCNet(BaseNetwork):

    def create_network(self):
                
        self.add(layers.Flatten(input_shape=self.input_dim))
        self.add(layers.Dense(128,activation='tanh'))
        self.add(layers.Dense(64,activation='tanh'))
        self.add(layers.Dense(self.output_n,activation='linear'))
        
        self.compile(optimizer='adam',loss='mean_absolute_error')
        print(self.summary())
        
        
        
class Q_RCNNet(BaseNetwork):

    def create_network(self):
        n_filters_1 = 64
        k_size_1 = 3
        stride_1 = 3
        
        self.add(layers.Conv2D(n_filters_1, k_size_1, strides=stride_1,
                               activation='relu', input_shape = self.input_dim))
        self.add(layers.ZeroPadding2D())
        self.add(layers.Conv2D(n_filters_1, k_size_1, strides=stride_1,
                               activation='relu'))
        self.add(layers.MaxPooling2d((2,2),strides=(2,2)))
        
        
        n_filters_2 = 128
        k_size_2 = 4
        stride_2 = 2

        self.add(layers.Conv2D(n_filters_2, k_size_2, strides=stride_2, 
                               activation='relu'))
        self.add(layers.ZeroPadding2D())        
        self.add(layers.Conv2D(n_filters_2, k_size_2, strides=stride_2, 
                               activation='relu'))
        self.add(layers.MaxPooling2d((2,2),strides=(2,2)))

        self.add(layers.Flatten())
        self.add(layers.Dense(512,activation='relu'))
        self.add(layers.Dense(128,activation='relu'))
        self.add(layers.Dense(self.output_n,activation='linear'))
        self.compile(optimizer='sgd',loss='mean_squared_error')
        print(self.summary())
        
def RCNN_layer(input_fitlers, output_filters):
    pass