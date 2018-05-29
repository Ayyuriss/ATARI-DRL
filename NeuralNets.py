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

class BaseNetwork(object):

    def __init__(self,states_dim,actions_n):

        self.model = keras.models.Sequential()   

        self.input_dim = states_dim
        self.output_n = actions_n
        self.create_network()
        self.progbar = Progbar(1)
    def create_network(self):
        raise NotImplementedError
        
    def fit(self,X,Y,batch_size=50):
        #total = len(X)
        #self.progbar.__init__(total)
        print("Fitting the NN:",X.shape, Y.shape)
        
        #loss0 = []
        #loss1 = []        
        #self.progbar.add(batch_size)
        #for i in range(batch_size,len(X),batch_size):

#            loss0.append(self.model.train_on_batch(X[i-batch_size:i],Y[i-batch_size:i]))
#            loss1.append(np.mean((self.model.predict(X[i-batch_size:i])-Y[i-batch_size:i])**2))
#            
#            self.progbar.add(batch_size)

 #       print("Initial loss: %f, Final loss: %f"%(sum(loss0),sum(loss1)))
  
        self.model.fit(X,Y,batch_size,1)      
    def zero_initializer(self):

        for x in self.trainable_variables:            
            K.set_value(x, np.zeros(x.shape))
            
    def reduce_weights(self,factor):

        for x in self.trainable_variables:            
            K.set_value(x, K.eval(x)/factor)
        
    def predict(self,image):

        if image.ndim == len(self.input_dim):
            _image = image.reshape((1,)+image.shape)
            return self.model.predict(_image)[0]
        else:
            return self.model.predict(image)

    def save(self,name):
        self.model.save(name)

    def load(self,name):
        self.model = keras.models.load_model(name)
        
    @property
    def trainable_variables(self):
        return self.model.trainable_weights

# =============================================================================
#  Fully Connected structures
# =============================================================================

class ValueFunction(BaseNetwork):
        
    def __init__(self,states_dim,actions_n):
        self.input_dim = (np.prod(states_dim)+actions_n+2,)
        self.output_n = 1
        super(ValueFunction,self).__init__(self.input_dim, self.output_n)

    def create_network(self):
        self.model.add(layers.Dense(1, input_shape = self.input_dim,activation='tanh'))
        self.compile(optimizer = 'adam',loss='mean_squared_error')
        print(self.model.summary())
    

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
        
        self.model.add(layers.Conv2D(n_filters_1, k_size_1, strides=stride_1,
                               activation='softplus', input_shape = self.input_dim))
        self.model.add(layers.Conv2D(n_filters_2, k_size_2, strides=stride_2, 
                               activation='softplus'))


        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256,activation='relu'))
        self.model.add(layers.Dense(self.output_n,activation='softmax'))
        self.model.compile(optimizer='sgd',loss='kullback_leibler_divergence')
        print(self.model.summary())
        
class Policy_FCNet(BaseNetwork):

    def create_network(self):
        self.model.add(layers.MaxPooling2D(input_shape = self.input_dim))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1024,activation='tanh'))
        self.model.add(layers.Dense(1024,activation='tanh'))
        self.model.add(layers.Dense(self.output_n, activation='softmax'))
        self.compile(optimizer='sgd',loss='kullback_leibler_divergence')
        print(self.model.summary())

# =============================================================================
# Neural Nets for Q Learning
# =============================================================================

class Q_CNNet(BaseNetwork):

    def create_network(self):        
        
        inputs = layers.Input(shape=self.input_dim)
        
        conv1 = conv_block(inputs)
        conv2 = conv_block(inputs)
        flat = layers.Flatten()(layers.Add()([conv1,conv2]))        
        block1 = layers.Dense(64,activation='relu')(flat)
        
        
        conv1 = conv_block(inputs)
        conv2 = conv_block(inputs)        
        flat = layers.Flatten()(layers.Add()([conv1,conv2]))  
        block2 = layers.Dense(64,activation='relu')(layers.Dense(128,activation='relu')(flat))
        
        concat1 = layers.concatenate([block1, block2])
        block4 = layers.Dense(32,activation='relu')(concat1)
        
        conv1 = conv_block(inputs)
        conv2 = conv_block(inputs)        
        flat = layers.Flatten()(layers.Add()([conv1,conv2]))     
        block3 = layers.Dense(32,activation='relu')(layers.Dense(64,activation='relu')(flat))
        
        
        
        concat2 = layers.concatenate([block3,block4]) 
        
        
        output = layers.Dense(self.output_n,activation='linear')(concat2)
        
        self.model = keras.models.Model(inputs, output)
        self.model.compile(optimizer='sgd' ,loss ='mean_squared_error')
        print(self.model.summary())
    
    
class Q_FCNet(BaseNetwork):

    def create_network(self):
        
        
        inputs = layers.Input(shape=self.input_dim)
        block0 = layers.BatchNormalization()(inputs)
        block1 = conv_block(block0)
        x = layers.Flatten()(block1)
        x = layers.Dense(256,activation="softplus")(x)
        #x = layers.Dense(25,activation="relu")(x)
        x = layers.Dense(256,activation="relu")(x)
        #x = layers.Dense(128,activation="relu")(x)
        #x = layers.Dense(256,activation="relu")(x)
        x = layers.Dense(16,activation="relu")(x)
        #block1 = conv_block(inputs)
        #x = layers.Flatten()(block1)
        #x = layers.Dense(128,activation='softplus')(x)
        #x = layers.Dense(64,activation='relu')(x)
        
        outputs = layers.Dense(self.output_n,activation='linear')(x)
        
        self.model = keras.models.Model(inputs,outputs)
        self.model.compile(optimizer='adam',loss='mean_squared_error')
        print(self.model.summary())
        
        
        
class Q_RCNNet(BaseNetwork):

    def create_network(self):
        n_filters_1 = 16
        k_size_1 = 2
        stride_1 = 1
        
        self.model.add(layers.Conv2D(n_filters_1, k_size_1, strides=stride_1,
                               activation='relu', input_shape = self.input_dim))
        self.model.add(layers.ZeroPadding2D())
        self.model.add(layers.Conv2D(n_filters_1, k_size_1, strides=stride_1,
                               activation='relu'))
        self.model.add(layers.MaxPooling2d((2,2),strides=(2,2)))
        
        
        n_filters_2 = 128
        k_size_2 = 4
        stride_2 = 2

        self.model.add(layers.Conv2D(n_filters_2, k_size_2, strides=stride_2, 
                               activation='relu'))
        self.model.add(layers.ZeroPadding2D())        
        self.model.add(layers.Conv2D(n_filters_2, k_size_2, strides=stride_2, 
                               activation='relu'))
        self.model.add(layers.MaxPooling2d((2,2),strides=(2,2)))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512,activation='relu'))
        self.model.add(layers.Dense(128,activation='relu'))
        self.model.add(layers.Dense(self.output_n,activation='linear'))
        self.compile(optimizer='sgd',loss='mean_squared_error')
        print(self.model.summary())
        
def RCNN_layer(input_fitlers, output_filters):
    pass


def conv_block(inputs):
    n_filters_1 = 16
    k_size_1 = 3
    stride_1 = 2
        
    n_filters_2 = 32
    k_size_2 = 4
    stride_2 = 2
    
    #a = layers.ZeroPadding2D()(inputs)
    a = layers.Conv2D(n_filters_1, k_size_1, strides=stride_1,
                               activation='relu')(inputs)
    a = layers.Conv2D(n_filters_2, k_size_2, strides=stride_2, 
                               activation='relu')(a)
    

    return a
    
def vgg_block(inputs):
    a = layers.Conv2D(64, 3, activation='relu')(inputs)
    a = layers.Conv2D(64, 3, activation='relu')(a)
    a = layers.MaxPool2D()(a)
    a = layers.Conv2D(128, 3, activation='relu')(a)
    a = layers.Conv2D(128, 3, activation='relu')(a)
    a = layers.MaxPool2D()(a)
    a = layers.Flatten()(a)
    a = layers.Dense(256)(a)
    return a
