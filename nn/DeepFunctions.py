# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:07:46 2018

@author: gamer
"""
import sys,os
import numpy as np
import keras
import keras.backend as K
from keras import layers

sys.path.append(os.path.dirname(os.getcwd()))

from utils.console import Progbar
from nn import reducer

# ================================================================
# Base class for Q and deep Policy
# ================================================================
class BaseDeep(object):
    name = "Base"
    def __init__(self,env):
        
        self.input_dim = env.observation_space.shape
        self.output_n = env.action_space.n
        self.net = keras.models.Sequential()
        
        self.setup_model()

    def setup_model(self):
        raise (NotImplementedError, self.network_type)

    def fit(self,X,Y,batch_size=50):
        
        print("Fitting the NN:",X.shape, Y.shape)
        self.model.fit(X,Y,batch_size,1)

    def train_on_batch(self,X,Y):
        self.net.train_on_batch(X,Y)
        
    def zero_initializer(self):
        for x in self.trainable_variables:            
            K.set_value(x, np.zeros(x.shape))

    def reduce_weights(self,factor):
        for x in self.trainable_variables:            
            K.set_value(x, K.eval(x)/factor)
        
    def predict(self,image):
        
        if image.ndim == len(self.input_dim):
            _image = image.reshape((1,)+image.shape)
            return self.net.predict(_image)[0]
        else:
            return self.net.predict(image)

    def save(self,name):
        self.net.save(name+self.name)

    def load(self,name):
        self.net = keras.models.load_model(name)
        
    @property
    def trainable_variables(self):
        return self.net.trainable_weights
    
    @property
    def input(self):
        return self.net.input
    
    @property
    def output(self):
        return self.net.output


        
# ================================================================
# Object class for Q and policy
# ================================================================

"""
class DeepPolicy(BaseDeep):
    
    def setup_model(self):
        
        assert self.network_type in ['FC','CNN']
            
        if self.network_type =='FC':
            self.net = NeuralNets.Policy_FCNet(self.observation_dim, self.actions_n)
        else:
            self.net = NeuralNets.Policy_CNNet(self.observation_dim, self.actions_n)
"""   
class DeepPolicy(BaseDeep):

    def __init__(self,env):        
        self.name = "policy"
        super(DeepPolicy,self).__init__(env)
        #self.reduce_weights(5)
    def setup_model(self):
        inputs = layers.Input(shape=self.input_dim)
        block0 = layers.BatchNormalization()(inputs)
        block1 = conv_block(block0)
        x = layers.Flatten()(block1)
        x = layers.Dense(256,activation="relu")(x)
         
        outputs = layers.Dense(self.output_n,activation='softmax')(x)
        self.net = keras.models.Model(inputs, outputs)
        optim = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.net.compile(optimizer=optim,loss='mse')
        #self.reducer.compile(self.model)
        print(self.net.summary())

class DeepPolicy2(BaseDeep):
    def __init__(self,env):        
        self.name = "dqn"
        super(DeepPolicy2,self).__init__(env)
    def setup_model(self):
        
        inputs = layers.Input(shape=self.input_dim)
        block0 = layers.BatchNormalization()(inputs)
        x = layers.Flatten()(block0)
        x = layers.Dense(64,activation="relu")(x)
        x = layers.Dense(64,activation="relu")(x)
        x = layers.Dense(64,activation="relu")(x)

        outputs = layers.Dense(self.output_n,activation='softmax')(x)
        self.net = keras.models.Model(inputs, outputs)        
        optim = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.net.compile(optimizer=optim,loss='mse')
        #self.reducer.compile(self.model)
        print(self.net.summary())
   
        
class DeepQ(BaseDeep):
    def __init__(self,env):        
        self.name = "dqn"
        super(DeepQ,self).__init__(env)
    def setup_model(self):
        
        inputs = layers.Input(shape=self.input_dim)
        block0 = layers.BatchNormalization()(inputs)
        block1 = conv_block(block0)
        #self.reducer = reducer.ReductionLayer(6,64,0.0001)
        #block1 = self.reducer(block0)
        #block2 = conv_block(block1)
        x = layers.Flatten()(block1)
        x = layers.Dense(256,activation="relu")(x)
        #x = layers.Dense(25,activation="relu")(x)
        #x = layers.Dense(64,activation="relu")(x)
        #x = layers.Dense(128,activation="relu")(x)
        #x = layers.Dense(256,activation="relu")(x)
        #x = layers.Dense(64,activation="relu")(x)
        #block1 = conv_block(inputs)
        #x = layers.Flatten()(block1)
        #x = layers.Dense(128,activation='softplus')(x)
        #x = layers.Dense(64,activation='relu')(x)
         
        outputs = layers.Dense(self.output_n,activation='linear')(x)
        self.net = keras.models.Model(inputs, outputs)        
        optim = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.net.compile(optimizer=optim,loss='mse')
        #self.reducer.compile(self.model)
        print(self.net.summary())

class DeepQ2(BaseDeep):
    def __init__(self,env):        
        self.name = "dqn2"
        super(DeepQ2,self).__init__(env)
    def setup_model(self):
        
        inputs = layers.Input(shape=self.input_dim)
        x = layers.Flatten()(inputs)
        x = layers.Dense(128,activation="softplus")(x)
        x = layers.Dense(128,activation="softplus")(x)
        x = layers.Dense(128,activation="softplus")(x)
        x = layers.Dense(128,activation="softplus")(x)
        x = layers.Dense(64,activation="relu")(x)

        outputs = layers.Dense(self.output_n,activation='linear')(x)
        self.net = keras.models.Model(inputs, outputs)        
        optim = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.net.compile(optimizer=optim,loss='mse')
        #self.reducer.compile(self.model)
        print(self.net.summary())
        
            

            
# ================================================================
# Value Function for baseline
# ================================================================
class BaselineValueFunction(BaseDeep):

    def __init__(self,env):
        self.name = "baseline"
        self.input_dim = np.prod(env.observation_space.shape) + env.action_space.n +2
        self.output_n = 1
        self.net = keras.models.Sequential()
        self.setup_model()
        
    def setup_model(self):
        self.net.add(layers.Dense(self.output_n, input_shape = (self.input_dim,),activation='linear'))
        self.net.compile(optimizer = 'adam',loss='mean_squared_error')
        print("Baseline Value Function\n", self.net.summary())
    
    def _features(self, episode):
        states = episode["state"].astype('float32')
        states = states.reshape(len(states),-1)
        proba = episode["output"].astype('float32')
        n = len(episode["reward"])
        al = episode["t"].reshape(-1,1).astype('float32')/10
        ret = np.concatenate([states, proba, al, np.ones((n, 1))], axis=1)
        return ret

    def fit(self, episodes):
        featmat = np.concatenate([self._features(episode) for episode in episodes])
        returns = np.concatenate([episode["return"] for episode in episodes])
        self.net.fit(featmat,returns,batch_size=10,epochs=2)

    def predict(self, episode):
        return self.net.predict(self._features(episode))


def conv_block(inputs):
    n_filters_1 = 16
    k_size_1 = 8
    stride_1 = 4
        
    n_filters_2 = 32
    k_size_2 = 4
    stride_2 = 2
    
    #a = layers.ZeroPadding2D()(inputs)
    a = layers.Conv2D(n_filters_1, k_size_1, strides=stride_1,
                               activation='relu')(inputs)
    a = layers.Conv2D(n_filters_2, k_size_2, strides=stride_2, 
                               activation='relu')(a)

    return a
