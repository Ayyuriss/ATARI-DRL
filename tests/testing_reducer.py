#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:06:54 2018

@author: thinkpad
"""
import keras
import keras.backend as K


inputs = keras.layers.Input(shape=(64,64,1))
reducer = ReductionLayer(16,24,0.001)
x = reducer(inputs)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1)(x)

model = keras.models.Model(inputs,x)


model.compile("adam","mse")

model.summary()


X = np.random.normal(0,1,(1000,64,64,1))
Y = np.mean(X,axis=(1,2))
W = np.random.normal(0,1,(1000,64,64,1))
Z = np.mean(W,axis=(1,2))


reducer.compile(model)

model.fit(X,Y)
reducer.fit(X,Y)
print(reducer.display_update())
model.evaluate(W,Z)


targets = K.Function(model.targets,model.targets)
output = K.Function([model.input],[model.output])
W = output([X])
Z = targets([Y])
loss = K.Function([model.input,model.targets[0],model.sample_weights[0]],[model.total_loss])
sess.run(model.total_loss,feed_dict={model.input:X,model.targets[0]:[y for y in Y]})
model.fit(X,Y,batch_size=len(X))
f
K.eval()
loss([X,Y,np.ones(len(X))])

model.loss_functions()
from keras.utils import plot_model
plot_model(model, to_file='model.png')


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

