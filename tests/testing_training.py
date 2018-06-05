#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:20:15 2018

@author: thinkpad
"""

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Fixed
batch = 4000

alpha = np.random.normal(0,0.5,100)
X_tr = np.random.normal(0,0.5,(batch,))
Z_tr = np.mean(np.exp(-X_tr.reshape(-1,1)**2*alpha),axis=1)
# sort
idx = np.argsort(X_tr)
X_tr=X_tr[idx]
Z_tr=Z_tr[idx]

# Learned
coefs = K.random_normal_variable((100,),0,0.1)
pows = K.arange(100,dtype='float32')
X = K.placeholder((None,1))
Y = K.mean(K.exp(-K.square(X)*coefs),axis=1)
Z = K.placeholder(ndim=1)

loss = K.mean(K.square(Y-Z))

optimizer = tf.train.AdamOptimizer()
opt = optimizer.minimize(loss,var_list=[coefs])
grad = K.Function([X,Z],[opt])

get_loss = K.Function([X,Z],[loss])
get_Y = K.Function([X],[Y])

for i in range(1000):
    grad([X_tr.reshape(-1,1),Z_tr])
    if not i%100:
        print(get_loss([X_tr.reshape(-1,1),Z_tr]))

        plt.plot(X_tr,Z_tr)
        plt.plot(X_tr,get_Y([X_tr.reshape(-1,1)])[0])
        plt.show()

# Change Z_tt
Z_tt = Z_tr.copy()
for i in range(4000):
    Z_tt[i] = Z_tr[i]*(1+np.exp(-X_tr[i]**2*7))
plt.plot(X_tr,Z_tr)
plt.plot(X_tr,Z_tt)
plt.plot(X_new,Z_new)
plt.show()

to_keep=np.abs(X_tr)<1
X_new = X_tr
Z_new = Z_tt.copy()
for i in range(len(to_keep)):
    Z_new[i] = Z_tt[i]*to_keep[i]+(1-to_keep[i])*Z_tr[i]


for i in range(1000):
    grad([X_new.reshape(-1,1),Z_new])
    if not i%100:
        print(get_loss([X_new.reshape(-1,1),Z_new]),get_loss([X_tr.reshape(-1,1),Z_tr]))
        plt.plot(X_tr,Z_tr)
        plt.plot(X_tr,get_Y([X_tr.reshape(-1,1)])[0])
        plt.plot(X_new,get_Y([X_new.reshape(-1,1)])[0])
        plt.show()
        
plt.plot(X_new,Z_new)
plt.plot(X_new,get_Y([X_new.reshape(-1,1)])[0])
plt.show()

plt.plot(X_new[:,0],get_Y([X_new])[0])
plt.plot(X_tr[:,0],get_Y([X_tr])[0])
