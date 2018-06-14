#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 07:01:24 2018

@author: gamer
"""

import keras
import numpy as np
model = keras.models.load_model("./trained/learnedGRID0.1_DQN")



pos = []
V = []


for i in range(50):
    s = env.reset()
    done = False
    while not done:
        s,rew,done, _ = env.step(np.random.randint(4))
        np.argmax(model.predict(s.reshape((1,36,36,2))))
        if env.mouse_x!=28:
            V.append(np.max(model.predict(s.reshape((1,36,36,2)))))
            pos.append([env.x,env.y])
    
env.draw("hello")
X = np.array(pos)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = X[:,0]
y = X[:,1]
z = np.array(V)

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


x, y = np.meshgrid(np.unique(x), np.unique(y))
Z = np.zeros_like(x)+np.min(V)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        get = np.logical_and(X[:,0]==x[i,j],X[:,1]==y[i,j])
        Z[i,j] += np.sum(get*z)/max(1,np.sum(get))
        print(i,j)
        
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x, y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# load some test data for demonstration and plot a wireframe
ax.plot_wireframe(x, y, Z, rstride=5, cstride=5)

# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
    plt.show()