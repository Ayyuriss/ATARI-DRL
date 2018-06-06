#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:27:24 2018

@author: thinkpad
"""

import skvideo
import skvideo.io
import skimage
import numpy as np
import sys

sys.path.append("../")

from base.spaces import Discrete, Continuous

class GRID(object):
    
    def __init__(self, grid_size=16, max_time=500, square_size = 2, stochastic=True):
        
        self.name = "GRID"
        
        
        self.max_time = max_time
        
        self.grid_size = grid_size
        self.square = square_size
        self.step = int(np.ceil(self.square/2))
        self.stochastic = stochastic
        self.board = np.zeros((self.grid_size,self.grid_size))

        # recording states
        self.to_draw = np.zeros((max_time+2, self.grid_size, self.grid_size,1))
        
        
        self.action_space = Discrete(4)
        
        self.state_space = Continuous(-1,1,(self.grid_size, self.grid_size,2))

        self.reset()
        
    def draw(self,file):
        
        video = np.zeros((len(self.to_draw),self.grid_size, self.grid_size,3)).astype('uint8')
        
        #Turns the mouse cell to red        
        video[self.to_draw[:,:,:,0]>0,0] = 255

        #Turns the cat position to white
        video[self.to_draw[:,:,:,0]<0,:] = 255

        skvideo.io.vwrite(file+ '.mp4', video,inputdict={'-r': '25'},outputdict={'-vcodec': 'libx264',
                                                                                              '-pix_fmt': 'yuv420p',
                                                                                             '-r': '25'})
    
    def draw_frame(self):
        
        skimage.io.imshow(self.to_draw[self.t])
        
    def get_frame(self):
        
        self.to_draw[self.t][self.board>0,0] = 1      

        self.to_draw[self.t][self.x:min(self.grid_size,self.x+self.square),self.y:min(self.grid_size,self.y+self.square),0] = -1
        
    def step(self, action):
        
        """This function returns the new state, reward and decides if the
        game ends."""

 

        reward = 0
        # clear current position


        if action == 0:
            if self.x >= self.grid_size-self.square-1:
                self.x = self.x - self.step
                reward += -1 
            else:
                self.x = self.x + self.step
        elif action == 1:
            if self.x <= self.step:
                self.x = self.x + self.step
                reward += -1 
            else:
                self.x = self.x - self.step
        elif action == 2:
            if self.y >= self.grid_size - self.square- 1:
                reward += -1 
                self.y = self.y - self.step
            else:
                self.y = self.y + self.step
        elif action == 3:
            if self.y <= self.step:
                reward += -1 
                self.y = self.y + self.step
            else:
                self.y = self.y - self.step
        else:
            RuntimeError('Error: action not recognized')

        self.t = self.t + 1

        self.get_frame()
        
        reward = reward + np.max(self.board[self.x:min(self.grid_size,self.x+self.square),self.y:min(self.grid_size,self.y+self.square)])

        game_over = self.t > self.max_time
        
        if reward ==1:
             #game_over = True
            print("mouse found")    
            self.add_mouse()

        return self.current_state(), reward, game_over

    def reset(self):

        """This function resets the game and returns the initial state"""
        
        self.start = True
        self.t = 0
        self.board *= 0
        self.to_draw *= 0
        self.x = self.square#np.random.randint(0, self.grid_size)
        self.y = self.square#np.random.randint(0, self.grid_size)

        self.add_mouse()
        self.get_frame()
        
        return self.current_state()
        
    def add_mouse(self):
        
        self.board*=0
        mouse_x,mouse_y = self.x,self.y
        if self.stochastic:
            while (mouse_x,mouse_y)==(self.x,self.y): 
                mouse_x = np.random.randint(self.square, self.grid_size-self.square)
                mouse_y = np.random.randint(self.square, self.grid_size-self.square)
        else:

            mouse_x = self.square+self.start*(self.grid_size-3*self.square)
            mouse_y = self.square+self.start*(self.grid_size-3*self.square)
            self.start = not self.start

                    
        self.board[mouse_x:min(self.grid_size,mouse_x+self.square), mouse_y:min(self.grid_size,mouse_y+self.square)] = 1
        
    def current_state(self):
        
        return np.concatenate([self.to_draw[self.t-1],self.to_draw[self.t]],axis=-1)
