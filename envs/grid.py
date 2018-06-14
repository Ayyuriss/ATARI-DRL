#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:27:24 2018

@author: thinkpad
"""

import skvideo
import skvideo.io
import skimage.io
import numpy as np
import sys
import gym.spaces
sys.path.append("../")
PLAY_PATH="./plays/"

from base.spaces import Discrete, Continuous

class GRID(object):
    name = "GRID"
    def __init__(self, grid_size=16, max_time=500, square_size = 2, stochastic=True):
        self.max_time = max_time
        
        self.grid_size = grid_size
        self.square = square_size
        self.move_step = int(np.ceil(self.square/2))
        self.stochastic = stochastic
        self.board = np.zeros((self.grid_size,self.grid_size))

        self.to_draw = np.zeros((max_time+2, self.grid_size, self.grid_size,3)).astype(int)
        
        self.action_space = gym.spaces.Discrete(4)
        
        self.observation_space = gym.spaces.Box(low=-1,high=1,shape=(self.grid_size,self.grid_size,3),dtype=np.int32)#Continuous((self.grid_size,self.grid_size,1))
	
        self.reset()
        
    def draw(self,file):
        
        skvideo.io.vwrite(PLAY_PATH+file+ '.mp4', self.to_draw,inputdict={'-r': '25'},outputdict={'-vcodec': 'libx264',
                                                                                              '-pix_fmt': 'yuv420p',
                                                                                             '-r': '25'})
    def draw_frame(self):
        
        skimage.io.imshow(self.to_draw[self.t].astype(float)/255)
        
    def get_frame(self):
        
        self.to_draw[self.t][self.board>0,0] = 255     

        self.to_draw[self.t][self.x:min(self.grid_size,self.x+self.square),self.y:min(self.grid_size,self.y+self.square),:] = 255
        
    def step(self, action):
        
        """This function returns the new state, reward and decides if the
        game ends."""

 
        reward = 0
        # clear current position

        found = False
        if action == 0:
            if self.x >= self.grid_size-self.square-1:
                self.x = self.x - self.move_step
                reward += -1 
            else:
                self.x = self.x + self.move_step
        elif action == 1:
            if self.x <= self.move_step:
                self.x = self.x + self.move_step
                reward += -1 
            else:
                self.x = self.x - self.move_step
        elif action == 2:
            if self.y >= self.grid_size - self.square- 1:
                reward += -1 
                self.y = self.y - self.move_step
            else:
                self.y = self.y + self.move_step
        elif action == 3:
            if self.y <= self.move_step:
                reward += -1 
                self.y = self.y + self.move_step
            else:
                self.y = self.y - self.move_step
        else:
            RuntimeError('Error: action not recognized')

        self.t = self.t + 1

        self.get_frame()
        
        reward = reward + np.max(self.board[self.x:min(self.grid_size,self.x+self.square),self.y:min(self.grid_size,self.y+self.square)])

        game_over = self.t > self.max_time-1
        
        if game_over:
            return self.current_state(), reward, game_over, found
        
        if reward ==1:
             #game_over = True
            found = True
            self.add_mouse()
        return self.current_state(), reward, game_over, found
        

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
        
        self.board *= 0
        self.mouse_x,self.mouse_y = self.x,self.y

        if self.stochastic:
            while (self.mouse_x,self.mouse_y)==(self.x,self.y): 
                self.mouse_x = np.random.randint(self.square, self.grid_size-self.square)
                self.mouse_y = np.random.randint(self.square, self.grid_size-self.square)
        else:

            self.mouse_x = self.square+self.start*(self.grid_size-3*self.square)
            self.mouse_y = self.square+self.start*(self.grid_size-3*self.square)
            self.start = not self.start

                    
        self.board[self.mouse_x:min(self.grid_size,self.mouse_x+self.square), self.mouse_y:min(self.grid_size,self.mouse_y+self.square)] = 1
        self.t = self.t + 1
        self.get_frame()
    def current_state(self):

        return self.to_draw[self.t]
    
    def get_mouse(self):
        return np.array([self.mouse_x,self.mouse_y])/self.grid_size

    def get_cat(self):
        return np.array([self.x,self.y])/self.grid_size

class GRID2(GRID):
    name = "GRID2"
    def __init__(self,*args,**kwargs):
        super(GRID2,self).__init__(*args,**kwargs)
        self.observation_space = Continuous((2,2,1))
    
    def current_state(self):
        
        return np.array([self.x, self.y, self.mouse_x, self.mouse_y]).reshape(2,2,1)
    
