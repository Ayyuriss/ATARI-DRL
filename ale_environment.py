# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:59:20 2018

@author: gamer
"""
from ale_python_interface import ALEInterface
import utils.env as utils
import numpy as np
import collections

OPTIONS = {"IMAGES_SIZE":(80,80)}
CROP = {"breakout":(32,10,8,8)}

class ALE(ALEInterface):
    
    def __init__(self,game_name, num_frames= 4, skip_frames = 2, render=True):
        
        super(ALE, self).__init__()    
        
        self.crop = CROP[game_name]
        self.num_frames = num_frames
        self.skip_frames = skip_frames
        self.load_rom(game_name,render)
        self.load_params()
    
    def load_params(self):
        
        self._actions_raw = self.getMinimalActionSet().tolist()
        self._actions_n = len(self.actions_set)
        self._states_dim = OPTIONS["IMAGES_SIZE"]+(self.num_frames,)
        self._memory = collections.deque([],self.num_frames)
        self._start_lives = self.lives()
        self._current_state = np.zeros(self._states_dim) 
        
        while len(self._memory)<self.num_frames:
            self.capture_current_frame()
    def load_rom(self,rom_file,render):

        self.setInt(str.encode('random_seed'), 123)
        self.setFloat(str.encode('repeat_action_probability'), 0.0)        
        self.setBool(str.encode('sound'), False)
        self.setBool(str.encode('display_screen'), render)
        self.loadROM(str.encode("./roms/"+utils.game_name(rom_file)))
        
    def capture_current_frame(self):
        up,down,left,right = self.crop
        self._memory.append(utils.process_frame(
                self.getScreenRGB()[up:-down,left:-right],
                            OPTIONS["IMAGES_SIZE"])
                            )
    
    def get_current_state(self):
        
        return np.concatenate(self._memory,axis = -1)
    
    def step(self,action):
        
        
        reward = 0
        assert action in range(self.actions_n), "Action not available"
        
        for i in range(self.num_frames-1):
            reward = max(reward, self.act(self.actions_set[action]))
            
        
        state = self.get_current_state()
        
        return state, reward, self.lives() != self._start_lives

    def reset(self):
        self.reset_game()
        self.load_params()

    def clone(self):
        env = self.cloneSystemState()
        env.params()
        return env


    def act(self,action):

        res = 0
        for _ in range(self.skip_frames):
            res = max(res,super(ALE,self).act(action))

        self.capture_current_frame()

        return res


    @property
    def states_dim(self):
        return self._states_dim
    @property
    def actions_n(self):
        return self._actions_n
    @property
    def actions_set(self):
        return self._actions_raw

import skvideo
import skvideo.io
import skimage

class GRID(object):
    
    def __init__(self, grid_size=32, max_time=500):
        
        grid_size = grid_size+4

        self.grid_size = grid_size

        self.max_time = max_time


        #board on which one plays
        self.board = np.zeros((grid_size,grid_size))
        self.position = np.zeros((grid_size,grid_size))

        # coordinate of the cat
        self.x = 0
        self.y = 1


        # self time
        self.t = 0

        self.scale = self.grid_size

        self.to_draw = np.zeros((max_time+2, self.scale, self.scale,3))

        self.actions_n = 4
        self.states_dim = (self.scale,self.scale,3)
        
        self.reset()
        
    def draw(self,e):
        
        skvideo.io.vwrite(str(e) + '.mp4', self.to_draw)
    
    def draw_frame(self):
        
        skimage.io.imshow(self.to_draw[self.t])
        
    def get_frame(self,t):
        
        b = np.zeros((self.grid_size,self.grid_size,3))+64

        #Turns the mouse cell to red
        b[self.board>0,0] = 256
        
        #Turns the cat position to white
        b[self.x,self.y,:] = 256
        
        #Coloring the borders to black
        b[-2:,:,:]=0
        b[:,-2:,:]=0
        b[:2,:,:]=0
        b[:,:2,:]=0
        
        b =  skimage.transform.resize(b, (self.scale, self.scale,3),mode='reflect')

        #Storing the frame
        self.to_draw[t] = b/256#utils.process_frame(b,(self.scale, self.scale))


    def step(self, action):
        
        """This function returns the new state, reward and decides if the
        game ends."""

        self.get_frame(int(self.t))

        self.position = np.zeros((self.grid_size, self.grid_size))
        
        # forbid borders
        self.position[0:2,:]= -1
        self.position[:,0:2] = -1
        self.position[-2:, :] = -1
        self.position[-2:, :] = -1

        # set current position
        self.position[self.x, self.y] = 1
        
        if action == 0:
            if self.x == self.grid_size-3:
                self.x = self.x-1
            else:
                self.x = self.x + 1
        elif action == 1:
            if self.x == 2:
                self.x = self.x+1
            else:
                self.x = self.x-1
        elif action == 2:
            if self.y == self.grid_size - 3:
                self.y = self.y - 1
            else:
                self.y = self.y + 1
        elif action == 3:
            if self.y == 2:
                self.y = self.y + 1
            else:
                self.y = self.y - 1
        else:
            RuntimeError('Error: action not recognized')

        self.t = self.t + 1
        reward = self.board[self.x, self.y]
        self.board[self.x, self.y] = 0
        if reward ==1:
            self.add_mouse()
        game_over = self.t > self.max_time

        self.get_frame(self.t)
        return self.to_draw[self.t], reward, game_over

    def reset(self):

        """This function resets the game and returns the initial state"""

        self.x = np.random.randint(3, self.grid_size-3)
        self.y = np.random.randint(3, self.grid_size-3)


        self.add_mouse()

        self.to_draw = np.zeros((self.max_time+2, self.scale, self.scale,3))

        self.position = np.zeros((self.grid_size, self.grid_size))
        self.position[0:2,:]= -1
        self.position[:,0:2] = -1
        self.position[-2:, :] = -1
        self.position[-2:, :] = -1
        self.t = 0

        self.get_frame(0)
        
    def add_mouse(self):
        
        self.board*=0
        mouse_x,mouse_y = self.x,self.y
        while (mouse_x,mouse_y)==(self.x,self.y): 
            mouse_x = np.random.randint(3, self.grid_size-2)
            mouse_y = np.random.randint(3, self.grid_size-2)
        self.board[mouse_x, mouse_y]=1