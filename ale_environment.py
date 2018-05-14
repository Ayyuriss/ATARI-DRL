# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:59:20 2018

@author: gamer
"""
from ale_python_interface import ALEInterface
import utils.env as utils
import numpy as np
import collections

OPTIONS = {"IMAGES_SIZE":(84,84)}
CROP = {"breakout":(30,10,6,6)}
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
