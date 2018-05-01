# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:59:20 2018

@author: gamer
"""
from ale_python_interface import ALEInterface
import rl_tools
import numpy as np

OPTIONS = {"IMAGES_SIZE":(84,84)}
class ALE(ALEInterface):
    
    def __init__(self,rom_file, num_steps= 4, skip_frame = 1, render=True):
        
        super(ALE, self).__init__()    
        
        self.num_steps = num_steps
        self.skip_frame = skip_frame
        self.load_rom(rom_file,render)
        self.params()
        self.save_current_frame()
    
    def params(self):
        
        self._actions_raw = self.getMinimalActionSet().tolist()
        self._actions_n = len(self.actions_set)
        self._states_dim = OPTIONS["IMAGES_SIZE"]+(self.num_steps,)
        self._start_lives = self.lives()
        self._current_state = np.zeros(self._states_dim) 
    
    def load_rom(self,rom_file,render):

        self.setInt(str.encode('random_seed'), 123)
        self.setFloat(str.encode('repeat_action_probability'), 0.0)        
        self.setBool(str.encode('sound'), False)
        self.setBool(str.encode('display_screen'), render)
        self.loadROM(str.encode("./roms/"+rl_tools.game_name(rom_file)))
        
    def save_current_frame(self):
        self._current_frame = rl_tools.process_frame(self.getScreenRGB(),OPTIONS["IMAGES_SIZE"])
    
    def get_current_state(self):
        new_frame = rl_tools.process_frame(self.getScreenRGB(),OPTIONS["IMAGES_SIZE"])
        return np.concatenate([self._current_frame,new_frame],axis = -1)
    
    def step(self,action):
 
        reward = 0
        assert action in range(self.actions_n), "Action not available"
        for i in range(self.num_steps*self.skip_frame):
            self.save_current_frame()            
            reward = max(reward, self.act(self.actions_set[action]))
            if not (i+1)%self.skip_frame:
                self._current_state[:,:,(i+1)//self.skip_frame-1] = self._current_frame[:,:,0]
        state = self._current_state
        
        return state, reward, self.lives() != self._start_lives

    def reset(self):
        self.reset_game()

    def clone(self):
        env = self.cloneSystemState()
        env.params()
        return env
    
    @property
    def states_dim(self):
        return self._states_dim
    @property
    def actions_n(self):
        return self._actions_n
    @property
    def actions_set(self):
        return self._actions_raw
