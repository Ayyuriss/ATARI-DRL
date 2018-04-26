# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:59:20 2018

@author: gamer
"""
from ale_python_interface import ALEInterface
import tools


class ALE(ALEInterface):
    
    def __init__(self,rom_file, skip_frame = 2, display=True):
        
        super(ALE, self).__init__()    
        
        self.skip_frame = skip_frame
        self.load_rom(rom_file,display)
        self.params()
        
    def params(self):
        self.actions_raw = self.getMinimalActionSet().tolist()
        self.actions_n = len(self.actions_raw)

        self._states_dim = self.getScreenDims()
    
    def load_rom(self,rom_file,display):

        self.setInt(str.encode('random_seed'), 123)
        self.setFloat(str.encode('repeat_action_probability'), 0.0)        
        self.setBool(str.encode('sound'), False)
        self.setBool(str.encode('display_screen'), display)
        self.loadROM(str.encode("./roms/"+rom_file))
        
 
    def act(self,action):

        reward = 0
        assert action in range(self.actions_n), "Action not available"
        for _ in range(self.skip_frame):
            reward = max(reward, super(ALE, self).act(self.actions_raw[action]))
        state = tools.grayscale(self.getScreenRGB())
        
        return reward, state, self.lives()

    def reset(self):
        self.reset_game()

    def clone(self):
        env = self.cloneSystemState()
        env.params()
        return env
    
    @property
    def states_dim(self):
        return self._states_dim
