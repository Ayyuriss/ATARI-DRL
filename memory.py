#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:39:26 2018

@author: thinkpad
"""
from collections import deque

class Memory(deque):
    
    def __init__(self, max_size):
    
        super(Memory, self).__init__([], max_size)
