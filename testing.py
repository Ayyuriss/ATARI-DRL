#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:31:42 2018

@author: thinkpad
"""

class Ayoub(object):
    
    def __init__(self):
        
        self.display()
        
    def display(self):
        print("Ayoub")
        
        
class Ghriss(Ayoub):
    
    def __ini__(self):
        
        super(Ghriss, self).__init__()
    
    def display(self):
        print("Ghriss")

