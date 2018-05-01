#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:28:01 2018

@author: thinkpad
"""

from skimage import transform
from scipy.misc import imshow
import numpy as np

def process_frame(img,size):
    return np.expand_dims(transform.resize(grayscale(img),size,mode='reflect'),axis=2)

def grayscale(frame):
    return (0.2989*frame[:,:, 0] + 0.5870*frame[:,:, 1]
                                 + 0.1140*frame[:,:, 2])/255
    
def get_luminescence(frame):
	R = frame[:,:, 0]
	G = frame[:,:, 1]
	B = frame[:,:, 2]
	return (0.2126*R + 0.7152*G + 0.0722*B).astype(int)

def show(frame):
    imshow(frame)
    
def game_name(name):
    idx = name.find(".")
    if idx==-1:
        return name+".bin"
    else:
        if name[idx:]=='.bin':
            return name
        else:
            raise(NameError,name)
            return ""