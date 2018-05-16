#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:28:01 2018

@author: thinkpad
"""

from skimage import transform
from skimage import io
from scipy.misc import imshow
import numpy as np

def process_frame(img,size):
    return np.floor(1000*np.clip(np.expand_dims(transform.resize(grayscale(img),size,mode='reflect'),axis=2),0,1))/1000

def grayscale(frame):
    return (0.2989*frame[:,:, 0] + 0.5870*frame[:,:, 1]
                                 + 0.1140*frame[:,:, 2])/255
    
def get_luminescence(frame):
	R = frame[:,:, 0]
	G = frame[:,:, 1]
	B = frame[:,:, 2]
	return (0.2126*R + 0.7152*G + 0.0722*B).astype(int)

def show(frame):
    io.imshow(frame)
    
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
        
def stack_show(filter_mat, n_col):
    
    shape = filter_mat.shape
    cols = n_col
    rows = shape[-1]//cols
    
    screen = np.zeros((shape[0]*rows,shape[1]*cols))
    
    for i in range(rows):
        for j in range(cols):
            screen[shape[0]*i:shape[0]*(i+1),shape[1]*j:shape[1]*(j+1)] = filter_mat[:,:,i*cols+j]
    show(screen)
    #return screen
