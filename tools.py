#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:43:53 2018

@author: thinkpad
"""

import numpy as np

def to_categorical(Y,n):
    _Y = np.zeros((len(Y),n))
    mY = int(min(Y))
    MY = int(max(Y))
    assert(MY-mY+1) == n
    for i,y in enumerate(Y):
        _Y[i,int(y)-mY]=1
    return _Y.astype(int)

