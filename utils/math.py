#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:44:56 2018

@author: thinkpad
"""

import numpy as np


def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)
    print("fval before", fval)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac*fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate*stepfrac
        ratio = actual_improve/expected_improve
        print("a/e/r:", actual_improve, expected_improve, ratio)
        if ratio > accept_ratio and actual_improve > 0:
            print("fval after:", newfval)
            return True, xnew
    return False, x


def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print(fmtstr % (i+1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x



def line_search(f, x, fullstep, expected_improve_rate, LN_ACCEPT_RATE):
    """ We perform the line search in direction of fullstep, we shrink the step 
        exponentially (multi by beta**n) until the objective improves.
        Without this line search, the algorithm occasionally computes
        large steps that cause a catastrophic degradation of performance

        f : callable , function to improve    
        x : starting evaluation    
        fullstep : the maximal value of the step length
        expected_improve_rate : stop if 
                    improvement_at_step_n/(expected_improve_rate*beta**n)>0.1
    """
    
    accept_ratio = LN_ACCEPT_RATE
    max_backtracks = 10
    fval = f(x)
    stepfrac=1
    stepfrac=stepfrac*0.5
    for stepfrac in .5**np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew

    return x


def conjugate_gradient(f_Ax, b, n_iters=10, gtol=1e-10):
    """Search for Ax-b=0 solution using conjugate gradient algorithm
       
        f_Ax : callable, f(x, *args) (returns A.dot(x) with A Symetric Definite)
        b : b such we search for Ax=b
        cg_iter : max number of iterations
        gtol: iterations stop when norm(residual) < gtol
    """
    
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for _ in range(n_iters):
        if rdotr < gtol:
            break
        z = f_Ax(p)
        alpha = rdotr / p.dot(z)
        x += alpha * p
        r -= alpha * z
        newrdotr = r.dot(r)
        beta = newrdotr / rdotr
        p = r + beta * p
        rdotr = newrdotr
        
    return x

def argmax(vect):
    mx = max(vect)
    idx = np.where(vect==mx)[0]
    return np.random.choice(idx)