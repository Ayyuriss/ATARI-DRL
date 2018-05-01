#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:43:53 2018

@author: thinkpad
"""


import numpy as np
from skimage import transform
from scipy.misc import imshow
import scipy.signal


EPS = np.finfo(np.float64).tiny


def to_categorical(Y,n):
    _Y = np.zeros((len(Y),n))
    mY = int(min(Y))
    MY = int(max(Y))
    assert(MY-mY+1) == n
    for i,y in enumerate(Y):
        _Y[i,int(y)-mY]=1
    return _Y.astype(int)

def compute_advantage(vf, paths, gamma, lam):
    # Compute return, baseline, advantage
    for path in paths:
        path["return"] = discount(path["reward"], gamma)
        b = path["baseline"] = vf.predict(path)
        b1 = np.append(b, 0 if path["terminated"] else b[-1])
        deltas = path["reward"] + gamma*b1[1:] - b1[:-1] 
        path["advantage"] = discount(deltas, gamma * lam)
    alladv = np.concatenate([path["advantage"] for path in paths])    
    # Standardize advantage
    std = alladv.std()
    mean = alladv.mean()
    for path in paths:
        path["advantage"] = (path["advantage"] - mean) / std

def positify(y):
    _y = y.reshape(-1,3)
    mins = _y.min(axis=0)
    maxs = _y.max(axis=0)
    y2 = (y - mins)/(maxs-mins)
    return y2

def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.
    inputs
    ------
    x: ndarray
    gamma: float
    outputs
    -------
    y: ndarray with same shape as x, satisfying
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1
    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

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


# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:24:22 2018

@author: gamer
"""
    
    
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
    
def choice_weighted(pi):
#    np.random.seed(np.random.randint(0,2**10))
    #print(pi.shape)
    return np.random.choice(np.arange(len(pi)), 1, p=pi)[0]
        
        

            
def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(g, [numel(v)])for (v, g) in zip(var_list, grads)],0)
    
class GetFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list],0)

    def __call__(self):
        return self.op.eval(session=self.session)

def slice_2d(x, inds0, inds1):
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)


def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

class SetFromFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        shapes = list(map(var_shape, var_list))
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = tf.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []
        i = 0
        for v in var_list:
            size = np.prod(shapes[i])
            #assigns.append(tf.assign(v,tf.reshape(self.theta[start:start +size],shapes[i])))
            assigns.append(tf.assign(v,tf.reshape(self.theta[start:start +size],shapes[i])))
            i = i+1
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})
        
    
def write_dict(dic):

    fout = "./here.txt"
    fo = open(fout, "a+")
    fo.write('\n'+'-'*10+'\n')
    for k, v in dic.items():
        fo.write(str(k) + ' >>> '+ str(v) + '\n')
    fo.close()

def process_frame(img,size):
    return np.expand_dims(transform.resize(grayscale(img),size,mode='reflect'),axis=2)
def grayscale(frame):
    return (0.2989*frame[:,:, 0]
            + 0.5870*frame[:,:, 1] 
            + 0.1140*frame[:,:, 2])/255
    
def get_luminescence(frame):
	R = frame[:,:, 0]
	G = frame[:,:, 1]
	B = frame[:,:, 2]
	return (0.2126*R + 0.7152*G + 0.0722*B).astype(int)

def show(frame):
    imshow(frame)


def rollout(env, agent, len_episode):
    """
    Simulate the env and agent for timestep_limit steps
    """
    env.reset()
    state,_,start_lives = env.step(0)
    
    episode = {"state":[],"action":[],"reward":[],"terminated":[]}
    
    for _ in range(len_episode):
        episode["state"].append(state)
        action = agent.act(state)
        
        episode["action"].append(action)
        
        state, rew, done = env.step(action)


        episode["reward"].append(rew)        
        episode["terminated"].append(done)
        if done:
            break
    episode = {k:np.array(v) for (k,v) in episode.items()}
    return episode

def argmax(vect):
    mx = max(vect)
    idx = np.where(vect==mx)[0]
    return np.random.choice(idx)

def rollouts(env, agent, num_episodes, len_episode):
    print("Starting rollouts")    
    episodes = []
    for i in range(num_episodes):
        print("Rollout %d/%d"%(i,num_episodes))
        episodes.append(rollout(env, agent, len_episode))
    states = np.concatenate([episode["state"] for episode in episodes], axis = 0)
    actions = np.concatenate([episode["action"] for episode in episodes]).astype(int)
    rewards = np.concatenate([episode["reward"] for episode in episodes])
    terminated = np.concatenate([episode["terminated"] for episode in episodes])
    return {"states":states, "actions":actions, "rewards": rewards,"terminated":terminated}
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