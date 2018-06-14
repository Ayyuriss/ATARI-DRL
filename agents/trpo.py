#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:02:18 2018

@author: thinkpad
"""
import sys
import numpy as np
import keras.backend as K
import scipy.signal

sys.path.append("../")

import utils.math as m_utils
import utils.agent as utils
from utils.console import Progbar

from nn import deepfunctions
from base.agent import Agent

EPS = np.finfo(np.float32).tiny

class TRPO(Agent):
    
    options = {"cg_damping": (1e-1, "Add multiple of the identity to Fisher matrix during CG"),
        "max_kl": (1e-2, "KL divergence between old and new policy (averaged over state-space)"),
	"linesearch_accept": (1e-1, "Lineseach accept ratio")
    }
    deep = deepfunctions.DeepPolicy

    def __init__(self, env, gamma, max_steps):
        self.agent_type = "TRPO"          
        policy = self.deep(env)
        self.old_policy = self.deep(env)
        super(TRPO, self).__init__(policy)
    

        self.discount = gamma
        self.env = env
        self.max_steps = max_steps

        self.setup_agent()
        
        self.baseline = deepfunctions.BaselineValueFunction(env)
        self.episodes = []
        self.progbar = Progbar(100)
        
    def setup_agent(self):
        
        self.states = self.model.input
        self.actions = K.placeholder(ndim=1, dtype = 'int32')
        self.advantages = K.placeholder(ndim=1)
        current_pi = self.model.output
        
        old_pi = self.old_policy(self.states)

        log_likeli_pi = utils.loglikelihood(self.actions, current_pi)
        log_likeli_old_pi = utils.loglikelihood(self.actions, old_pi)

        N = K.cast(K.shape(self.states)[0],dtype='float32')

        # Policy gradient:

        surrogate_loss = (-1.0 / N) * K.sum( K.exp(log_likeli_pi - log_likeli_old_pi)*self.advantages)
        
        policy_gradient = self.model.flattener.flatgrad(self.model.output)
        
        kl_firstfixed = K.mean(utils.entropy(current_pi))
        
        grads = self.model.flattener.flatgrad(kl_firstfixed)
        
        flat_tangent = K.placeholder(ndim=1)
        
        grad_vector_product = K.sum(grads*flat_tangent)
        
        
        # Fisher-vector product
        
        fisher_vector_product = self.model.flattener.flatgrad(grad_vector_product)
        
        entropy = K.mean(utils.entropy(current_pi))
        

        losses = [surrogate_loss, kl_firstfixed, entropy]
        
        self.loss_names = ["Surrogate", "KL", "Entropy"]

        args = [self.states, self.actions, self.advantages]

        self.compute_policy_gradient = K.function(args, [policy_gradient])
        self.compute_losses = K.function(args, losses)
        self.compute_fisher_vector_product = K.function([flat_tangent] + args, [fisher_vector_product])

    def train(self):

        self.rollout()
        
        states = np.concatenate([episode["state"] for episode in self.episodes],axis=0)
        actions = np.concatenate([episode["action"] for episode in self.episodes],axis=0)
        advantages = np.concatenate([episode["advantage"] for episode in self.episodes],axis=0)
        
        args = (states, actions, advantages)
        
        thprev = self.model.flattener.get_value()
        self.old_policy.flattener.set_value(thprev)
        
        g = self.compute_policy_gradient([*args])[0]
        
        losses_before = self.compute_losses([*args])
        
        if np.allclose(g, 0):
            print("got zero gradient. not updating")
        else:
            print("Using Conjugate gradient")
            stepdir = m_utils.conjugate_gradient(lambda x : self.fisher_vector_product(x,args), -g)
            shs = .5*stepdir.dot(self.fisher_vector_product(stepdir,args))
            lm = np.sqrt(shs / self.options["max_kl"][0])
            
            print("Lagrange multiplier:", lm, "norm(g):", np.linalg.norm(g))
            
            fullstep = stepdir / lm
            def loss(th):
                self.model.flattener.set_value(th)
                return self.compute_losses([*args])[0]
            
            success, theta = m_utils.linesearch(loss, thprev, fullstep)
            print("Line-Search Success", success)
            self.model.flattener.set_value(theta)
        losses_after = self.compute_losses([*args])

        for (lname, lbefore, lafter) in zip(self.loss_names, losses_before, losses_after):
            self.log(lname+"_before", lbefore)
            self.log(lname+"_after", lafter)
        self.print_log()
        self.model.save(self.env.name)
    
    def act(self,state,train=False):
        
        proba = self.model.predict(state)
        if train:
            action = utils.choice_weighted(proba)
        else:
            action = np.argmax(proba)
        return action

    def fisher_vector_product(self,p,args):
            return self.compute_fisher_vector_product([p]+[*args])[0]+self.options["cg_damping"][0]*p

    def rollout(self):

        self.episodes = []
        self.collected = 0
        self.progbar.__init__(self.max_steps)

        while self.collected < self.max_steps:
            self.get_episode()
            
        self.compute_advantage()
        self.baseline.fit(self.episodes)

    def get_episode(self):
        
        state = self.env.reset()
        
        episode = {s : [] for s in ["t","state","action","reward","terminated"]}
        
        i = 0
        
        while self.collected < self.max_steps:
                        
            
            
            episode["t"].append(i)
            episode["state"].append(state)
            # act
            action = self.act(state,train=True)
            
            state, rew, done, info = self.env.step(action)
            
            episode["action"].append(action)
            episode["reward"].append(rew)        
            episode["terminated"].append(done)
            
            i += 1
            self.collected +=1
            self.progbar.add(1,values=[('Info',info)])
            if done:
                break
        for k,v in episode.items():
            episode[k] = np.array(v)
        episode["return"] = discount(np.array(episode["reward"]), self.discount)
        
        self.episodes.append(episode)

    def compute_advantage(self):

        # Compute baseline, advantage
        for episode in self.episodes:
            b = episode["baseline"] = self.baseline.predict(episode)
            b1 = np.append(b, 0 if episode["terminated"][-1] else b[-1])
            deltas = episode["reward"] + self.discount*b1[1:] - b1[:-1] 
            episode["advantage"] = discount(deltas, self.discount)

        alladv = np.concatenate([episode["advantage"] for episode in self.episodes])    
        # Standardize advantage
        std = alladv.std()
        mean = alladv.mean()
        for episode in self.episodes:
            episode["advantage"] = (episode["advantage"] - mean) / std

    def play(self,name='play'):
        
        state = self.env.reset()
        done = False
        
        while not done:
            action = self.act(state)
            state, _, done,_ = self.env.step(action)
        
        self.env.draw(name)
        
class TRPO2(TRPO):
    deep= deepfunctions.DeepPolicy2

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]        


def normalize(v):
    norm = np.linalg.norm(v)
    v_tmp = v-np.mean(v)
    if norm < EPS: 
       return v_tmp
    return v_tmp/np.max(np.abs(v_tmp))


