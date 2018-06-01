#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:02:18 2018

@author: thinkpad
"""

import utils.math as m_utils
import DeepFunctions
import keras.backend as K
import utils.agent as utils
import numpy as np
import scipy.signal
from utils.console import Progbar
EPS = np.finfo(np.float32).tiny

from base_classes.agent import Agent

class TRPO(Agent):
    
    options = [
        ("cg_damping", float, 1e-3, "Add multiple of the identity to Fisher matrix during CG"),
        ("max_kl", float, 1e-2, "KL divergence between old and new policy (averaged over state-space)"),
    ]

    def __init__(self, env, neural_type, gamma):

        policy = DeepFunctions.DeepPolicy(env, neural_type)
        
        super(TRPO, self).__init__(policy)
        
        self.discount = gamma

        self.setup_agent()
        
        self.env = env
        
        self.episodes = []
        
        self.progbar = Progbar(10)
        
    def setup_agent(self):
        
        self.states = self.model.input
        
        self.actions = K.placeholder(ndim=1, dtype = 'int32')
        
        self.advantages = K.placeholder(ndim=1)

        old_pi = K.placeholder(shape=(None, self.actions_n))

        current_pi = self.model.output

        log_pi = utils.loglikelihood(self.actions, current_pi)
        old_log_pi = utils.loglikelihood(self.actions, old_pi)

        N = K.cast(K.shape(self.states)[0],dtype='float32')

        # Policy gradient:

        surr = (-1.0 / N) * K.sum( K.exp(log_pi - old_log_pi)*self.advantages)
        pg = self.Flaten.flatgrad(self.params)

        current_pi_fixed = K.stop_gradient(current_pi)
        
        kl_firstfixed = K.sum(utils.kl(current_pi_fixed, current_pi))/N
        
        grads = self.Flaten.flatgrad(kl_firstfixed)
        
        flat_tangent = K.placeholder(ndim=1)
        
        gvp = K.sum(grads*flat_tangent)
        
        
        # Fisher-vector product
        fvp = self.Flaten.flatgrad(gvp)
        
        ent = K.mean(utils.entropy(current_pi))
        
        kl = K.mean(utils.kl(old_pi, current_pi))

        losses = [surr, kl, ent]
        
        self.loss_names = ["surr", "kl", "ent"]


        args = [self.states, self.actions, self.advantages, old_log_pi]
        self.compute_policy_gradient = K.function(args, [pg])
        self.compute_losses = K.function(args, losses)
        self.compute_fisher_vector_product = K.function([flat_tangent] + args, [fvp])

    def train(self):

        proba = np.concatenate([episode["proba"] for episode in self.episodes],axis=0)
        states = np.concatenate([episode["state"] for episode in self.episodes],axis=0)
        actions = np.concatenate([episode["action"] for episode in self.episodes],axis=0)
        
        advantages = np.concatenate([episode["advantage"] for episode in self.episodes],axis=0)
        
        args = (states, actions, advantages, proba)

        thprev = self.Flaten.get()
        
        g = self.compute_policy_gradient(*args)
        
        losses_before = self.compute_losses(*args)
        
        if np.allclose(g, 0):
            print("got zero gradient. not updating")
        else:
            stepdir = m_utils.conjugate_gradient(lambda x : self.fisher_vector_product(x,args), -g)
            shs = .5*stepdir.dot(self.fisher_vector_product(stepdir,args))
            lm = np.sqrt(shs / self.options["max_kl"])
            print("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)
            def loss(th):
                self.Flaten.set(th)
                return self.compute_losses(*args)[0]
            
            success, theta = m_utils.line_search(loss, thprev, fullstep, neggdotstepdir/lm)
            print("success", success)
            self.Flaten.set(theta)
        losses_after = self.compute_losses(*args)

        out = {}
        for (lname, lbefore, lafter) in zip(self.loss_names, losses_before, losses_after):
            out[lname+"_before"] = lbefore
            out[lname+"_after"] = lafter
        return out
    
    def act(self,state,train=False):
        
        proba = self.model.predict(state)
        if train:
            action = utils.choice_weighted(proba)
        else:
            action = np.argmax(proba)
        # print(action)
        return action

    def fisher_vector_product(self,p,args):
            return self.compute_fisher_vector_product(p, *args)+self.options["cg_damping"]*p
            

    def rollout(self,num_frames, num_episodes):

        self.episodes = []
        collected = 0
        self.progbar.__init__(num_frames*num_episodes)

        while collected < num_frames*num_episodes:
            episode, new = self.get_episode(num_frames-collected)                
            self.episodes.append(episode)

        
    def get_episode(self, length, eps):
        
        state = self.env.reset()     
        
        episode = {s : [] for s in ["t","state","action","reward","proba"]}
        
        i = 0
        
        while i < length:
            
            
            self.progbar.add(1)

            episode["t"].append(i)
            episode["state"].append(state)
            # act
            proba = self.model.predict(state)
            action = utils.choice_weighted(proba)
            state, rew, done = self.env.step(action)

            episode["proba"].append(proba)
            episode["action"].append(action)
            episode["reward"].append(rew)        
            episode["terminated"].append(done)
            
            i += 1
            
            if done:
                break
            
            
        episode["return"] = discount(np.array(episode["reward"]), self.agent.discount)
        
        return episode,i

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
        
        self.agent.set_epsilon(0)
        state = self.env.reset()
        done = False
        
        while not done:
            
            action = self.agent.act(state)
            
            state, _, done = self.env.step(action)
        
        self.env.draw(name)

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]        


def normalize(v):
    norm = np.linalg.norm(v)
    v_tmp = v-np.mean(v)
    if norm < EPS: 
       return v_tmp
    return v_tmp/np.max(np.abs(v_tmp))


