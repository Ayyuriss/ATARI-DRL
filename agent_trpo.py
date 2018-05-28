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

from base_classes.agent import Agent

class TRPO(Agent):
    
    options = [
        ("cg_damping", float, 1e-3, "Add multiple of the identity to Fisher matrix during CG"),
        ("max_kl", float, 1e-2, "KL divergence between old and new policy (averaged over state-space)"),
    ]

    def __init__(self, states_dim, actions_n, neural_type, gamma):

        policy = DeepFunctions.DeepPolicy(states_dim, actions_n, neural_type)
        super(TRPO, self).__init__(policy)
        
        self.discount = gamma

        self.setup_agent()
    
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

    def __call__(self, rollout):
        
        proba = rollout["proba"]
        states = rollout["states"]
        actions = rollout["actions"]
        
        advantages = rollout["advantages"]
        
        args = (states, actions, advantages, proba)

        thprev = self.Flaten.get()
        
        def fisher_vector_product(p):
            return self.compute_fisher_vector_product(p, *args)+self.options["cg_damping"]*p
        
        g = self.compute_policy_gradient(*args)
        
        losses_before = self.compute_losses(*args)
        
        if np.allclose(g, 0):
            print("got zero gradient. not updating")
        else:
            stepdir = m_utils.conjugate_gradient(fisher_vector_product, -g)
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
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
    
    def act(self,state):
        
        proba = self.model.predict(state)
        action = utils.choice_weighted(proba)
        # print(action)
        return action


