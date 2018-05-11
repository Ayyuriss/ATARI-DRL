#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:27:48 2018

@author: thinkpad
"""
import numpy as np
import DeepFunctions
import utils.agent as utils
import utils.math as m_utils
import keras.backend as K
import collections

class Agent(object):
    
    def __init__(self, model):
        
        
        
        self.model = model
        
        self.actions_n = model.actions_n
        
        self.history = collections.OrderedDict()
        
        self.params = self.model.variables

        self.Flaten = utils.Flattener(self.params)
        
        self.checkpoints = "./checkpoints/"
        
    def act(self,state,train=False):
        
        raise NotImplementedError
    
    def reinforce(self,episodes):
        
        raise NotImplementedError
        
    def save(self,name):
        
        self.model.save(self.checkpoints+name)
        
    def load(self,name):
        
        return self.model.load(self.checkpoints+name)
    def log(self, key,value):
        if key not in self.history.keys():    
            self.history[key] = [value]
        else:
            self.history[key] = np.concatenate([self.history[key],[value]])


class DQN(Agent):
    
    def __init__(self, states_dim, actions_n, neural_type, gamma, epsilon):
        
        model = DeepFunctions.DeepQ(states_dim, actions_n, neural_type)
        super(DQN,self).__init__(model)
        self.eps = epsilon
        self.discount = gamma

    def act(self,state,train=True):
        if train:
            if np.random.rand()<self.eps:
                return np.random.choice(range(self.actions_n))
         
        return m_utils.argmax(self.model.evaluate(state))
    
    def reinforce(self,rollout):

        #t = rollout("t")
        states = rollout["state"]
        actions = rollout["action"]
        rewards = rollout["reward"]
        not_final = np.logical_not(rollout["terminated"])

        target_q = rollout["output"]
        
        old_theta = self.Flaten.get()
        old_q = self.model.evaluate(rollout["state"]) 
        
        target_q[np.arange(len(actions)),actions] = rewards 
        target_q[np.arange(len(actions)),actions][not_final] += self.discount*np.max(target_q,axis=1)[not_final]
        
        
        target_q = 0.9*old_q + 0.1*target_q
        
        self.model.learn(states,target_q)
                
        new_theta = self.Flaten.get()
        new_q = self.model.evaluate(rollout["state"])
        
        self.log("Average reward",np.mean(rewards))
        self.log("Min reward",np.min(rewards))
        self.log("Average return",np.mean(rollout["return"]))

        self.log("Theta MSE",np.linalg.norm(new_theta-old_theta))
        self.log("Q MSE",np.linalg.norm(new_q-target_q))

        self.log("Epsilon",self.eps)
        
        for k,v in self.history.items():
            print(k,": %f"%v[-1])
        
    def set_epsilon(self,eps):
        self.eps = eps
    
        
    

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
        
        proba = self.model.evaluate(state)
        action = utils.choice_weighted(proba)
        # print(action)
        return action