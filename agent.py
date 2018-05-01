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


class Agent(object):
    
    def __init__(self, model, model_type):
        
        assert model_type in ["Q","Policy"]
        
        self.model = model
        self.actions_n = model.actions_n
        
    def act(self,state,train=False):
        
        raise NotImplementedError
    
    def reinforce(self,episodes):
        
        raise NotImplementedError
        
    def save(self,name):
        self.model.save(name)
    def load(self,name):
        self.model.load(name)


class DQN(Agent):
    
    def __init__(self, states_dim, actions_n, neural_type, gamma, epsilon):
        model = DeepFunctions.DeepQ(states_dim, actions_n, neural_type)
        super(DQN,self).__init__(model, "Q")
        self.eps = epsilon
        self.discount = gamma
    def act(self,state,train=True):
        if train:
            if np.random.rand()<self.eps:
                return np.random.choice(range(self.actions_n))
         
        return utils.argmax(self.model.evaluate(state))
    
    def reinforce(self,rollout):
        states = rollout["states"]
        actions = rollout["actions"]
        rewards = rollout["rewards"]
        not_final = np.logical_not(rollout["terminated"])
        target_q = self.model.evaluate(states)
        target_q[np.arange(len(actions)),actions] = rewards 
        target_q[np.arange(len(actions)),actions][not_final] += self.discount*np.max(target_q,axis=1)[not_final]
        

        self.model.learn(states,target_q)
    
    def set_epsilon(self,eps):
        self.eps = eps
    
        
    

class TRPO(Agent):
    
    options = [
        ("cg_damping", float, 1e-3, "Add multiple of the identity to Fisher matrix during CG"),
        ("max_kl", float, 1e-2, "KL divergence between old and new policy (averaged over state-space)"),
    ]

    def __init__(self, states_dim, actions_n, neural_type, gamma):

        self.policy = DeepFunctions.DeepPolicy(states_dim, actions_n, neural_type)
        
        self.discount = gamma
        
        self.params = self.policy.variables
        
        self.Flaten = utils.Flattener(self.params)

        states = self.policy.input
        actions = K.placeholder(ndim=1, dtype = 'int32')
        advantages = K.placeholder(ndim=1)

        old_pi = K.placeholder(shape=(None, self.policy.actions_n))
        current_pi = self.policy.output

        log_pi = utils.loglikelihood(actions, current_pi)
        old_log_pi = utils.loglikelihood(actions, old_pi)

        N = K.cast(K.shape(states)[0],dtype='float32')

        # Policy gradient:

        surr = (-1.0 / N) * K.sum( K.exp(log_pi - old_log_pi)*advantages)
        
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

        args = [states, actions, advantages, old_log_pi]

        self.compute_policy_gradient = K.function(args, [pg])
        self.compute_losses = K.function(args, losses)
        self.compute_fisher_vector_product = K.function([flat_tangent] + args, [fvp])

    def __call__(self, rollout):
        
        prob_np = rollout["proba"]
        ob_no = rollout["states"]
        action_na = rollout["actions"]
        advantage_n = rollout["advantages"]
        args = (ob_no, action_na, advantage_n, prob_np)

        thprev = self.get_params_flat()
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
                self.set_params_flat(th)
                return self.compute_losses(*args)[0] #pylint: disable=W0640
            success, theta = m_utils.line_search(loss, thprev, fullstep, neggdotstepdir/lm)
            print("success", success)
            self.set_params_flat(theta)
        losses_after = self.compute_losses(*args)

        out = {}
        for (lname, lbefore, lafter) in zip(self.loss_names, losses_before, losses_after):
            out[lname+"_before"] = lbefore
            out[lname+"_after"] = lafter
        return out
    
    def act(self,state):
        
        proba = self.policy.evaluate(state)
        action = utils.choice_weighted(proba)
        print(action)
        return action,proba