#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:27:48 2018

@author: thinkpad
"""
import numpy as np
import DeepFunctions
import rl_tools

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
         
        return rl_tools.argmax(self.model.evaluate(state))
    
    def reinforce(self,rollout):
        states = rollout["states"]
        actions = rollout["actions"]
        rewards = rl_tools.discount(rollout["rewards"],self.discount)
        not_final = np.logical_not(rollout["terminated"])
        target_q = self.model.evaluate(states)
        target_q[np.arange(len(actions)),actions] = rewards 
        target_q[np.arange(len(actions)),actions][not_final] += self.discount*np.max(target_q,axis=1)[not_final]
        
#        target_q[-1,actions[-1]] = rewards[-1]
        self.model.learn(states,target_q)
    
    def set_epsilon(self,eps):
        self.eps = eps
    
        
    
"""

class TRPO(Agent):
    
    options = [
        ("cg_damping", float, 1e-3, "Add multiple of the identity to Fisher matrix during CG"),
        ("max_kl", float, 1e-2, "KL divergence between old and new policy (averaged over state-space)"),
    ]

    def __init__(self, policy, usercfg):

        self.policy = policy
        self.cfg = cfg

        probtype = policy.type
        params = policy.trainable_variables
        

        ob_no = policy.states_dim
        act_na = policy.actions_n
        adv_n = T.vector("adv_n")

        # Probability distribution:
        prob_np = policy.get_output()
        oldprob_np = probtype.prob_variable()

        logp_n = probtype.loglikelihood(act_na, prob_np)
        oldlogp_n = probtype.loglikelihood(act_na, oldprob_np)
        N = ob_no.shape[0]

        # Policy gradient:
        surr = (-1.0 / N) * T.exp(logp_n - oldlogp_n).dot(adv_n)
        pg = flatgrad(surr, params)

        prob_np_fixed = theano.gradient.disconnected_grad(prob_np)
        kl_firstfixed = probtype.kl(prob_np_fixed, prob_np).sum()/N
        grads = T.grad(kl_firstfixed, params)
        flat_tangent = T.fvector(name="flat_tan")
        shapes = [var.get_value(borrow=True).shape for var in params]
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            tangents.append(T.reshape(flat_tangent[start:start+size], shape))
            start += size
        gvp = T.add(*[T.sum(g*tangent) for (g, tangent) in zipsame(grads, tangents)]) #pylint: disable=E1111
        # Fisher-vector product
        fvp = flatgrad(gvp, params)

        ent = probtype.entropy(prob_np).mean()
        kl = probtype.kl(oldprob_np, prob_np).mean()

        losses = [surr, kl, ent]
        self.loss_names = ["surr", "kl", "ent"]

        args = [ob_no, act_na, adv_n, oldprob_np]

        self.compute_policy_gradient = theano.function(args, pg, **FNOPTS)
        self.compute_losses = theano.function(args, losses, **FNOPTS)
        self.compute_fisher_vector_product = theano.function([flat_tangent] + args, fvp, **FNOPTS)

    def __call__(self, paths):
        cfg = self.cfg
        prob_np = concat([path["prob"] for path in paths])
        ob_no = concat([path["observation"] for path in paths])
        action_na = concat([path["action"] for path in paths])
        advantage_n = concat([path["advantage"] for path in paths])
        args = (ob_no, action_na, advantage_n, prob_np)

        thprev = self.get_params_flat()
        def fisher_vector_product(p):
            return self.compute_fisher_vector_product(p, *args)+cfg["cg_damping"]*p #pylint: disable=E1101,W0640
        g = self.compute_policy_gradient(*args)
        losses_before = self.compute_losses(*args)
        if np.allclose(g, 0):
            print "got zero gradient. not updating"
        else:
            stepdir = cg(fisher_vector_product, -g)
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / cfg["max_kl"])
            print "lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g)
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)
            def loss(th):
                self.set_params_flat(th)
                return self.compute_losses(*args)[0] #pylint: disable=W0640
            success, theta = linesearch(loss, thprev, fullstep, neggdotstepdir/lm)
            print "success", success
            self.set_params_flat(theta)
        losses_after = self.compute_losses(*args)

        out = OrderedDict()
        for (lname, lbefore, lafter) in zipsame(self.loss_names, losses_before, losses_after):
            out[lname+"_before"] = lbefore
            out[lname+"_after"] = lafter
        return out

"""