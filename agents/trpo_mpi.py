#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:52:25 2018

@author: thinkpad
"""
from keras import optimizers

from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
import keras.backend as K

from collections import deque
from baselines.common.mpi_adam import MpiAdam
import utils.math as m_utils
import utils.console as c_utils

class TRPO(object):

    def __init__(self, env, env_test, seed, policy_func, *,
        timesteps_per_batch, # what to train on
        max_kl, cg_iters,
        gamma, lam, # advantage estimation
        entropy_coeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
        callback=None
        ):

        np.set_printoptions(precision=3)
        
        self.env = env
        
        # Setup losses and stuff
        # ----------------------------------------
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.setup_agent()
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.callback = callback
        
                #---for testing
            
        reward_path = "./Result/"
        import pathlib
        try: 
            pathlib.Path("./Result/" + self.env.name).mkdir(parents=True, exist_ok=True) 
            reward_path = "./Result/" + self.env.name + "/"
        except:
            print("cannot create result and model directories.")
    
        expname = "TRPO_%s_s%d" % (self.env.name, self.seed)
        print(expname)
        print("Seed: %d" % (self.seed))
        filename = reward_path + expname + ".txt"
        #---------
        result_tmp = ""
    
    def setup_agent(self):

        advantages = K.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        returns = K.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
        observations = self.pi.observation_input
        actions = self.pi.pdtype.sample_placeholder([None])
        
        self.pi = self.policy_func(self.env,observations)
        self.oldpi = self.policy_func(self.env,observations)
        
        self.value_function = self.value_func(self.env,observations, lr = self.vf_stepsize)
    
        kl_old_new = self.oldpi.kl(self.pi)
        entropy = self.pi.entropy()
        mean_kl = K.mean(kl_old_new)
        mean_entropy = K.mean(entropy)
        entropy_bonus = self.entropy_coeff * mean_entropy
    
        vferr = K.mean(K.square(self.pi.value_pred - returns))
    
        ratio = tf.exp(self.pi.pd.logp(actions) - self.oldpi.pd.logp(actions)) # advantage * pnew / pold
        surrogate_gain = K.mean(ratio * advantages)
    
        optimization_gain = surrogate_gain + entropy_bonus
        losses = [optimization_gain, mean_kl, entropy_bonus, surrogate_gain, mean_entropy]
        self.loss_names = ["Optim Gain", "mean KL", "entropy loss", "surrogate gain", "entropy"]
    
        dist = mean_kl
    
        
        kl_gradient = self.pi.flatten.gradient(dist)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")

        tangents = K.placeholder(shape=self.pi.variables.shape)
        gradient_vector_product = K.sum(kl_gradient*tangents) #pylint: disable=E1111
        fisher_vector_product = self.pi.flatten.flatgrad(gradient_vector_product)
    
        
        self.compute_losses = K.Function([observations, actions, advantages], losses)
        self.compute_loss_grad = K.Function([observations, actions, advantages],
                                          losses + [self.pi.flatten.flatgrad(optimization_gain)])
        self.compute_fisher_vector_product = K.Function([flat_tangent, observations, actions, advantages], fisher_vector_product)
        self.compute_value_function_lossandgrad = K.function([observations, returns], [self.value_function.flatten.flatgrad(vferr)])
    
        th_init = self.pi.flatten.get_value()

        print("Init param sum", th_init.sum(), flush=True)
    
    
    def train(self):

        # Prepare for rollouts
        # ----------------------------------------
        path_generator = self.roller()
    
        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        self.tstart = time.time()
        lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
    
        assert sum([self.max_iters>0, self.max_timesteps>0, self.max_episodes>0])==1
    
        while True:        
            if self.max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif self.max_episodes and episodes_so_far >= max_episodes:
                break
            elif self.max_iters and iters_so_far >= self.max_iters:
                break
            
            with c_utils.timed("Sampling"):
                path = path_generator.__next__()
                # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                
                self.old_pi.set_weights(self.pi.get_weights())
                ob, ac, adv, tdlamret = path["ob"], path["ac"], path["adv"], path["tdlamret"]
                
                vpredbefore = path["vpred"] # predicted value function before udpate
                
                adv = (adv - adv.mean()) / adv.std() # standardized advantage function estimate

                args = path["ob"], path["ac"], adv
                fvpargs = [arr[::5] for arr in args]
                def fisher_vector_product(p):
                    return self.compute_fvp(p, *fvpargs) + self.cg_damping * p
                
                with c_utils.timed("Compute Grad"):
                    *lossbefore, g = self.compute_loss_and_grad(*args)
    
                if np.allclose(g, 0):
                    logger.log("Got zero gradient. not updating")
                else:
                    with c_utils.timed("Conjugate Gradient"):
                        stepdir = m_utils.conjugate_gradient(fisher_vector_product, g, cg_iters = self.cg_iters, verbose = 1)
                    
                    assert np.isfinite(stepdir).all()
                    shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                    lm = np.sqrt(shs / self.max_kl)
                    # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                    fullstep = stepdir / lm
                    expected_improve = g.dot(fullstep)
                    surrogate_before = self.loss_before[0]
                    stepsize = 1.0
                    theta_before = self.pi.flatten.get_value()
                    for _ in range(10):
                        theta_new = theta_before + fullstep * stepsize
                        self.pi.flatten.set_value(theta_new)
                        meanlosses = surr, kl, *_ = np.array(self.compute_losses(*args))
                        improve = surr - surrogate_before
                        #logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                        if not np.isfinite(meanlosses).all():
                            pass
                        elif kl > max_kl * 1.5:
                            logger.log("Violated KL constraint. shrinking step.")
                            pass
                        elif improve < 0:
                            #logger.log("surrogate didn't improve. shrinking step.")
                            pass
                        else:
                            #logger.log("Stepsize OK!")
                            pass
                            break
                        stepsize *= .5
                    else:
                        logger.log("couldn't compute a good step")
                        self.pi.flatten.set_value(theta_before)
                    
                with c_utils.timed("Value Function Update"):
                    self.value_function.fit(path["ob"],path["tdlamret"], batch_size = 64, epochs = self.vf_iters)
        
                #test policy once every 1000 time steps            
                if timesteps_so_far >= 5000*test_iter or iters_so_far == 0:
                   
                       self.play()     
                #----------------------------        
                            
                #logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        
                lrlocal = (path["ep_lens"], path["ep_rets"]) # local values
                
                lens, rews = map(flatten_lists, zip(*listoflrpairs))
                lenbuffer.extend(lens)
                rewbuffer.extend(rews)
        
                episodes_so_far += len(lens)
                timesteps_so_far += sum(lens)
                iters_so_far += 1
                
                time_step_test_counter = time_step_test_counter + sum(lens)
        
                if self.rank==0:
                    pass
                    
                    
            #print result tmp
            print(result_tmp)
            try:
                f = open(filename, 'a')
                print(result_tmp, file=f)
                f.close()
            except:
                print("cannot save tmp file at iter %d")

    def roller(self, horizon):
        
        
        horizon = self.timesteps_per_batch
        
        state = self.env.reset()
        action = self.env.action_space.sample()
        i = 0
        path = {s : [] for s in ["t","state","action","reward","proba","vf","terminated","next_vf","ep_rew"]}
        rews = 0
        while True:
        
#                self.progbar.add(1)
                
            path["t"].append(i)
            path["prev_action"].append(action)
            path["state"].append(state)
            # act
            action, proba = self.pi.act(state, stochastic = True)
            vf = self.value_function.predict(state)
            
            state, rew, done = self.env.step(action)
            rews += rew
            path["action"].append(action)
            path["reward"].append(rew)        
            path["proba"].append(proba)
            path["vf"].append(vf)
            path["terminated"].append(done)
            path["next_vf"].append((1-done)*vf)
                            
            if done:
                path["ep_rew"].append(rews)
                path["ep_len"].append(i+1)
                i = 0
                state = self.env.reset()
            if not (i+1)%horizon:
                for k,v in path.items():
                    path[k] = np.array(v)
                self.add_vtarg_and_adv(path)
                yield path
                {s : [] for s in ["t","state","action","reward","proba","vf","terminated","next_vf","ep_rew"]}
        
    def add_vtarg_and_adv(self, path):
        # General Advantage Estimation
        terminal = np.append(path["terminated"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(path["vf"], path["next_vf"])
        T = len(path["reward"])
        path["advantage"] = np.empty(T, 'float32')
        rew = path["reward"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-terminal[t+1]
            delta = rew[t] + self.gamma * vpred[t+1] * nonterminal - vpred[t]
            path["advantage"][t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
        path["tdlamret"] = path["advantage"] + path["vpred"]
    
    def play(self):
        
            
        test_iter = 0
        T_max = 1000    
        N_test = 10
        rewards = np.zeros(N_test)
        t_te_sum = 0
        state_dim = self.env.observation_space.shape[0]
        
        for nn in range(0, N_test):
            state = self.env.reset() 
            state_dim = self.env.observation_space.shape[0]
            state = np.reshape(state, (-1, state_dim))
            for t in range(0, T_max):
                
                action = self.pi.act(state, stochastic = False)

                (state, reward, done, info) = self.env.step(action)  
                
                state = np.reshape(state, (-1, state_dim))
        
                rewards[nn]  = rewards[nn] + reward
                t_te_sum = t_te_sum + 1
                if done:
                    break
                
        rewards_mean = np.mean(rewards)
        rewards_std = np.std(rewards) / np.sqrt(N_test)
        t_run = t_te_sum / N_test
        
        #---prepare result
        # result = "Test iter %d : step_so_far %d, return %0.4f(%0.3f)" \
        #         % (test_iter, timesteps_so_far, ret_mean, ret_std)
        result = rewards_mean
        
        #---print / save the result
        print(result)
                
        try:
            f = open(self.filename, 'a')
            print(result, file=f)
            f.close()
        except:
            print("cannot save file at iter %d" % test_iter)
            result_tmp += result + "\n"
