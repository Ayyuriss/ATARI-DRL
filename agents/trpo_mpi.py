#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:52:25 2018

@author: thinkpad
"""

from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from contextlib import contextmanager

class TRPO_MPI(object):

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
    
        self.nworkers = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()
        np.set_printoptions(precision=3)
        
        self.env = env
        
        # Setup losses and stuff
        # ----------------------------------------
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.setup_agent()
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
    def setup_agent(self):
        pi = self.policy_func("pi", self.env)
        oldpi = self.policy_func("oldpi", self.env)

        advantage_target = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        returns = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    
        observations = self.pi.oberservation_input
        actions = pi.pdtype.sample_placeholder([None])
    
        kl_old_new = oldpi.pd.kl(pi.pd)
        entropy = pi.pd.entropy()
        mean_kl = U.mean(kl_old_new)
        mean_entropy = U.mean(entropy)
        entropy_bonus = self.entropy_coeff * mean_entropy
    
        vferr = U.mean(tf.square(pi.value_pred - returns))
    
        ratio = tf.exp(pi.pd.logp(actions) - oldpi.pd.logp(actions)) # advantage * pnew / pold
        surrogate_gain = U.mean(ratio * advantage_target)
    
        optimization_gain = surrogate_gain + entropy_bonus
        losses = [optimization_gain, mean_kl, entropy_bonus, surrogate_gain, mean_entropy]
        self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]
    
        dist = mean_kl
    
        var_list = pi.variables["policy"]
        vf_var_list = pi.variables["value_function"]
        vfadam = MpiAdam(vf_var_list)
    
        get_flat = U.GetFlat(var_list)
        set_from_flat = U.SetFromFlat(var_list)
        kl_gradient = tf.gradients(dist, var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
            start += sz
        gvp = tf.add_n([U.sum(g*tangent) for (g, tangent) in zipsame(kl_gradient, tangents)]) #pylint: disable=E1111
        fvp = U.flatgrad(gvp, var_list)
    
        self.assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
        self.compute_losses = U.function([observations, actions, advantage_target], losses)
        self.compute_lossandgrad = U.function([observations, actions, advantage_target],
                                          losses + [U.flatgrad(optimization_gain, var_list)])
        self.compute_fvp = U.function([flat_tangent, observations, actions, advantage_target], fvp)
        self.compute_vflossandgrad = U.function([observations, returns], U.flatgrad(vferr, vf_var_list))
    
        @contextmanager
        def timed(msg):
            if self.rank == 0:
                print(colorize(msg, color='magenta'))
                tstart = time.time()
                yield
                print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
            else:
                yield
        
        def allmean(x):
            assert isinstance(x, np.ndarray)
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= self.nworkers
            return out
    
        U.initialize()
        th_init = get_flat()
        MPI.COMM_WORLD.Bcast(th_init, root=0)
        set_from_flat(th_init)
        vfadam.sync()
        print("Init param sum", th_init.sum(), flush=True)
    
    
    def rollout(self):

        # Prepare for rollouts
        # ----------------------------------------
        seg_gen = rollout_generator(self.pi, self.env, self.timesteps_per_batch, stochastic=True)
    
        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
    
        assert sum([self.max_iters>0, self.max_timesteps>0, self.max_episodes>0])==1
    
        #---for testing
        time_step_test_counter = 0
        N_test = 10    
            
        ret_te = np.zeros(N_test)
        test_iter = 0
        T_max = 1000    
            
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
        
        while True:        
            if self.callback: 
                callback(locals(), globals())
            if self.max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif self.max_episodes and episodes_so_far >= max_episodes:
                break
            elif self.max_iters and iters_so_far >= self.max_iters:
                break
            #logger.log("********** Iteration %i ************"%iters_so_far)
    
            with timed("sampling"):
                seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, self.gamma, lam)
    
            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
    
            if hasattr(self.pi, "ret_rms"): self.pi.ret_rms.update(tdlamret)
            if hasattr(self.pi, "ob_rms"): self.pi.ob_rms.update(ob) # update running mean/std for policy
    
    
    
            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]
            def fisher_vector_product(p):
                return allmean(self.compute_fvp(p, *fvpargs)) + cg_damping * p
    
            assign_old_eq_new() # set old parameter values to new parameter values
            
            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters = cg_iters, verbose=rank==0)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / max_kl)
                # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrogate_before = loss_before[0]
                stepsize = 1.0
                theta_before = self.pi.get_flat()
                for _ in range(10):
                    theta_new = theta_before + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                    improve = surr - surrogate_before
                    #logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        #logger.log("Got non-finite value of losses -- bad!")
                        pass
                    elif kl > max_kl * 1.5:
                        #logger.log("violated KL constraint. shrinking step.")
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
                    set_from_flat(theta_before)
                if nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
    
            #for (lossname, lossval) in zip(loss_names, meanlosses):
                #logger.record_tabular(lossname, lossval)
    
            with timed("vf"):
    
                for _ in range(vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]), 
                    include_final_partial_batch=False, batch_size=64):
                        g = allmean(compute_vflossandgrad(mbob, mbret))
                        value_ffunction_adam.update(g, vf_stepsize)
    
            #test policy once every 1000 time steps            
            if timesteps_so_far >= 5000*test_iter or iters_so_far == 0:
            #if 1:
                test_ite = iters_so_far
                time_step_test_counter = 0
                ret_te = np.zeros(N_test)
                t_te_sum = 0
                for nn in range(0, N_test):
                    state_te = self.env_test.reset() 
                    state_dim = self.env_test.observation_space.shape[0]
                    state_te = np.reshape(state_te, (-1, state_dim))
                    for t_te in range(0, T_max):
                        #if do_render and nn == N_test-1:
                        #    env_test.render()
                        #    time.sleep(0.01)    #to slowdown rendering
    
                        action_te = pi.act_only(False, state_te)    
                        #action_te, dummy = pi.act(False, state_te)                
                        #ac, vpred = pi.act(stochastic, ob)
                        
                        #no need to clip for deterministic policy    
                        #action_te = np.clip(action_te, a_min=ac_space.low[0], a_max=ac_space.high[0])
                        (next_state_te, reward_te, done_te, info_te) = env_test.step(action_te)  
                        state_dim = env.observation_space.shape[0]
                        next_state_te = np.reshape(next_state_te, (-1, state_dim))
    
                        state_te = next_state_te
                        ret_te[nn]  = ret_te[nn] + reward_te 
                        t_te_sum = t_te_sum + 1
                        if done_te:
                            break 
                ret_mean = np.mean(ret_te)
                ret_std = np.std(ret_te) / np.sqrt(N_test)
                t_run = t_te_sum / N_test
    
                #---prepare result
                # result = "Test iter %d : step_so_far %d, return %0.4f(%0.3f)" \
                #         % (test_iter, timesteps_so_far, ret_mean, ret_std)
                result = ret_mean
                
                #---print / save the result
                print(result)
                        
                try:
                    f = open(filename, 'a')
                    print(result, file=f)
                    f.close()
                except:
                    print("cannot save file at iter %d" % test_iter)
                    result_tmp += result + "\n"
    
                test_iter = test_iter + 1
                        
            #----------------------------        
                        
            #logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
    
            lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
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

        
def rollout_generator(pi, env, horizon, stochastic):

    # Initialize state variables

    t = 0
    ac = env.action_space.sample()
    
    new = True
    rew = 0.0
    ob = env.reset()
    state_dim = env.observation_space.shape[0]
    ob = np.reshape(ob, (-1, state_dim))

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:  #--> this will return a vector of obs once horizon is completed
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred = pi.act(stochastic, ob)            
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        state_dim = env.observation_space.shape[0]
        ob = np.reshape(ob, (-1, state_dim))
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            state_dim = env.observation_space.shape[0]
            ob = np.reshape(ob, (-1, state_dim))
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]



def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
