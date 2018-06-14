#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 10:17:00 2018

@author: gamer
"""
import sys
sys.path.append("../")
from mpi4py import MPI
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
from envs.grid import GRID
from baselines.trpo_mpi.nosharing_cnn_policy import CnnPolicy
def train( num_timesteps):
    
    env = GRID(grid_size=36,square_size=4, stochastic = True)
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    
    def policy_fn(name, ob_space, ac_space):
        return CnnPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space)
    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def main():
    args = mujoco_arg_parser().parse_args()
    train(num_timesteps=args.num_timesteps)


if __name__ == '__main__':
    main()
