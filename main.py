"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters used in the paper.
"""
import argparse
import glob
import os
import warnings

import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# get configs
from config.gridworld import args_grid_rl2, args_grid_varibad, args_grid_ldm
from config.mujoco import args_mujoco_ant_dir_rl2, args_mujoco_ant_dir_varibad, args_mujoco_ant_dir_ldm
from config.mujoco import args_mujoco_ant_goal_rl2, args_mujoco_ant_goal_varibad, args_mujoco_ant_goal_ldm
from config.mujoco import args_mujoco_cheetah_vel_rl2, args_mujoco_cheetah_vel_varibad, args_mujoco_cheetah_vel_ldm


# VariBAD
from config.mujoco import args_mujoco_cheetah_vel_inter_varibad
from config.mujoco import args_mujoco_ant_goal_inter_varibad
from config.mujoco import args_mujoco_ant_dir_2_varibad
from config.mujoco import args_mujoco_ant_dir_4_varibad
from config.mujoco import args_mujoco_walker_mass_inter_varibad
from config.mujoco import args_mujoco_hopper_mass_inter_varibad
from config.mujoco import args_mujoco_cheetah_mass_inter_varibad

# LDM
from config.mujoco import args_mujoco_cheetah_vel_inter_ldm
from config.mujoco import args_mujoco_ant_goal_inter_ldm
from config.mujoco import args_mujoco_ant_dir_2_ldm
from config.mujoco import args_mujoco_ant_dir_4_ldm
from config.mujoco import args_mujoco_walker_mass_inter_ldm
from config.mujoco import args_mujoco_hopper_mass_inter_ldm
from config.mujoco import args_mujoco_cheetah_mass_inter_ldm

from config.mujoco import args_mujoco_cheetah_vel_inter_rl2
from config.mujoco import args_mujoco_ant_goal_inter_rl2
from config.mujoco import args_mujoco_ant_dir_2_rl2
from config.mujoco import args_mujoco_ant_dir_4_rl2
from config.mujoco import args_mujoco_walker_mass_inter_rl2
from config.mujoco import args_mujoco_hopper_mass_inter_rl2
from config.mujoco import args_mujoco_cheetah_mass_inter_rl2


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='gridworld_varibad')
    parser.add_argument('--seed', type=int, default=73)
    args, rest_args = parser.parse_known_args()
    env = args.env_type
    seed = args.seed

    # --- GridWorld ---
    if env == 'gridworld_varibad':
        args = args_grid_varibad.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'gridworld_rl2':
        args = args_grid_rl2.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'gridworld_ldm':
        from metalearner_ldm import MetaLearner
        args = args_grid_ldm.get_args(rest_args)
        args2 = args_grid_rl2.get_args(rest_args)

    # --- AntDir ---
    if env == 'mujoco_ant_dir_varibad':
        args = args_mujoco_ant_dir_varibad.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'mujoco_ant_dir_rl2':
        args = args_mujoco_ant_dir_rl2.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'mujoco_ant_dir_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_ant_dir_ldm.get_args(rest_args)
        args2 = args_mujoco_ant_dir_rl2.get_args(rest_args)

    # --- AntGoal ---
    if env == 'mujoco_ant_goal_varibad':
        args = args_mujoco_ant_goal_varibad.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'mujoco_ant_goal_rl2':
        args = args_mujoco_ant_goal_rl2.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'mujoco_ant_goal_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_ant_goal_ldm.get_args(rest_args)
        args2 = args_mujoco_ant_goal_rl2.get_args(rest_args)

    # --- CheetahVel ---
    if env == 'mujoco_cheetah_vel_varibad':
        args = args_mujoco_cheetah_vel_varibad.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'mujoco_cheetah_vel_rl2':
        args = args_mujoco_cheetah_vel_rl2.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'mujoco_cheetah_vel_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_cheetah_vel_ldm.get_args(rest_args)
        args2 = args_mujoco_cheetah_vel_rl2.get_args(rest_args)
    


    # VariBAD
    ##############################################
    if env == 'mujoco_cheetah_vel_inter_varibad':
        args = args_mujoco_cheetah_vel_inter_varibad.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner

    elif env == 'mujoco_ant_goal_inter_varibad':
        args = args_mujoco_ant_goal_inter_varibad.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner
    
    elif env == 'mujoco_ant_dir_2_varibad':
        args = args_mujoco_ant_dir_2_varibad.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner
    
    elif env == 'mujoco_ant_dir_4_varibad':
        args = args_mujoco_ant_dir_4_varibad.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner
    
    elif env == 'mujoco_walker_mass_inter_varibad':
        args = args_mujoco_walker_mass_inter_varibad.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner
    
    elif env == 'mujoco_hopper_mass_inter_varibad':
        args = args_mujoco_hopper_mass_inter_varibad.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner
    
    elif env == 'mujoco_cheetah_mass_inter_varibad':
        args = args_mujoco_cheetah_mass_inter_varibad.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner
    

    # LDM
    ##############################################
    if env == 'mujoco_cheetah_vel_inter_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_cheetah_vel_inter_ldm.get_args(rest_args)
        args2 = args_mujoco_cheetah_vel_inter_rl2.get_args(rest_args)
        args2.seed = seed

    elif env == 'mujoco_ant_goal_inter_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_ant_goal_inter_ldm.get_args(rest_args)
        args2 = args_mujoco_ant_goal_inter_rl2.get_args(rest_args)
        args2.seed = seed
    
    elif env == 'mujoco_ant_dir_2_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_ant_dir_2_ldm.get_args(rest_args)
        args2 = args_mujoco_ant_dir_2_rl2.get_args(rest_args)
        args2.seed = seed
    
    elif env == 'mujoco_ant_dir_4_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_ant_dir_4_ldm.get_args(rest_args)
        args2 = args_mujoco_ant_dir_4_rl2.get_args(rest_args)
        args2.seed = seed
    
    elif env == 'mujoco_walker_mass_inter_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_walker_mass_inter_ldm.get_args(rest_args)
        args2 = args_mujoco_walker_mass_inter_rl2.get_args(rest_args)
        args2.seed = seed
    
    elif env == 'mujoco_hopper_mass_inter_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_hopper_mass_inter_ldm.get_args(rest_args)
        args2 = args_mujoco_hopper_mass_inter_rl2.get_args(rest_args)
        args2.seed = seed
    
    elif env == 'mujoco_cheetah_mass_inter_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_cheetah_mass_inter_ldm.get_args(rest_args)
        args2 = args_mujoco_cheetah_mass_inter_rl2.get_args(rest_args)
        args2.seed = seed
    

    # RL2
    ##############################################
    if env == 'mujoco_cheetah_vel_inter_rl2':
        args = args_mujoco_cheetah_vel_inter_rl2.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner

    elif env == 'mujoco_ant_goal_inter_rl2':
        args = args_mujoco_ant_goal_inter_rl2.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner
    
    elif env == 'mujoco_ant_dir_2_rl2':
        args = args_mujoco_ant_dir_2_rl2.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner
    
    elif env == 'mujoco_ant_dir_4_rl2':
        args = args_mujoco_ant_dir_4_rl2.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner
    
    elif env == 'mujoco_walker_mass_inter_rl2':
        args = args_mujoco_walker_mass_inter_rl2.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner
    
    elif env == 'mujoco_hopper_mass_inter_rl2':
        args = args_mujoco_hopper_mass_inter_rl2.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner
    
    elif env == 'mujoco_cheetah_mass_inter_rl2':
        args = args_mujoco_cheetah_mass_inter_rl2.get_args(rest_args)
        args.seed = seed
        from metalearner import MetaLearner



    # warning
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # start training
    if args.disable_varibad:
        # When the flag `disable_varibad` is activated, the file `learner.py` will be used instead of `metalearner.py`.
        # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
        learner = Learner(args)
    else:
        # if env in ['gridworld_ldm', 'mujoco_ant_dir_ldm', 'mujoco_ant_goal_ldm', 'mujoco_cheetah_vel_ldm']:
        if 'ldm' in env:
            learner = MetaLearner(args, args2)
        else:
            learner = MetaLearner(args)
    learner.train()


if __name__ == '__main__':
    main()
