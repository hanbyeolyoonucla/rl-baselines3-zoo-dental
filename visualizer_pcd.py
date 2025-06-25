import dental_env
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import h5py
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from ibrl_td3.bc_policies import CustomActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from hyperparams.python.ppo_config import hyperparams
from gymnasium.wrappers import TransformReward
from ibrl_td3.ibrl import IBRL
from stable_baselines3 import TD3
from ibrl_custom_td3 import CustomTD3
from rrl_td3 import ResidualTD3
import os
from spatialmath import UnitQuaternion
from traction import Traction
import time


if __name__ == "__main__":

    # Select model
    policy_type = ['demo', 'random', 'IL', 'TD3', 'IBRL'][1]
    model = ['coverage', 'traction', 'human'][1]
    apf = Traction()

    # Select random tooth
    # tooth_dir = f'dental_env/demos_augmented/{model}'
    # dirlist = os.listdir(tooth_dir)
    # fname = dirlist[np.random.randint(1, len(dirlist))]
    # tooth = fname[:-4]

    # Select specific tooth
    # tooth='tooth_3_1.0_None_top_0_144_313_508'  # MIP: 7.494094488188976 # reward: 18.685433070866146 # tooth 3 
    # tooth='tooth_3_1.0_None_top_1_227_258_489'  # MIP: 7.002118644067797 # reward: 18.782838983050844
    # tooth='tooth_2_1.0_None_top_1_119_303_490'  # MIP: 5.384761904761905 # reward: 19.116761904761905 # tooth 2 
    # tooth='tooth_2_1.0_None_top_3_228_317_483'  # MIP: 7.802105263157895 # reward: 18.624
    # tooth='tooth_2_1.0_None_top_4_284_262_509'  # MIP: 9.994252873563218 # reward: 18.186206896551724
    # tooth='tooth_4_1.0_None_top_1_142_349_479'  # MIP: 4.699074074074074 # reward: 19.24652777777778
    # tooth='tooth_4_1.0_None_top_2_197_295_494'  # MIP: 5.74375 # reward: 19.0459375                   # tooth 4
    # tooth='tooth_4_1.0_None_top_3_190_229_502'  # MIP: 8.657777777777778 # reward: 18.468444444444447
    # tooth='tooth_5_1.0_None_top_0_272_249_489'  # MIP: 13.678571428571429 # reward: 17.464285714285715
    tooth='tooth_5_1.0_None_top_1_118_219_484'  # MIP: 11.854961832061068 # reward: 17.825954198473283
    # tooth='tooth_5_1.0_None_top_2_159_241_487'  # MIP: 7.428571428571429 # reward: 18.714285714285715
    # tooth='tooth_syn_top_[1 1 2]'               # MIP: 2.937126800140499 # reward: 19.612574639971903 # tooth synthetic

    # Initialize gym environment
    env = gym.make("DentalEnvPCD-v0", render_mode=None, max_episode_steps=200, tooth=tooth)
    state, info = env.reset(seed=42)

    if policy_type == "demo":
        demos = np.loadtxt(f'dental_env/demos_augmented/{model}/{tooth}.csv', delimiter=' ')
        time_steps = len(demos)
    elif policy_type == "IL":
        policy = CustomActorCriticPolicy.load(f'models/bc_traction_policy_30')
    elif policy_type == "TD3":
        # policy = ResidualTD3.load(f'models/td3_best_models/jh25epx9/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth all
        # policy = ResidualTD3.load(f'models/td3_best_models/yl3qdnxq/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth all
        # policy = ResidualTD3.load(f'models/td3_best_models/u9y1h58p/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth
        # policy = ResidualTD3.load(f'models/td3_best_models/vwrsfir3/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 2 top 1
        # policy = ResidualTD3.load(f'models/td3_best_models/z1jx7a2v/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 2 top 3
        # policy = ResidualTD3.load(f'models/td3_best_models/ad4yky08/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 2 top 4
        # policy = ResidualTD3.load(f'models/td3_best_models/qxvwmwxh/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 3 top 0
        # policy = ResidualTD3.load(f'models/td3_best_models/ye5lsi2i/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 3 top 1
        # policy = ResidualTD3.load(f'models/td3_best_models/gen7evvj/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 4 top 1
        # policy = ResidualTD3.load(f'models/td3_best_models/4yt86ukd/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 4 top 2
        # policy = ResidualTD3.load(f'models/td3_best_models/7mppwcmc/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 4 top 3
        # policy = ResidualTD3.load(f'models/td3_best_models/rz238wla/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 5 top 0
        # policy = ResidualTD3.load(f'models/td3_best_models/tfhu9pdd/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 5 top 1
        policy = ResidualTD3.load(f'models/td3_best_models/u9y1h58p/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 5 top 2
        # policy = ResidualTD3.load(f'models/td3_best_models/6hc2lbv4/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 2 shifted
        # policy = ResidualTD3.load(f'models/td3_best_models/9gn2dhzk/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 3 shifted
        # policy = ResidualTD3.load(f'models/td3_best_models/ei8kldrx/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 4 shifted
        # policy = ResidualTD3.load(f'models/td3_best_models/r0c0bz2m/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 2 simple
        # policy = ResidualTD3.load(f'models/td3_best_models/rfh9oelj/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 3 simple
        # policy = ResidualTD3.load(f'models/td3_best_models/4w8klpn6/best_model.zip', bc_replay_buffer_path=None, buffer_size=0) # tooth 4 simple
    elif policy_type == "IBRL":
        # policy = IBRL.load(f'models/ibrl_ezf2013i.zip', bc_replay_buffer_path=None)
        # policy = IBRL.load(f'models/ibrl_sgvy41b4.zip', bc_replay_buffer_path=None)
        policy = IBRL.load(f'models/best_best/best_model.zip', bc_replay_buffer_path=None)

    total_reward = 0
    total_collisions = 0
    itr = 0
    cutpath = []

    while True:

        # Predict action from model
        if policy_type == "demo":
            if itr+1 < time_steps:
                pos_action = demos[itr+1, :3] - demos[itr, :3]
                quat_action = UnitQuaternion(demos[itr, 3:]).inv() * UnitQuaternion(demos[itr+1, 3:])
                rpy_action = quat_action.rpy(unit='deg')
                action = np.concatenate((pos_action, rpy_action))
            else:
                break
        elif policy_type in ["IL", "TD3", "IBRL" ]:
            # action, _ = policy.predict(state, deterministic=True)
            action, _ = policy.predict(state, deterministic=True, total_action=True)
        else:  # random
            # action = env.action_space.sample()
            action = apf.predict(state)

        # Step env and log info
        cur_time = time.time()
        state, reward, terminated, truncated, info = env.step(action)
        print(f"colapsed time for each step {time.time()-cur_time}")

        total_reward = total_reward + reward
        decay_removal = info['decay_removal']
        enamel_damage = info['enamel_damage']
        dentin_damage = info['dentin_damage']
        total_collisions = total_collisions + info['is_collision']
        success = info['is_success']
        tooth_name = info['tooth']
        cre = info['CRE']
        mip = info['MIP']
        traverse_length = info['traverse_length']
        traverse_angle = info['traverse_angle']
        initial_caries = info['initial_caries']
        initial_enamel = info['initial_enamel']
        initial_dentin = info['initial_dentin']
        pos = info['position']
        rot = info['rotation']

        print(f'-------iteration: {itr}-------')
        print(f'tooth: {tooth_name}')
        print(f'success: {success}')
        print(f'total_reward: {total_reward}')
        print(f'decay_removal [%]: {decay_removal}')
        print(f'enamel_damage [voxel]: {enamel_damage}')
        print(f'dentin_damage [voxel]: {dentin_damage}')
        print(f'initial_caries [voxel]: {initial_caries}')
        print(f'initial_enamel [voxel]: {initial_enamel}')
        print(f'initial_dentin [voxel]: {initial_dentin}')
        print(f'CRE: {cre}')
        print(f'MIP: {mip}')
        print(f'total_collisions: {total_collisions}')
        print(f'traverse_length: {traverse_length}')
        print(f'traverse_angle: {traverse_angle}')
        print(f'position: {pos}')
        print(f'rotation: {rot}')

        itr += 1
        cutpath.append(np.concatenate((pos, rot)))
        if terminated or truncated:
            env.close()
            break
    # np.savetxt('cutpath_apf_rrl.txt', cutpath)
    env.close()
