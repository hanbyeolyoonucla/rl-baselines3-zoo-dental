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
import os
from spatialmath import UnitQuaternion
from traction import Traction


if __name__ == "__main__":

    # select model
    policy_type = ['demo', 'random', 'IL', 'TD3', 'IBRL'][3]
    model = ['coverage', 'traction', 'human'][1]
    apf = Traction()

    # select tooth
    # tooth_dir = f'dental_env/demos_augmented/{model}'
    # dirlist = os.listdir(tooth_dir)
    # fname = dirlist[np.random.randint(1, len(dirlist))]
    # tooth = fname[:-4]
    tooth='tooth_3_1.0_None_top_0_144_313_508'
    # tooth='tooth_2_1.1_None_top_1_158_346_562'

    # Initialize gym environment
    env = gym.make("DentalEnvPCD-v0", render_mode=None, max_episode_steps=200, tooth=tooth)
    state, info = env.reset(seed=42)

    if policy_type == "demo":
        demos = np.loadtxt(f'dental_env/demos_augmented/{model}/{tooth}.csv', delimiter=' ')
        time_steps = len(demos)
    elif policy_type == "IL":
        policy = CustomActorCriticPolicy.load(f'models/bc_traction_policy_30')
    elif policy_type == "TD3":
        # policy = CustomTD3.load(f'models/dental_td3_ak9uohu2.zip')
        # policy = CustomTD3.load(f'models/best_best/best_model.zip', bc_replay_buffer_path=None)
        policy = CustomTD3.load(f'models/td3_best_models/pa9mjcwi/best_model.zip', bc_replay_buffer_path=None)
    elif policy_type == "IBRL":
        # policy = IBRL.load(f'models/ibrl_ezf2013i.zip', bc_replay_buffer_path=None)
        # policy = IBRL.load(f'models/ibrl_sgvy41b4.zip', bc_replay_buffer_path=None)
        policy = IBRL.load(f'models/best_best/best_model.zip', bc_replay_buffer_path=None)

    total_reward = 0
    total_collisions = 0
    itr = 0

    while True:
        if policy_type == "demo":
            if itr+1 < time_steps:
                pos_action = demos[itr+1, :3] - demos[itr, :3]
                quat_action = UnitQuaternion(demos[itr, 3:]).inv() * UnitQuaternion(demos[itr+1, 3:])
                rpy_action = quat_action.rpy(unit='deg')
                action = np.concatenate((pos_action, rpy_action))
            else:
                break
        elif policy_type in ["IL", "TD3", "IBRL" ]:
            action, _ = policy.predict(state, deterministic=True, use_actor_proposal=True)
        else:  # random
            action = env.action_space.sample()
            action = apf.predict(state)
        state, reward, terminated, truncated, info = env.step(action)

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

        print(f'-------iteration: {itr}-------')
        print(f'tooth: {tooth_name}')
        print(f'success: {success}')
        print(f'total_reward: {total_reward}')
        print(f'decay_removal [%]: {decay_removal}')
        print(f'enamel_damage [voxel]: {enamel_damage}')
        print(f'dentin_damage [voxel]: {dentin_damage}')
        print(f'CRE: {cre}')
        print(f'MIP: {mip}')
        print(f'total_collisions: {total_collisions}')
        print(f'traverse_length: {traverse_length}')
        print(f'traverse_angle: {traverse_angle}')

        pos = info['position']
        rot = info['rotation']
        print(f'position: {pos}')
        print(f'rotation: {rot}')

        itr += 1

        if terminated or truncated:
            env.close()
            break
    env.close()
