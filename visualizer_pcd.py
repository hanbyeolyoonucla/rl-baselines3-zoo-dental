import dental_env
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import h5py
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from hyperparams.python.ppo_config import hyperparams
from gymnasium.wrappers import TransformReward
from ibrl_td3.ibrl import IBRL
from stable_baselines3 import TD3
import os
from spatialmath import UnitQuaternion


if __name__ == "__main__":

    # tooth
    policy_type = ['demo', 'random', 'IL'][2]
    model = ['coverage', 'traction', 'human'][0]
    tooth_dir = f'dental_env/demos_augmented/{model}'
    dirlist = os.listdir(tooth_dir)
    fname = dirlist[np.random.randint(1, len(dirlist))]
    tooth = fname[:-4]
    # tooth = 'tooth_2_1.1_0_top_1_280_320_515'
    # tooth = 'tooth_2_1.1_1_top_1_104_312_515'
    # tooth = 'tooth_2_1.1_None_top_1_104_320_515'
    # tooth = 'tooth_2_1.1_None_top_1_131_333_539'
    # tooth = 'tooth_2_1.1_None_top_1_158_346_562'
    # tooth = 'tooth_4_1.0_None_right_0_223_135_415'
    # tooth = 'tooth_2_0.9_0_left_2_266_295_346'
    # tooth = 'tooth_2_0.9_1_left_2_241_309_379'
    # tooth = 'tooth_2_0.9_0_right_0_109_229_379'
    # tooth = 'tooth_2_1.0_0_right_0_131_250_416'
    # tooth = 'tooth_2_1.0_0_top_1_251_287_464'
    # tooth = 'tooth_2_1.0_1_top_1_91_279_464'
    # tooth = 'tooth_2_1.0_None_top_1_91_287_464'
    # tooth = 'tooth_2_0.9_0_top_1_221_254_412'
    # tooth = 'tooth_2_0.9_1_top_1_77_246_412'
    # tooth = 'tooth_3_1.0_None_top_0_144_279_508'

    # Initialize gym environment
    env = gym.make("DentalEnvPCD-v0", render_mode=None, max_episode_steps=1000, tooth=tooth)
    state, info = env.reset(seed=42)

    demos = np.loadtxt(f'dental_env/demos_augmented/{model}/{tooth}.csv', delimiter=' ')
    time_steps = len(demos)

    if policy_type == "IL":
        policy = MultiInputActorCriticPolicy(observation_space=env.observation_space,
                                                action_space=env.action_space,
                                                lr_schedule=get_schedule_fn(0.003),
                                                **hyperparams["DentalEnv6D-v0"]['policy_kwargs'])
        policy = policy.load('dental_env/demos_augmented/bc_traction_policy_tooth3_2')

    total_reward = 0
    total_collisions = 0

    for itr in range(time_steps-1):
        if policy_type == "demo":
            pos_action = demos[itr+1, :3] - demos[itr, :3]
            quat_action = UnitQuaternion(demos[itr, 3:]).inv() * UnitQuaternion(demos[itr+1, 3:])
            rpy_action = quat_action.rpy(unit='deg')
            action = np.concatenate((pos_action, rpy_action))
        elif policy_type == "IL":
            action, _ = policy.predict(state, deterministic=True)
        else:
            action = env.action_space.sample()
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

        pos = info['position']
        rot = info['rotation']
        print(f'position: {pos}')
        print(f'rotation: {rot}')

        # if terminated or truncated:
        #     env.close()
        #     observation, info = env.reset()
    env.close()
