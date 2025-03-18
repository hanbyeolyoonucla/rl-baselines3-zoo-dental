import dental_env
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import h5py
from gymnasium.wrappers import TransformReward
from itertools import product
import random
from tqdm import tqdm
import yaml
import os
from spatialmath import UnitQuaternion

tnums = [2, 3, 4]
cut_types = ['left', 'right']
models = ['traction', 'coverage']

for tnum in tnums:
    for cut_type in cut_types:
        for model in models:
            data = f'tooth_{tnum}_{cut_type}'
            tooth_dir = f'dental_env/demos_augmented/{model}'
            print(f'{model}: {data}')

            with h5py.File(f'dental_env/demos_augmented/{model}_hdf5/{data}.hdf5', 'w') as f:
                with open(f'dental_env/demos_augmented/{model}_hdf5/{data}.yml', 'w') as s:

                    tooth_stat = {}
                    dirlist = os.listdir(tooth_dir)
                    for fname in tqdm(dirlist):

                        # remove .npy from file name
                        tooth = fname[:-4]

                        # only consider tnum and cut type defined
                        if not (f'tooth_{tnum}' in tooth and f'{cut_type}' in tooth):
                            continue

                        # initialize environment
                        env = gym.make("DentalEnvPCD-v0", render_mode=None, max_episode_steps=1000, tooth=tooth)
                        obs, info = env.reset(seed=42)
                        voxel_size = obs['voxel'].shape

                        # test demonstration
                        demos = np.loadtxt(f'dental_env/demos_augmented/{model}/{tooth}.csv', delimiter=' ')
                        time_steps = len(demos)
                        if time_steps > 500:
                            continue

                        # create h5py dataset
                        obs_voxel = f.create_dataset(f'{tooth}/obs/voxel',
                                                    (time_steps, voxel_size[0], voxel_size[1], voxel_size[2], voxel_size[3]),
                                                    compression='gzip', dtype=bool)
                        obs_pos = f.create_dataset(f'{tooth}/obs/burr_pos', (time_steps, 3), compression='gzip', dtype=np.float32)
                        obs_rot = f.create_dataset(f'{tooth}/obs/burr_rot', (time_steps, 4), compression='gzip', dtype=np.float32)
                        actions = f.create_dataset(f'{tooth}/acts', (time_steps - 1, 6), compression='gzip', dtype=np.float32)
                        success = f.create_dataset(f'{tooth}/info/is_success', (time_steps - 1,), compression='gzip', dtype=bool)
                        rews = f.create_dataset(f'{tooth}/rews', (time_steps - 1,), compression='gzip', dtype=np.float32)

                        obs_voxel[0] = obs['voxel']
                        obs_pos[0] = obs['burr_pos']
                        obs_rot[0] = obs['burr_rot']
                        total_reward = 0
                        total_collisions = 0
                        for itr in range(time_steps-1):
                            # take step
                            pos_action = demos[itr+1, :3] - demos[itr, :3]
                            quat_action = UnitQuaternion(demos[itr, 3:]).inv() * UnitQuaternion(demos[itr+1, 3:])
                            rpy_action = quat_action.rpy(unit='deg')
                            action = np.concatenate((pos_action, rpy_action))
                            obs, reward, terminated, truncated, info = env.step(action)

                            # save data
                            obs_voxel[itr+1] = obs['voxel']
                            obs_pos[itr+1] = obs['burr_pos']
                            obs_rot[itr+1] = obs['burr_rot']
                            actions[itr] = action
                            rews[itr] = reward
                            success[itr] = info['is_success']

                            # statistics
                            total_reward = total_reward + reward
                            total_collisions = total_collisions + info['is_collision']
                            # if terminated or truncated:
                            #     env.close()
                            #     observation, info = env.reset()

                        # save stat
                        tooth_stat[tooth] = {
                            'success': bool(info['is_success']),
                            'total_reward': float(total_reward),
                            "initial_caries": int(info['initial_caries']),
                            "decay_remained": int(info['decay_remained']),
                            "processed_cavity": int(info['processed_cavity']),
                            'CRE': float(info['CRE']),
                            'MIP': float(info['MIP']),
                            'traverse_length': float(info['traverse_length']),
                            'traverse_angle': float(info['traverse_angle']),
                            'total_collisions': int(total_collisions),
                        }
                        env.close()
                    yaml.dump(tooth_stat, s)

