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

tnums = [2, 3, 4, 5]
scales = [0.9, 1.0, 1.1]
rotations_z = [0, 45, 90, 135, 180, 225, 270, 315]
rotations_y = [-10, 0, 10]
rotations_x = [-10, 0, 10]
translations_x = [-5, 0, 5]
translations_y = [-5, 0, 5]
translations_z = [-10, -5, 0]
tooth_dataset = list(product(tnums, scales, rotations_z, rotations_y, rotations_x,
                             translations_x, translations_y, translations_z))
random.shuffle(tooth_dataset)
for i in range(0, len(tooth_dataset), 1000):
    with h5py.File(f'D:/dental_RL_data/demo_dataset/demo_dataset_{i}.hdf5', 'w') as f:
        with open(f'D:/dental_RL_data/demo_dataset/demo_dataset_{i}_stat.yml', 'w') as s:
            tooth_stat = {}
            for tooth_info in tqdm(tooth_dataset[i:i+1000]):
                tnum, scale, rz, ry, rx, tx, ty, tz = tooth_info
                tooth = f'tooth_{tnum}_{scale}_{rx}_{ry}_{rz}_{tx}_{ty}_{tz}'
                env = gym.make("DentalEnv6D-v0", render_mode=None, max_episode_steps=1000, tooth=tooth)
                # env = TransformReward(env, lambda r: np.sign(r) * np.log(1+np.abs(r)))

                obs, info = env.reset(seed=42)
                voxel_size = obs['voxel'].shape

                # test demonstration
                demos = pd.read_csv(f'dental_env/demos_augmented/{tooth}.csv')
                time_steps = len(demos)

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
                    action = demos.iloc[itr+1].to_numpy() - demos.iloc[itr].to_numpy()
                    action[3:] = action[3:]//3
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
                    'decay_removal [%]': float(info['decay_removal']),
                    'enamel_damage [voxel]': int(info['enamel_damage']),
                    'dentin_damage [voxel]': int(info['dentin_damage']),
                    'total_collisions': int(total_collisions),
                }
                env.close()
            yaml.dump(tooth_stat, s)

