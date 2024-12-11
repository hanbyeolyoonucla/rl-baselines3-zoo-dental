import dental_env
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import h5py
from gymnasium.wrappers import TransformReward


tnums = [2, 3, 4, 5]

with h5py.File('dental_env/demonstrations/train_dataset_scaled_reward.hdf5', 'w') as f:

    for tnum in tnums:

        # env = gym.make("DentalEnv-v0", render_mode="human", max_episode_steps=1024, down_sample=30)
        # env = gym.make("DentalEnv5D-v1", render_mode="human", max_episode_steps=1024, down_sample=10)
        env = gym.make("DentalEnv6D-v0", render_mode="human", max_episode_steps=1000, down_sample=10,
                       tooth=f"tooth_{tnum}_1.0_0_0_0_0_0_0")
        # env = TransformReward(env, lambda r: np.sign(r) * np.log(1+np.abs(r)))

        obs, info = env.reset(seed=42)
        voxel_size = obs['voxel'].shape

        # test demonstration
        demons = pd.read_csv(f'dental_env/demonstrations/tooth_{tnum}_demonstration.csv')
        time_steps = len(demons)

        # create h5py dataset
        obs_voxel = f.create_dataset(f'tooth_{tnum}/obs/voxel',
                                     (time_steps, voxel_size[0], voxel_size[1], voxel_size[2], voxel_size[3]), dtype=int)
        obs_pos = f.create_dataset(f'tooth_{tnum}/obs/burr_pos', (time_steps, 3), dtype=float)
        obs_rot = f.create_dataset(f'tooth_{tnum}/obs/burr_rot', (time_steps, 4), dtype=float)
        actions = f.create_dataset(f'tooth_{tnum}/acts', (time_steps - 1, 6), dtype=float)
        info_decay = f.create_dataset(f'tooth_{tnum}/info/decay_remained', (time_steps - 1,), dtype=int)
        info_success = f.create_dataset(f'tooth_{tnum}/info/is_success', (time_steps - 1,), dtype=int)
        rews = f.create_dataset(f'tooth_{tnum}/rews', (time_steps - 1,), dtype=float)

        obs_voxel[0] = obs['voxel']
        obs_pos[0] = obs['burr_pos']
        obs_rot[0] = obs['burr_rot']
        for itr in range(time_steps-1):
            # take step
            action = demons.iloc[itr+1].to_numpy() - demons.iloc[itr].to_numpy()
            action[3:] = action[3:]//3
            obs, reward, terminated, truncated, info = env.step(action)
            # save data
            obs_voxel[itr+1] = obs['voxel']
            obs_pos[itr+1] = obs['burr_pos']
            obs_rot[itr+1] = obs['burr_rot']
            actions[itr] = action
            rews[itr] = reward
            info_decay[itr] = info['decay_remained']
            info_success[itr] = info['is_success']

            # if terminated or truncated:
            #     env.close()
            #     observation, info = env.reset()

        env.close()
