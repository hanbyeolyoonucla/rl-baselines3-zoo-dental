import dental_env
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import h5py

tnums = [2, 3, 4, 5]

with h5py.File('D:/dental_RL_data/train_set.hdf5', 'w') as f:

    for tnum in tnums:

        # env = gym.make("DentalEnv-v0", render_mode="human", max_episode_steps=1024, down_sample=30)
        # env = gym.make("DentalEnv5D-v1", render_mode="human", max_episode_steps=1024, down_sample=10)
        env = gym.make("DentalEnv6D-v0", render_mode="human", max_episode_steps=1000, down_sample=10,
                       tooth=f"tooth_{tnum}_1.0_0_0_0_0_0_0")

        state, info = env.reset(seed=42)
        voxel_size = state['state'].shape

        # test demonstration
        demons = pd.read_csv(f'dental_env/demonstrations/tooth_{tnum}_demonstration.csv')
        time_steps = len(demons)

        states = f.create_dataset(f'tooth_{tnum}/states', (time_steps-1, voxel_size[0], voxel_size[1], voxel_size[2], voxel_size[3]), dtype=int)
        burr_pos = f.create_dataset(f'tooth_{tnum}/burr_pos', (time_steps - 1, 3), dtype=float)
        burr_rot = f.create_dataset(f'tooth_{tnum}/burr_rot', (time_steps - 1, 4), dtype=float)
        actions = f.create_dataset(f'tooth_{tnum}/actions', (time_steps - 1, 6), dtype=float)

        for itr in range(time_steps-1):
            # action = env.action_space.sample()
            action = demons.iloc[itr+1].to_numpy() - demons.iloc[itr].to_numpy()
            action[3:] = action[3:]//3
            states[itr] = state['state']
            burr_pos[itr] = state['burr_pos']
            burr_rot[itr] = state['burr_rot']
            actions[itr] = action
            state, reward, terminated, truncated, info = env.step(action)

            # if terminated or truncated:
            #     env.close()
            #     observation, info = env.reset()

        env.close()
