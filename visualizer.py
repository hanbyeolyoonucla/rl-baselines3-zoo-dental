import dental_env
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import h5py
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from hyperparams.python.ppo_config import hyperparams

tnum = 2
# env = gym.make("DentalEnv-v0", render_mode="human", max_episode_steps=1024, down_sample=30)
# env = gym.make("DentalEnv5D-v1", render_mode="human", max_episode_steps=1024, down_sample=10)
env = gym.make("DentalEnv6D-v0", render_mode="human", max_episode_steps=1000, down_sample=10,
               tooth=f"tooth_{tnum}_1.0_0_0_0_0_0_0")
state, info = env.reset(seed=42)

# test demonstration
demons = pd.read_csv(f'dental_env/demonstrations/tooth_{tnum}_demonstration.csv')
time_steps = len(demons)

# trained policy
policy = MultiInputActorCriticPolicy(observation_space=env.observation_space,
                                     action_space=env.action_space,
                                     lr_schedule=get_schedule_fn(0.0003),
                                     **hyperparams["DentalEnv6D-v0"]['policy_kwargs'])
policy.load('dental_env/demonstrations/bc_policy')

for itr in range(time_steps-1):
    # action = env.action_space.sample()
    action, _ = policy.predict(state, deterministic=True)
    print(action)
    # action = demons.iloc[itr+1].to_numpy() - demons.iloc[itr].to_numpy()
    # action[3:] = action[3:]//3
    state, reward, terminated, truncated, info = env.step(action)

    # if terminated or truncated:
    #     env.close()
    #     observation, info = env.reset()

env.close()
