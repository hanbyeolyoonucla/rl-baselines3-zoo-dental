import dental_env
import gymnasium as gym
import torch
import pandas as pd
import numpy as np

# env = gym.make("DentalEnv-v0", render_mode="human", max_episode_steps=1024, down_sample=30)
# env = gym.make("DentalEnv5D-v1", render_mode="human", max_episode_steps=1024, down_sample=10)
env = gym.make("DentalEnv6D-v0", render_mode="human", max_episode_steps=1000, down_sample=10)

observation, info = env.reset(seed=42)

# test demonstration
cutpath = pd.read_csv('dental_env/demonstrations/tooth_2_demonstration.csv')

for itr in range(1000):
    # action = env.action_space.sample()
    if itr+1 < len(cutpath):
        translation = cutpath.iloc[itr+1].to_numpy() - cutpath.iloc[itr].to_numpy()
        translation[3] = translation[3]//3
    else:
        translation = [0, 0, 0, 0]
    action = np.append(translation, [0, 0])
    state, reward, terminated, truncated, info = env.step(action)
    # print(state['agent_rot'])
    if terminated or truncated:
        env.close()
        observation, info = env.reset()

env.close()
