import dental_env
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv


def make_venv():
    def _init():
        env = gym.make("DentalEnv6D-v0", render_mode="rgb_array", down_sample=10, tooth=f"tooth_{tnum}_1.0_0_0_0_0_0_0")
        return env
    return _init

tnums = [2, 3, 4, 5]
trajectories = []
trajectories_accum = rollout.TrajectoryAccumulator()
for tnum in tnums:

    env = DummyVecEnv([make_venv() for _ in range(1)])

    state = env.reset()
    trajectories_accum.add_step(dict(obs=state), 0)
    voxel_size = state['state'].shape

    # test demonstration
    demons = pd.read_csv(f'dental_env/demonstrations/tooth_{tnum}_demonstration.csv')
    time_steps = len(demons)

    for itr in range(time_steps-1):
        # action = env.action_space.sample()
        action = demons.iloc[itr+1].to_numpy() - demons.iloc[itr].to_numpy()
        action[3:] = action[3:]//3
        action = action[np.newaxis, :]
        state, reward, terminated, info = env.step(action)
        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            action, state, reward, terminated, info
        )
        trajectories.extend(new_trajs)

        # if terminated or truncated:
        #     env.close()
        #     observation, info = env.reset()

    env.close()

np.random.Generator.shuffle(trajectories)
transitions = rollout.flatten_trajectories(trajectories)
