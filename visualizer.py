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

tnum = 5
env = gym.make("DentalEnv6D-v0", render_mode="human", max_episode_steps=1000, down_sample=10,
               tooth=f"tooth_{tnum}_1.0_0_0_0_0_0_0")
env = TransformReward(env, lambda r: np.sign(r) * np.log(1+np.abs(r)))

# test demonstration
demons = pd.read_csv(f'dental_env/demonstrations/tooth_{tnum}_demonstration.csv')
time_steps = len(demons)

# trained policy
policy = MultiInputActorCriticPolicy(observation_space=env.observation_space,
                                     action_space=env.action_space,
                                     lr_schedule=get_schedule_fn(0.003),
                                     **hyperparams["DentalEnv6D-v0"]['policy_kwargs'])
policy = policy.load('dental_env/demonstrations/bc_policy_ct_action_7')

#trained ibrl
# policy = IBRL.load(f'models/s01yrbci_2200.zip')

state, info = env.reset(seed=42)
total_reward = 0
total_collisions = 0
for itr in range(time_steps-1):
    # action = env.action_space.sample()
    action, _ = policy.predict(state, deterministic=True)
    # action = demons.iloc[itr+1].to_numpy() - demons.iloc[itr].to_numpy()
    # action[3:] = action[3:]//3
    state, reward, terminated, truncated, info = env.step(action)

    total_reward = total_reward + reward
    decay_removal = info['decay_removal']
    enamel_damage = info['enamel_damage']
    dentin_damage = info['dentin_damage']
    total_collisions = total_collisions + info['is_collision']
    success = info['is_success']

    print(f'-------iteration: {itr}-------')
    print(f'success: {success}')
    print(f'total_reward: {total_reward}')
    print(f'decay_removal: {decay_removal}')
    print(f'enamel_damage: {enamel_damage}')
    print(f'dentin_damage: {dentin_damage}')
    print(f'total_collisions: {total_collisions}')


    # if terminated or truncated:
    #     env.close()
    #     observation, info = env.reset()
env.close()
