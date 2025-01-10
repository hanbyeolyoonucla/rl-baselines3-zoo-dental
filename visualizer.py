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

if __name__ == "__main__":

    # settings
    tnum, scale, rz, ry, rx, tx, ty, tz = 2, 1.0, 0, 0, 0, 0, 0, 0
    tooth = f'tooth_{tnum}_{scale}_{rx}_{ry}_{rz}_{tx}_{ty}_{tz}'
    # tooth = None
    policy_type = ["demo", "IL", "RL", "IBRL", "random"][0]

    # Initialize gym environment
    env = gym.make("DentalEnv6D-v0", render_mode="human", max_episode_steps=1000, tooth=tooth)
    state, info = env.reset(seed=42)
    # env = TransformReward(env, lambda r: np.sign(r) * np.log(1+np.abs(r)))

    demos = pd.read_csv(f'dental_env/demos_augmented/{info["tooth"]}.csv')
    time_steps = len(demos)
    if policy_type == "IL":
        policy = MultiInputActorCriticPolicy(observation_space=env.observation_space,
                                             action_space=env.action_space,
                                             lr_schedule=get_schedule_fn(0.003),
                                             **hyperparams["DentalEnv6D-v0"]['policy_kwargs'])
        policy = policy.load('dental_env/demonstrations/bc_policy_ct_action_7')
    elif policy_type == "RL":
        policy = TD3.load(f'models/td3_j0g63jcw_v1.zip')
    elif policy_type == "IBRL":
        policy = IBRL.load(f'models/ibrl_n7i1j8w1_v1.zip')

    total_reward = 0
    total_collisions = 0
    for itr in range(time_steps-1):
        if policy_type == "demo":
            action = demos.iloc[itr+1].to_numpy() - demos.iloc[itr].to_numpy()
            action[3:] = action[3:]//3
        elif policy_type in ["IL", "RL", "IBRL"]:
            action, _ = policy.predict(state, deterministic=True)
            # action, _ = policy.predict(state, deterministic=True, use_actor_target=False)
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

        print(f'-------iteration: {itr}-------')
        print(f'tooth: {tooth_name}')
        print(f'success: {success}')
        print(f'total_reward: {total_reward}')
        print(f'decay_removal [%]: {decay_removal}')
        print(f'enamel_damage [voxel]: {enamel_damage}')
        print(f'dentin_damage [voxel]: {dentin_damage}')
        print(f'total_collisions: {total_collisions}')

        # if terminated or truncated:
        #     env.close()
        #     observation, info = env.reset()
    env.close()
