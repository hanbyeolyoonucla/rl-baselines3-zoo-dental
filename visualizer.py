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


def keyboard_to_action(key):
    act = np.array([0, 0, 0, 0, 0, 0])
    keymap = {'4': np.array([-1, 0, 0, 0, 0, 0]), '6': np.array([1, 0, 0, 0, 0, 0]),
              '1': np.array([0, -1, 0, 0, 0, 0]), '9': np.array([0, 1, 0, 0, 0, 0]),
              '2': np.array([0, 0, -1, 0, 0, 0]), '8': np.array([0, 0, 1, 0, 0, 0]),
              'a': np.array([0, 0, 0, -1, 0, 0]), 'd': np.array([0, 0, 0, 1, 0, 0]),
              'z': np.array([0, 0, 0, 0, -1, 0]), 'e': np.array([0, 0, 0, 0, 1, 0]),
              'x': np.array([0, 0, 0, 0, 0, -1]), 'w': np.array([0, 0, 0, 0, 0, 1]),
              }
    for c in key:
        act += keymap[c]
    return act


if __name__ == "__main__":

    # settings
    # tnum, scale, rz, ry, rx, tx, ty, tz = 5, 1.0, 0, 0, 0, 0, 0, 0
    # tooth = f'tooth_{tnum}_{scale}_{rx}_{ry}_{rz}_{tx}_{ty}_{tz}'
    # tooth = f'tooth_5_0_240_210_450_300_270_510'
    # tooth = None
    policy_type = ["demo", "IL", "RL", "IBRL", "random"][4]

    # Initialize gym environmentenvironment
    env = gym.make("DentalEnvPCD-v0", render_mode="human", max_episode_steps=1000)
    # env = gym.make("DentalEnv6D-v0", render_mode="human", max_episode_steps=1000, tooth=tooth)
    state, info = env.reset(seed=42)
    # env = TransformReward(env, lambda r: np.sign(r) * np.log(1+np.abs(r)))

    # demos = pd.read_csv(f'dental_env/demos_augmented/{info["tooth"]}.csv')
    res = 0.034
    demos = pd.read_csv(f'dental_env/labels_crop/tooth_5_0_demonstration.csv')
    time_steps = 1000  # len(demos)
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
    paths = [np.concatenate((info['position'], info['rotation']))]
    for itr in range(time_steps-1):
        if policy_type == "demo":
            # action = demos.iloc[itr+1].to_numpy() - demos.iloc[itr].to_numpy()
            action = demos.iloc[itr].to_numpy()
            action[:3] -= info['position']*res
            action[3:] = action[3:]//3
        elif policy_type in ["IL", "RL", "IBRL"]:
            action, _ = policy.predict(state, deterministic=True)
            # action, _ = policy.predict(state, deterministic=True, use_actor_target=False)
        else:
            action = env.action_space.sample()
            # user_input = input("Keyboard input: ")
            # action = keyboard_to_action(user_input)
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
        paths.append(np.concatenate((pos, rot)))
        print(f'position: {pos}')
        print(f'rotation: {rot}')

        # if terminated or truncated:
        #     env.close()
        #     observation, info = env.reset()
    env.close()
    # np.savetxt("demo_cutpath.csv", np.array(paths), delimiter=',')
