import dental_env
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
from imitation.data import rollout, types, serialize
from imitation.algorithms import bc
from stable_baselines3.ppo import PPO
from hyperparams.python.ppo_config import hyperparams
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
import h5py
from imitation.data.types import TrajectoryWithRew, DictObs

with h5py.File('dental_env/demos_augmented/traction_hdf5/tooth_2_top.hdf5', 'r') as f:
    trajectories = []
    for demo in f.keys():
        if not ('1.0_None_top_1' in demo):
            continue
        dictobs = dict(voxel=f[demo]['obs']['voxel'][:],
                       burr_pos=f[demo]['obs']['burr_pos'][:],
                       burr_rot=f[demo]['obs']['burr_rot'][:])
        trajectory = TrajectoryWithRew(obs=DictObs(dictobs),
                                       acts=f[demo]['acts'][:],  # .astype(int)+1
                                       infos=f[demo]['info']['is_success'][:],
                                       rews=f[demo]['rews'][:],
                                       terminal=True)
        trajectories.append(trajectory)

num_trajectories = len(trajectories)
train_ratio, val_ratio = 0.8, 0.9
train_idx = int(num_trajectories*train_ratio)
val_idx = int(num_trajectories*val_ratio)
indices = np.arange(num_trajectories)
np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[:train_idx], indices[train_idx:val_idx], indices[val_idx:]

train_transitions = rollout.flatten_trajectories([trajectories[idx] for idx in train_indices])
val_transitions = rollout.flatten_trajectories([trajectories[idx] for idx in val_indices])
test_transitions = rollout.flatten_trajectories([trajectories[idx] for idx in test_indices])

tooth = 'tooth_2_1.0_None_top_1_119_303_464'
env = gym.make("DentalEnvPCD-v0", max_episode_steps=200, tooth=tooth)
policy = MultiInputActorCriticPolicy(observation_space=env.observation_space,
                                     action_space=env.action_space,
                                     lr_schedule=get_schedule_fn(0.005),
                                     **hyperparams["DentalEnv6D-v0"]['policy_kwargs'])
rng = np.random.default_rng(0)
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    batch_size=512,
    demonstrations=train_transitions,
    policy=policy,  # .load('dental_env/demonstrations/bc_policy_ct_action_6')
    rng=rng,
)

bc_trainer.train(n_epochs=10, log_interval=1)
# reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 1)
# print(f"Reward after training: {reward_after_training}")

# bc_trainer.policy.save('dental_env/demos_augmented/bc_traction_policy')
