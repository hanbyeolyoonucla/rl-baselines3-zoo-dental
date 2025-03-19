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

tnums = [2, 4, 5]
cut_types = ['top']
train_trajectories = []
for tnum in tnums:
    for cut_type in cut_types:
        with h5py.File(f'dental_env/demos_augmented/traction_hdf5/tooth_{tnum}_{cut_type}.hdf5', 'r') as f:
            for demo in f.keys():
                dictobs = dict(voxel=f[demo]['obs']['voxel'][:],
                            burr_pos=f[demo]['obs']['burr_pos'][:],
                            burr_rot=f[demo]['obs']['burr_rot'][:])
                trajectory = TrajectoryWithRew(obs=DictObs(dictobs),
                                            acts=f[demo]['acts'][:],  # .astype(int)+1
                                            infos=f[demo]['info']['is_success'][:],
                                            rews=f[demo]['rews'][:],
                                            terminal=True)
                train_trajectories.append(trajectory)
train_transitions = rollout.flatten_trajectories(train_trajectories)

env = gym.make("DentalEnvPCD-v0", max_episode_steps=500)
vec_env = make_vec_env("DentalEnvPCD-v0")
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

# train for 100 epochs
bc_trainer.train(n_epochs=10, log_interval=1, log_rollouts_venv=vec_env, log_rollouts_n_episodes=5)

# rollout
# reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
# print(f"Reward after training: {reward_after_training}")

# save
bc_trainer.policy.save('dental_env/demos_augmented/bc_traction_policy')
