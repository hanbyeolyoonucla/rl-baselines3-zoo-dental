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
import os
from tqdm import tqdm
import wandb
from stable_baselines3.common.logger import configure
from imitation.util.logger import HierarchicalLogger

model = 'traction'
model_dir = f'dental_env/demos_augmented/{model}_hdf5'
dirlist = os.listdir(model_dir)
train_trajectories = []
for fname in tqdm(dirlist):
    if not fname.endswith('hdf5') or 'tooth_3' in fname:
        continue
    with h5py.File(f'{model_dir}/{fname}', 'r') as f:
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
print(f'num_trajectries: {len(train_trajectories)}')
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
    policy=policy.load(f'dental_env/demos_augmented/bc_{model}_policy_4'),  # .load(f'dental_env/demos_augmented/bc_{model}_policy')
    rng=rng,
)

# train for 100 epochs  # 10/100/100/300/500
bc_trainer.train(n_epochs=500, log_interval=10000, log_rollouts_venv=vec_env, log_rollouts_n_episodes=5)

# rollout
# reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
# print(f"Reward after training: {reward_after_training}")

# save
bc_trainer.policy.save(f'dental_env/demos_augmented/bc_{model}_policy_5')
