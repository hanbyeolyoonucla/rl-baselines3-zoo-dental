import dental_env
import gymnasium as gym
import torch
from torch import nn
import pandas as pd
import numpy as np
import h5py
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from hyperparams.python.ppo_config import hyperparams
from stable_baselines3 import SAC
from ibrl_td3 import IBRL
import wandb
from wandb.integration.sb3 import WandbCallback
from hyperparams.python.td3_config import hyperparams
from hyperparams.python.ppo_config import CustomCombinedExtractor
from gymnasium.wrappers import TransformReward
import pickle

# Define train configs
config = dict(
    total_timesteps=50_000,
    buffer_size=10_000,
    bc_buffer_size=1_504,
    learning_starts=500,
    batch_size=256,
    rl_bc_batch_ratio=0.5,
    train_freq=1,
    action_noise_mu=0,
    action_noise_std=0.1,
    policy_kwargs=dict(
                activation_fn=nn.ReLU,
                features_extractor_class=CustomCombinedExtractor,
                features_extractor_kwargs=dict(cnn_output_dim=256),
                net_arch=dict(pi=[128, 128], qf=[400, 300]),
                normalize_images=False
            ),
    env_max_episode_steps=500,
    env_tnum=5,
    env_down_sample=10,
    env_reward="original",
    env_use_log_reward=False
)

# Initiate train logger (wandb)
run = wandb.init(
    project="dental_ibrl",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)

# Save config pickle
with open(f'models/configs/td3_{run.id}_v1.pkl', 'wb') as f:
    pickle.dump(config, f)

# Define environment
tnum = config["env_tnum"]
env = gym.make("DentalEnv6D-v0",
               render_mode=None,
               max_episode_steps=config["env_max_episode_steps"],
               down_sample=config["env_down_sample"],
               tooth=f"tooth_{tnum}_1.0_0_0_0_0_0_0")
if config["env_use_log_reward"]:
    env = TransformReward(env, lambda r: np.sign(r) * np.log(1+np.abs(r)))

# Define train model
model = IBRL(config["policy_type"], env, verbose=1,
             buffer_size=config["buffer_size"],
             bc_buffer_size=config["bc_buffer_size"],
             batch_size=config["batch_size"],
             rl_bc_batch_ratio=config["rl_bc_batch_ratio"],
             learning_starts=config["learning_starts"],
             train_freq=config["train_freq"],  # train every 100 rollout
             model_save_freq=config['total_timesteps']//3,  # don't save
             model_save_path=f'D:/dental_RL_data/models/ibrl_{run.id}',
             bc_replay_buffer_path=f'dental_env/demonstrations/train_dataset_log_reward.hdf5',
             tensorboard_log=f"runs/ibrl_{run.id}",
             policy_kwargs=config['policy_kwargs'])
model.learn(total_timesteps=config["total_timesteps"],
            tb_log_name=f'first_run',
            reset_num_timesteps=True,
            progress_bar=True,
            )

# Save train results and replay buffer for continuing training
model.save(f'models/ibrl_{run.id}_v1')
model.save_replay_buffer(f'D:/dental_RL_data/replay_buffer/ibrl_{run.id}')
run.finish()
