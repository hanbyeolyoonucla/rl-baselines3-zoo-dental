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
    total_timesteps=5_000_000,
    buffer_size=50_000,
    bc_buffer_size=25_000,
    learning_starts=1_000,
    learning_rate=1e-4,
    batch_size=512,
    rl_bc_batch_ratio=0.5,
    train_freq=(1, "episode"),
    tau=0.01,
    action_noise_mu=0,
    action_noise_std=0.1,
    target_policy_noise=0.1,
    target_policy_clip=0.3,
    policy_delay=5,
    policy_kwargs=dict(
                activation_fn=nn.ReLU,
                features_extractor_class=CustomCombinedExtractor,
                features_extractor_kwargs=dict(cnn_output_dim=256),
                net_arch=dict(pi=[128, 128], qf=[400, 300]),
                normalize_images=False
            ),
    env_max_episode_steps=200,
)

# Initiate train logger (wandb)
run = wandb.init(
    project="dental_ibrl",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)

# Save config pickle
with open(f'models/configs/ibrl_{run.id}_v1.pkl', 'wb') as f:
    pickle.dump(config, f)

# Define environment
env = gym.make("DentalEnvPCD-v0",
               render_mode=None,
               max_episode_steps=config["env_max_episode_steps"],)

# Define train model
model = IBRL("MultiInputPolicy", env, verbose=1,
             learning_rate=config["learning_rate"],
             buffer_size=config["buffer_size"],
             bc_buffer_size=config["bc_buffer_size"],
             batch_size=config["batch_size"],
             rl_bc_batch_ratio=config["rl_bc_batch_ratio"],
             learning_starts=config["learning_starts"],
             train_freq=config["train_freq"],  # train every 100 rollout
             tau=config["tau"],
             policy_delay=config["policy_delay"],
             target_policy_noise=config["target_policy_noise"],
             target_noise_clip=config["target_policy_clip"],
             model_save_freq=config['total_timesteps']//3,  # don't save
             model_save_path=f'D:/dental_RL_data/models/ibrl_{run.id}',
             bc_replay_buffer_path=f'dental_env/demos_augmented/traction_hdf5',
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
