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
from stable_baselines3 import SAC, TD3
from ibrl_td3 import IBRL
import wandb
from wandb.integration.sb3 import WandbCallback
from hyperparams.python.td3_config import hyperparams
from hyperparams.python.ppo_config import CustomCombinedExtractor
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium.wrappers import TransformReward
import pickle

# Define train configs
config = dict(
    total_timesteps=10_000,
    buffer_size=5_000,
    learning_starts=500,
    learning_rate=1e-3,
    batch_size=256,
    train_freq=1,
    tau=0.01,
    action_noise_mu=0,
    action_noise_std=0.1,
    target_policy_noise=0.1,
    target_policy_clip=0.3,
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
    project="dental_td3",
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
model = TD3("MultiInputPolicy", env, verbose=1,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            learning_starts=config["learning_starts"],
            batch_size=config["batch_size"],
            train_freq=config["train_freq"],  # train every 100 rollout
            tau=config["tau"],
            target_policy_noise=config["target_policy_noise"],
            target_noise_clip=config["target_policy_clip"],
            tensorboard_log=f"runs/td3_{run.id}",
            action_noise=NormalActionNoise(config["action_noise_mu"]*np.ones(6), config["action_noise_std"]*np.ones(6)),
            policy_kwargs=config['policy_kwargs'])

# Prefill replay buffer with demonstration dataset
with h5py.File(f'dental_env/demonstrations/train_dataset_scaled_reward.hdf5', 'r') as f:
    for demo in f.keys():
        for i in range(len(f[demo]['acts'][:])):
            model.replay_buffer.add(
                obs=dict(voxel=f[demo]['obs']['voxel'][i],
                         burr_pos=f[demo]['obs']['burr_pos'][i],
                         burr_rot=f[demo]['obs']['burr_rot'][i]),
                next_obs=dict(voxel=f[demo]['obs']['voxel'][i + 1],
                              burr_pos=f[demo]['obs']['burr_pos'][i + 1],
                              burr_rot=f[demo]['obs']['burr_rot'][i + 1]),
                action=f[demo]['acts'][i],
                reward=f[demo]['rews'][i],
                done=f[demo]['info']['is_success'][i],
                infos=[dict(placeholder=None)]
            )
model.learn(total_timesteps=config["total_timesteps"],
            tb_log_name=f'first_run',
            reset_num_timesteps=True,
            progress_bar=True,
            )

# Save train results and replay buffer for continuing training
model.save(f'models/td3_{run.id}_v1')
model.save_replay_buffer(f'models/replay_buffer/td3_{run.id}_v1')
run.finish()
