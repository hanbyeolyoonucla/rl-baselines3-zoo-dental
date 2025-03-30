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
import yaml

# Define train configs
config = dict(
    total_timesteps=100_000,
    buffer_size=50_000,
    learning_starts=10_000,
    learning_rate=1e-4,
    batch_size=512,
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
    project="dental_td3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)
# wandb.tensorboard.patch(root_logdir=f"./runs/dental_td3_{run.id}/")
# Save config pickle
with open(f'models/configs/dental_td3_{run.id}.pkl', 'wb') as f:
    pickle.dump(config, f)

# Define environment
env = gym.make("DentalEnvPCD-v0",
               render_mode=None,
               max_episode_steps=config["env_max_episode_steps"],)

# Define train model
model = TD3("MultiInputPolicy", env, verbose=1,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            learning_starts=config["learning_starts"],
            batch_size=config["batch_size"],
            train_freq=config["train_freq"],  # train every 100 rollout
            tau=config["tau"],
            policy_delay=config["policy_delay"],
            target_policy_noise=config["target_policy_noise"],
            target_noise_clip=config["target_policy_clip"],
            tensorboard_log=f"runs/dental_td3_{run.id}",
            action_noise=NormalActionNoise(config["action_noise_mu"]*np.ones(6), config["action_noise_std"]*np.ones(6)),
            policy_kwargs=config['policy_kwargs'])

# Prefill replay buffer with demonstration dataset
with h5py.File(f'dental_env/demos_augmented/traction_hdf5/tooth_3_top.hdf5', 'r') as f:
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
            callback=WandbCallback(verbose=2))

# Save train results and replay buffer for continuing training
model.save(f'models/dental_td3_{run.id}')
model.save_replay_buffer(f'models/replay_buffer/dental_td3_{run.id}')
run.finish()
