import dental_env
import gymnasium as gym
import torch
from torch import nn
import pandas as pd
import numpy as np
import h5py
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.monitor import Monitor
from ibrl_td3.custom_callback import CustomEvalCallback
from ibrl_custom_td3 import CustomTD3
from ibrl_td3 import IBRL
import wandb
from wandb.integration.sb3 import WandbCallback
from hyperparams.python.ibrl_config import CustomCombinedExtractor
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium.wrappers import TransformReward
import pickle
import yaml

# Define train configs
config = dict(
    total_timesteps=100_000,
    buffer_size=24_000,
    learning_starts=0,
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
                features_extractor_kwargs=dict(cnn_output_dim=1024),
                share_features_extractor=True,
                net_arch=dict(pi=[1024, 1024], qf=[1024, 1024]),
                normalize_images=False,
                bc_policy_path=f'models/bc_traction_policy_20',  # for use of pre-trained features extractor from bc policy
                use_bc_features_extractor=True,
                freeze_features_extractor=False,
            ),
    bc_replay_buffer_path=f'dental_env/demos_augmented/traction_new_hdf5',  #  None for non-prefill replay buffer
    env_max_episode_steps=200,
    stats_window_size=10,
)

# Initiate train logger (wandb)
run = wandb.init(
    project="dental_td3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)

# Save config pickle
with open(f'models/configs/dental_td3_{run.id}.pkl', 'wb') as f:
    pickle.dump(config, f)

# Define environment
env = gym.make("DentalEnvPCD-v0",
               render_mode=None,
               max_episode_steps=config["env_max_episode_steps"],
               tooth='tooth_3_1.0_None_top_0_144_313_508')
env = Monitor(env)

# define callbacks
eval_callback = CustomEvalCallback(eval_env=env, best_model_save_path='models/best_models',
                                   log_path=None, eval_freq=10_000,
                                   n_eval_episodes=10,
                                   deterministic=True, render=False)
# Define train model
model = CustomTD3("MultiInputPolicy", env, verbose=1,
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
                  policy_kwargs=config['policy_kwargs'],
                  bc_replay_buffer_path=config['bc_replay_buffer_path'],
                  stats_window_size=config['stats_window_size'],)
model.learn(total_timesteps=config["total_timesteps"],
            log_interval=config['stats_window_size'],
            tb_log_name=f'first_run',
            reset_num_timesteps=True,
            progress_bar=True,
            callback=eval_callback)

# Save train results and replay buffer for continuing training
model.save(f'models/dental_td3_{run.id}')
model.save_replay_buffer(f'models/replay_buffer/dental_td3_{run.id}')
run.finish()
