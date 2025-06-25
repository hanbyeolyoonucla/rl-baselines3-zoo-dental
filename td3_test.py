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
from rrl_td3 import ResidualTD3
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
    buffer_size=10_000,
    learning_starts=5_000,
    learning_rate=1e-5,
    batch_size=512,
    train_freq=(2, "step"),
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
                share_features_extractor=False,
                net_arch=dict(pi=[1024, 1024], qf=[1024, 1024]),
                normalize_images=False,
                bc_policy_path=None,  # for use of pre-trained features extractor from bc policy
                use_bc_features_extractor=False,
                freeze_features_extractor=False,
                alpha=0.01,
            ),
    bc_replay_buffer_path=None,  #
    env_max_episode_steps=200,
    stats_window_size=10,
    gamma=0.99,
)

# Initiate train logger (wandb)
run = wandb.init(
    project="dental_td3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)

# Define environment
env = gym.make("DentalEnvPCD-v0",
               render_mode=None,
               max_episode_steps=config["env_max_episode_steps"])
                # tooth='tooth_3_1.0_None_top_0_144_313_508'  # MIP: 7.494094488188976 # reward: 18.685433070866146 # tooth 3 
                # tooth='tooth_3_1.0_None_top_1_227_258_489'  # MIP: 7.002118644067797 # reward: 18.782838983050844
                # tooth='tooth_2_1.0_None_top_1_119_303_490'  # MIP: 5.384761904761905 # reward: 19.116761904761905 # tooth 2 
                # tooth='tooth_2_1.0_None_top_3_228_317_483'  # MIP: 7.802105263157895 # reward: 18.624
                # tooth='tooth_2_1.0_None_top_4_284_262_509'  # MIP: 9.994252873563218 # reward: 18.186206896551724
                # tooth='tooth_4_1.0_None_top_1_142_349_479'  # MIP: 4.699074074074074 # reward: 19.24652777777778
                # tooth='tooth_4_1.0_None_top_2_197_295_494'  # MIP: 5.74375 # reward: 19.0459375                   # tooth 4
                # tooth='tooth_4_1.0_None_top_3_190_229_502'  # MIP: 8.657777777777778 # reward: 18.468444444444447
                # tooth='tooth_5_1.0_None_top_0_272_249_489'  # MIP: 13.678571428571429 # reward: 17.464285714285715
                # tooth='tooth_5_1.0_None_top_1_118_219_484'  # MIP: 11.854961832061068 # reward: 17.825954198473283
                # tooth='tooth_5_1.0_None_top_2_159_241_487'  # MIP: 7.428571428571429 # reward: 18.714285714285715
                # tooth='tooth_syn_top_[1 1 2]'               # MIP: 2.937126800140499 # reward: 19.612574639971903 # tooth synthetic
env = Monitor(env)
# eval_env = gym.make("DentalEnvPCD-v0",
#                render_mode=None,
#                max_episode_steps=config["env_max_episode_steps"],
#                 tooth='tooth_3_1.0_None_top_0_144_313_508')
#                 # tooth='tooth_2_1.0_None_top_1_119_303_490'
#                 # tooth='tooth_4_1.0_None_top_2_197_295_494'
#                 # tooth='tooth_3_1.0_None_top_0_144_313_508'
# eval_env = Monitor(eval_env)

# define callbacks
eval_callback = CustomEvalCallback(eval_env=env, best_model_save_path=f'models/td3_best_models/{run.id}',
                                   log_path=None, eval_freq=2_000,
                                   n_eval_episodes=11,
                                   deterministic=True, render=False)
# Define train model  # CustomTD3  # ResidualTD3
model = ResidualTD3("MultiInputPolicy", env, verbose=1,
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
                  action_noise=NormalActionNoise(config["action_noise_mu"]*np.ones(6), config["action_noise_std"]*np.ones(6)),  #NormalActionNoise(config["action_noise_mu"]*np.ones(6), config["action_noise_std"]*np.ones(6)),
                  policy_kwargs=config['policy_kwargs'],
                  bc_replay_buffer_path=config['bc_replay_buffer_path'],
                  stats_window_size=config['stats_window_size'],
                  gamma=config['gamma'])
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
