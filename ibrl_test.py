import dental_env
import gymnasium as gym
import torch
from torch import nn
import pandas as pd
import numpy as np
import h5py
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from ibrl_td3.custom_callback import CustomEvalCallback
from stable_baselines3 import SAC
from ibrl_td3 import IBRL
import wandb
from wandb.integration.sb3 import WandbCallback
from hyperparams.python.ibrl_config import CustomCombinedExtractor
from gymnasium.wrappers import TransformReward
import pickle

# Define train configs
config = dict(
    total_timesteps=100_000,
    buffer_size=12_000,
    bc_buffer_size=12_000,
    learning_starts=5_000,
    learning_rate=1e-5,
    batch_size=512,
    rl_bc_batch_ratio=0.5,
    train_freq=(2, "step"), #(1, "episode"),  # (2, "step")
    tau=0.1,
    target_policy_noise=0.1,
    target_policy_clip=0.2,
    policy_delay=5,
    policy_kwargs=dict(
                activation_fn=nn.ReLU,
                features_extractor_class=CustomCombinedExtractor,
                features_extractor_kwargs=dict(cnn_output_dim=1024),
                share_features_extractor=False,
                net_arch=dict(pi=[1024, 1024], qf=[1024, 1024]),
                normalize_images=False,
                bc_policy_path=f'models/bc_traction_policy_30',
                use_bc_features_extractor=True,
                freeze_features_extractor=False,
            ),
    bc_replay_buffer_path=f'dental_env/demos_augmented/traction_new_hdf5',
    env_max_episode_steps=200,
    stats_window_size=10,
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
               max_episode_steps=config["env_max_episode_steps"],
               tooth='tooth_3_1.0_None_top_0_144_313_508')  #tooth='tooth_3_1.0_None_top_0_144_313_508'
env = Monitor(env)

# define callbacks
eval_callback = CustomEvalCallback(env, best_model_save_path=f'models/ibrl_best_models/{run.id}',
                                   log_path=None, eval_freq=5_000,
                                   n_eval_episodes=1,  # because of using one env
                                   deterministic=True, render=False)

# Define train model
# model = IBRL.load(f'models/ibrl_ezf2013i.zip', env)
# model.load_replay_buffer(f'models/replay_buffer/ibrl_ezf2013i.pkl')
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
             model_save_freq=config['total_timesteps']//5,  # don't save
             model_save_path=f'models/ibrl_{run.id}',
             bc_replay_buffer_path=config['bc_replay_buffer_path'],
             tensorboard_log=f"runs/ibrl_{run.id}",
             policy_kwargs=config['policy_kwargs'],
             stats_window_size=config['stats_window_size'],)
print(model.policy)
model.learn(total_timesteps=config["total_timesteps"],
            log_interval=config['stats_window_size'],  # log every 10 episodes (1step in wandb = 10 episodes)
            tb_log_name=f'first_run',
            reset_num_timesteps=True,
            progress_bar=True,
            callback=eval_callback,
            )

# Save train results and replay buffer for continuing training
model.save(f'models/ibrl_{run.id}')
model.save_replay_buffer(f'models/replay_buffer/ibrl_{run.id}')
run.finish()
