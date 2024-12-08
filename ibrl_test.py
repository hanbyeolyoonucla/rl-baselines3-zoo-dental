import dental_env
import gymnasium as gym
import torch
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
from gymnasium.wrappers import TransformReward

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 50_000,
}
run = wandb.init(
    project="dental_ibrl",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)

tnum = 5
env = gym.make("DentalEnv6D-v0", render_mode=None, max_episode_steps=500, down_sample=10,
               tooth=f"tooth_{tnum}_1.0_0_0_0_0_0_0")
env = TransformReward(env, lambda r: np.sign(r) * np.log(1+np.abs(r)))

model = IBRL(config["policy_type"], env, verbose=1,
             buffer_size=10_000,
             bc_buffer_size=1_504,
             rl_bc_batch_ratio=0.5,
             learning_starts=500,
             train_freq=1,  # train every 100 rollout
             model_save_freq=config['total_timesteps']//3,  # don't save
             model_save_path=f'D:/dental_RL_data/models/ibrl_{run.id}',
             bc_replay_buffer_path=f'dental_env/demonstrations/train_dataset_log_reward.hdf5',
             tensorboard_log=f"runs/ibrl_{run.id}",
             policy_kwargs=hyperparams['DentalEnv6D-v0']['policy_kwargs'])
model.learn(total_timesteps=config["total_timesteps"],
            tb_log_name=f'first_run',
            reset_num_timesteps=True,
            progress_bar=True,
            )
model.save(f'models/ibrl_{run.id}_v1')
model.save_replay_buffer(f'D:/dental_RL_data/replay_buffer/ibrl_{run.id}')
run.finish()
