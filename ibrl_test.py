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

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 50_000,
}
run = wandb.init(
    project="dental_ibrl",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

tnum = 5
env = gym.make("DentalEnv6D-v0", render_mode=None, max_episode_steps=1000, down_sample=10,
               tooth=f"tooth_{tnum}_1.0_0_0_0_0_0_0")

model = IBRL(config["policy_type"], env, verbose=1,
             buffer_size=10_000,
             bc_buffer_size=1_504,
             rl_bc_batch_ratio=0.7,
             learning_starts=0,
             train_freq=100,
             bc_replay_buffer_path=f'dental_env/demonstrations/train_dataset.hdf5',
             tensorboard_log=f"runs/{run.id}")  #
model.learn(total_timesteps=config["total_timesteps"],
            tb_log_name=f'first_run',
            reset_num_timesteps=True,
            progress_bar=True,
            # callback=WandbCallback(
            #     model_save_path=f"models/{run.id}",
            #     verbose=2,
            # )
            )
run.finish()

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(500):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
vec_env.close()
