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
from ibrl import IBRL

tnum = 5
env = gym.make("DentalEnv6D-v0", render_mode=None, max_episode_steps=1000, down_sample=10,
               tooth=f"tooth_{tnum}_1.0_0_0_0_0_0_0")

model = IBRL("MultiInputPolicy", env, verbose=1, buffer_size=100,
             rl_bc_batch_ratio=0.5)
model.learn(total_timesteps=1000, progress_bar=True)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
