import dental_env
import gymnasium as gym
import torch
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
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium.wrappers import TransformReward

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 10_000,
}
run = wandb.init(
    project="dental_td3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)

tnum = 5
env = gym.make("DentalEnv6D-v0", render_mode=None, max_episode_steps=500, down_sample=10,
               tooth=f"tooth_{tnum}_1.0_0_0_0_0_0_0")
env = TransformReward(env, lambda r: np.sign(r) * np.log(1+np.abs(r)))

model = TD3(config["policy_type"], env, verbose=1,
            buffer_size=5_000,
            learning_starts=500,
            batch_size=256,
            train_freq=1,  # train every 100 rollout
            tensorboard_log=f"runs/td3_{run.id}",
            action_noise=NormalActionNoise(np.zeros(6), 0.1*np.ones(6)),
            policy_kwargs=hyperparams['DentalEnv6D-v0']['policy_kwargs'])

with h5py.File(f'dental_env/demonstrations/train_dataset_log_reward.hdf5', 'r') as f:
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
model.save(f'models/td3_{run.id}_v1')
model.save_replay_buffer(f'models/replay_buffer/td3_{run.id}_v1')
run.finish()
