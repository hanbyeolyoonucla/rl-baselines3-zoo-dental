from typing import Dict

import torch as th
from torch import nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.type_aliases import TensorDict
from wandb.integration.sb3 import WandbCallback
from hyperparams.python.ppo_config import CustomCombinedExtractor
import numpy as np

hyperparams = {
    "DentalEnv6D-v0": dict(
        env_wrapper=[{"gymnasium.wrappers.TransformReward": {"f": lambda r: np.sign(r) * np.log(1+np.abs(r))}}],
        # normalize=True,
        n_timesteps=30_000,
        policy="MultiInputPolicy",
        batch_size=256,
        gamma=0.99,
        learning_rate=0.0003,
        buffer_size=20_000,
        learning_starts=10_000,
        train_freq=100,
        noise_type='normal',
        noise_std=0.2,
        gradient_steps=1,
        # use_sde=True,
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=256),
            net_arch=dict(pi=[128, 128], qf=[400, 300]),
            normalize_images=False
        ),
    )
}
