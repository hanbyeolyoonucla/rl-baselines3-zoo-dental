from typing import Dict

import torch as th
from torch import nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.type_aliases import TensorDict


class CustomCNN3D(BaseFeaturesExtractor):
    """
    3D CNN for 3D label map of caries, enamel, etc.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        # normalized_image: bool = False,
        normalize_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        # We assume CxDXHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(observation_space.shape[0], 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # original structure
            # nn.Conv3d(4, 32, kernel_size=8, stride=4, padding=0),
            # nn.ReLU(),
            # nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=0),
            # nn.ReLU(),
            # nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0),
            # nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 512,
        # normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == 'states':
                extractors[key] = CustomCNN3D(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

hyperparams = {
    "DentalEnv3D-v0": dict(
        # env_wrapper=[{"gymnasium.wrappers.TimeLimit": {"max_episode_steps": 100}}],
        # normalize=True,
        n_envs=1,
        n_timesteps=20000.0,
        policy="MultiInputPolicy",
        batch_size=8,
        n_steps=8,
        gamma=0.95,
        learning_rate=7.77e-05,
        ent_coef=0.00429,
        clip_range=0.1,
        n_epochs=2,
        gae_lambda=0.9,
        max_grad_norm=5,
        vf_coef=0.19,
        use_sde=True,
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=256),
            normalize_images=False
        ),
    )
}
