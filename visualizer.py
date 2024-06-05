import dental_env
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from hyperparams.python.ppo_config import CustomCombinedExtractor

# print(torch.__version__)
# print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")

env = gym.make("DentalEnv3D-v0", render_mode="human", size=11, max_episode_steps=10)  # , render_mode="human"


# wrapped_env = gym.wrappers.FlattenObservation(env)
#
# policy_kwargs = dict(
#             features_extractor_class=CustomCombinedExtractor,
#             features_extractor_kwargs=dict(cnn_output_dim=512),
#             normalize_images=False
#         )
# model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)


# model = PPO("MlpPolicy", wrapped_env, verbose=1)

# model.learn(total_timesteps=10)

observation, info = env.reset(seed=42)
# observation, info = wrapped_env.reset(seed=42)

for _ in range(10):
    # action = wrapped_env.action_space.sample()
    action = env.action_space.sample()
    # action, _ = model.predict(observation)
    # observation, reward, terminated, truncated, info = wrapped_env.step(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    print(terminated)
    print(truncated)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
# wrapped_env.close()
