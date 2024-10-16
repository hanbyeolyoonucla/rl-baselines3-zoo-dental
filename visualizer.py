import dental_env
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from hyperparams.python.ppo_config import CustomCombinedExtractor

env = gym.make("DentalEnv5D-v1", render_mode="human", max_episode_steps=1024, down_sample=50)

# policy_kwargs = dict(
#             features_extractor_class=CustomCombinedExtractor,
#             features_extractor_kwargs=dict(cnn_output_dim=256),
#             normalize_images=False
#         )
# model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
# model.load(
# model.learn(total_timesteps=10)

observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    # action, _ = model.predict(observation)
    _, reward, terminated, truncated, _ = env.step(action)
    print(reward)
    if terminated or truncated:
        env.close()
        observation, info = env.reset()

env.close()
