import dental_env
import gymnasium as gym
import torch

# env = gym.make("DentalEnv-v0", render_mode="human", max_episode_steps=1024, down_sample=30)
# env = gym.make("DentalEnv5D-v1", render_mode="human", max_episode_steps=1024, down_sample=10)
env = gym.make("DentalEnv6D-v0", render_mode="human", max_episode_steps=1000, down_sample=10)


observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)
    print(reward)
    if terminated or truncated:
        env.close()
        observation, info = env.reset()

env.close()
