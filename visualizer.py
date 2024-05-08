import dental_env
import gymnasium as gym
import torch

# print(torch.__version__)
# print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")

env = gym.make("DentalEnv5D-v0", render_mode="human", size=11)  # , render_mode="human"
observation, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
