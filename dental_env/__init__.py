from gymnasium.envs.registration import register

register(
     id="DentalEnv2D-v0",
     entry_point="dental_env.envs:DentalEnv2D",
     max_episode_steps=300,
)

register(
     id="DentalEnv3D-v0",
     entry_point="dental_env.envs:DentalEnv3D",
     max_episode_steps=300,
)

register(
     id="DentalEnv3D-v1",
     entry_point="dental_env.envs:DentalEnv3DSTL",
     max_episode_steps=300,
)
