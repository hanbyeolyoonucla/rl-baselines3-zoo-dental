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

register(
     id="DentalEnv3D-v2",
     entry_point="dental_env.envs:DentalEnv3DSTLALL",
     max_episode_steps=1000,
)

register(
     id="DentalEnv5D-v0",
     entry_point="dental_env.envs:DentalEnv5D",
     max_episode_steps=300,
)
