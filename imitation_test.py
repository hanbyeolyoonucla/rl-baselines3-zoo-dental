import dental_env
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
from imitation.data import rollout, types, serialize
from imitation.algorithms import bc
from stable_baselines3.ppo import PPO
from hyperparams.python.ppo_config import hyperparams
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

tnums = [2, 3]
trajectories = []
trajectories_accum = rollout.TrajectoryAccumulator()

for idx, tnum in enumerate(tnums):

    env = gym.make("DentalEnv6D-v0", render_mode="human", down_sample=10, tooth=f"tooth_{tnum}_1.0_0_0_0_0_0_0")
    # vec_env = make_vec_env("DentalEnv6D-v0", env_kwargs={'render_mode':"rgb_array", 'down_sample':10, 'tooth':f"tooth_{tnum}_1.0_0_0_0_0_0_0"})

    obs, info = env.reset(seed=42)
    wrapped_obs = types.maybe_wrap_in_dictobs(obs)
    trajectories_accum.add_step(dict(obs=wrapped_obs), idx)

    # test demonstration
    demons = pd.read_csv(f'dental_env/demonstrations/tooth_{tnum}_demonstration.csv')
    time_steps = len(demons)

    for itr in range(time_steps-1):
        # action = env.action_space.sample()
        action = demons.iloc[itr+1].to_numpy() - demons.iloc[itr].to_numpy()
        action[3:] = action[3:]//3
        obs, reward, terminated, truncated, info = env.step(action)
        wrapped_obs = types.maybe_wrap_in_dictobs(obs)
        trajectories_accum.add_step(dict(acts=action, rews=float(reward), obs=wrapped_obs, infos=info), idx)

        # if terminated or truncated:
        #     env.close()
        #     observation, info = env.reset()

    new_traj = trajectories_accum.finish_trajectory(idx, terminal=True)
    trajectories.append(new_traj)
    env.close()

serialize.save('dental_env/demonstrations', trajectories)
transitions = rollout.flatten_trajectories(trajectories)

tnum = 1
env = gym.make("DentalEnv6D-v0", render_mode="human", down_sample=10, tooth=f"tooth_{tnum}_1.0_0_0_0_0_0_0")
model = PPO('MultiInputPolicy', env, **hyperparams)
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    policy=model,
)
reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward before training: {reward_before_training}")
bc_trainer.train(n_epochs=1)
reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward after training: {reward_after_training}")
        # else:
        #     # Train an agent from scratch
        #     model = ALGOS[self.algo](
        #         env=env,
        #         tensorboard_log=self.tensorboard_log,
        #         seed=self.seed,
        #         verbose=self.verbose,
        #         device=self.device,
        #         **self._hyperp