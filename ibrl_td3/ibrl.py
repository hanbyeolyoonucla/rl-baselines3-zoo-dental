from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import h5py
import os

from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, DictReplayBufferSamples
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from ibrl_td3.policies import Actor, MlpPolicy, CnnPolicy, MultiInputPolicy, IBRLPolicy

SelfIBRL = TypeVar("SelfIBRL", bound="IBRL")

class IBRL(OffPolicyAlgorithm):
    """
    Imitation Bootstrapped Reinforcement Learning (IBRL)
    This implementation borrows code from original implementation (https://github.com/hengyuan-hu/ibrl)
    from Stable Baselines3 (https://github.com/DLR-RM/stable-baselines3)
    Paper: https://arxiv.org/abs/2311.02198

    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.
    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

     :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: IBRLPolicy
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
            self,
            policy: Union[str, type[IBRLPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 1e-3,
            buffer_size: int = 1_000_000,  # 1e6
            bc_buffer_size: int = 10_000,
            learning_starts: int = 100,
            batch_size: int = 256,
            rl_bc_batch_ratio: float = 1,  # 1 purely rl batch
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, tuple[int, str]] = 1,
            model_save_freq: int = 10_000,
            model_save_path: str = f'models/',
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = NormalActionNoise(0*np.ones(6), 0.2*np.ones(6)),
            replay_buffer_class: Optional[type[ReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[dict[str, Any]] = None,
            bc_replay_buffer_path: str = 'dental_env/demonstrations/train_dataset.hdf5',
            optimize_memory_usage: bool = False,
            policy_delay: int = 2,
            target_policy_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        # behavior cloning demonstration dataset buffer
        self.rl_bc_batch_ratio = rl_bc_batch_ratio
        self.bc_buffer_size = bc_buffer_size
        self.bc_replay_buffer_path = bc_replay_buffer_path

        # model save freq
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # TODO: set up bc replay buffer / move it to _setup_replay
        self.bc_replay_buffer = DictReplayBuffer(
                self.bc_buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )
        for fname in os.listdir(self.bc_replay_buffer_path):
            if not fname.endswith('hdf5') or 'tooth_3' in fname:
                continue
            with h5py.File(f'{self.bc_replay_buffer_path}/{fname}', 'r') as f:
                for demo in f.keys():
                    for i in range(len(f[demo]['acts'][:])):
                        self.bc_replay_buffer.add(
                            obs=dict(voxel=f[demo]['obs']['voxel'][i],
                                    burr_pos=f[demo]['obs']['burr_pos'][i],
                                    burr_rot=f[demo]['obs']['burr_rot'][i]),
                            next_obs=dict(voxel=f[demo]['obs']['voxel'][i+1],
                                        burr_pos=f[demo]['obs']['burr_pos'][i+1],
                                        burr_rot=f[demo]['obs']['burr_rot'][i+1]),
                            action=f[demo]['acts'][i],
                            reward=f[demo]['rews'][i],
                            done=f[demo]['info']['is_success'][i],
                            infos=[dict(placeholder=None)]
                        )

        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        use_actor_target: bool = True,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, episode_start, deterministic, use_actor_target)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            rl_batch_size = int(self.rl_bc_batch_ratio * batch_size)
            bc_batch_size = batch_size - rl_batch_size
            rl_replay_data = self.replay_buffer.sample(rl_batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # TODO: update replay_data so that it includes bc data
            bc_replay_data = self.bc_replay_buffer.sample(bc_batch_size, env=self._vec_normalize_env)
            replay_data = DictReplayBufferSamples(
                observations={key: th.cat([rl_replay_data.observations[key], bc_replay_data.observations[key]], dim=0) for key in rl_replay_data.observations.keys()},
                actions=th.cat([rl_replay_data.actions, bc_replay_data.actions], dim=0),
                next_observations={key: th.cat([rl_replay_data.next_observations[key], bc_replay_data.next_observations[key]], dim=0) for key in rl_replay_data.next_observations.keys()},
                # Only use dones that are not due to timeouts
                # deactivated by default (timeouts is initialized as an array of False)
                dones=th.cat([rl_replay_data.dones, bc_replay_data.dones], dim=0),
                rewards=th.cat([rl_replay_data.rewards, bc_replay_data.rewards], dim=0),
            )

            with th.no_grad():
                # Select action according to policy
                # TODO: select action according to IBRL policy
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                rl_next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                # rl_next_actions = self.actor_target(replay_data.next_observations)  # without noise
                bc_next_actions, _, _ = self.policy.bc_policy.forward(replay_data.next_observations, deterministic=True)
                rl_bc_next_actions = th.stack([rl_next_actions, bc_next_actions], dim=1)
                bsize, num_actions, _ = rl_bc_next_actions.size()

                # Compute the next Q values: min over all critics targets
                rl_next_q_values = th.cat(self.critic_target(replay_data.next_observations, rl_next_actions), dim=1)
                rl_next_q_values, _ = th.min(rl_next_q_values, dim=1, keepdim=True)
                bc_next_q_values = th.cat(self.critic_target(replay_data.next_observations, bc_next_actions), dim=1)
                bc_next_q_values, _ = th.min(bc_next_q_values, dim=1, keepdim=True)
                rl_bc_next_q_values = th.cat((rl_next_q_values, bc_next_q_values), dim=1)

                # TODO: epsilon greedy action
                greedy_action_idx = rl_bc_next_q_values.argmax(dim=1)
                greedy_action = rl_bc_next_actions[range(bsize), greedy_action_idx]
                next_q_values = rl_bc_next_q_values[range(bsize), greedy_action_idx].view(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss using q1
                actor_loss = -self.critic.q1_forward(replay_data.observations,
                                                     self.actor(replay_data.observations)).mean()
                # Compute actor loss using minimum q
                # actor_q_values = th.cat(self.critic.forward(replay_data.observations,
                #                                             self.actor(replay_data.observations)), dim=1)
                # rl_next_q_values, _ = th.min(actor_q_values, dim=1, keepdim=True)
                # actor_loss = -rl_next_q_values.mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)
            if self._n_updates % self.model_save_freq == 0:
                self.save(self.model_save_path+f'_{self._n_updates}')
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
            self: SelfIBRL,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "IBRL",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfIBRL:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []