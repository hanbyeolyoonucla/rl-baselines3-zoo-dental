import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common.logger import Logger

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None


from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

if TYPE_CHECKING:
    from stable_baselines3.common import base_class

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


class CustomEvalCallback(EvalCallback):
    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            callback_on_new_best: Optional[BaseCallback] = None,
            callback_after_eval: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
    ):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval, n_eval_episodes, eval_freq, log_path,
                         best_model_save_path, deterministic, render, verbose, warn)

        self._cre_buffer: List[float] = []
        self._mip_buffer: List[float] = []
        self._is_collision_buffer: List[bool] = []

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            maybe_cre = info.get("CRE")
            maybe_mip = info.get("MIP")
            maybe_is_collision = info.get("is_collision")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
            if maybe_cre is not None:
                self._cre_buffer.append(maybe_cre)
            if maybe_mip is not None:
                self._mip_buffer.append(maybe_mip)
            if maybe_is_collision is not None:
                self._is_collision_buffer.append(maybe_is_collision)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []
            self._cre_buffer = []
            self._mip_buffer = []
            self._is_collision_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            if len(self._is_collision_buffer) > 0:
                collision_rate = np.mean(self._is_collision_buffer)
                if self.verbose >= 1:
                    print(f"Collision rate: {100 * collision_rate:.2f}%")
                self.logger.record("eval/collision_rate", collision_rate)

            if len(self._cre_buffer) > 0:
                mean_cre, std_cre = np.mean(self._cre_buffer), np.std(self._cre_buffer)
                if self.verbose >= 1:
                    print(f"CRE: {mean_cre:.2f} +/- {std_cre:.2f}%")
                self.logger.record("eval/mean_cre", mean_cre)
                self.logger.record("eval/std_cre", std_cre)

            if len(self._mip_buffer) > 0:
                mean_mip, std_mip = np.mean(self._mip_buffer), np.std(self._mip_buffer)
                if self.verbose >= 1:
                    print(f"MIP: {mean_mip:.2f} +/- {std_cre:.2f}%")
                self.logger.record("eval/mean_mip", mean_mip)
                self.logger.record("eval/std_mip", std_mip)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
