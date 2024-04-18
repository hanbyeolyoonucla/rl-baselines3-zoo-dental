import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DentalEnv3D(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(3,), dtype=int),
                "states": spaces.MultiDiscrete(3 * np.ones((self.size, self.size, self.size))),
            }
        )
        self._state_label = {
            "empty": 0,
            "decay": 1,
            "enamel": 2,
        }

        self.action_space = spaces.Discrete(24)
        self._action_to_direction = {
            0: np.array([1, 0, 0]), 1: np.array([1, 1, 0]), 2: np.array([0, 1, 0]), 3: np.array([-1, 1, 0]),
            4: np.array([-1, 0, 0]), 5: np.array([-1, -1, 0]), 6: np.array([0, -1, 0]), 7: np.array([1, -1, 0]),
            8: np.array([1, 0, 1]), 9: np.array([1, 1, 1]), 10: np.array([0, 1, 1]), 11: np.array([-1, 1, 1]),
            12: np.array([-1, 0, 1]), 13: np.array([-1, -1, 1]), 14: np.array([0, -1, 1]), 15: np.array([1, -1, 1]),
            16: np.array([1, 0, -1]), 17: np.array([1, 1, -1]), 18: np.array([0, 1, -1]), 19: np.array([-1, 1, -1]),
            20: np.array([-1, 0, -1]), 21: np.array([-1, -1, -1]), 22: np.array([0, -1, -1]), 23: np.array([1, -1, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "states": self._states}

    def _get_info(self):
        return {
            "decay_remained": np.sum(self._states == self._state_label['decay'])
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array([np.ceil(self.size / 2) - 1, np.ceil(self.size / 2) - 1, self.size - 1],
                                        dtype=int)  # start from top center
        self._states = self.np_random.integers(1, 3, size=(self.size, self.size, self.size))
        self._states[:, :, -1] = 0  # empty space

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # action
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # reward
        burr_occupancy = self._states[self._agent_location[0], self._agent_location[1], self._agent_location[2]:]
        reward_decay_removal = np.sum(burr_occupancy == self._state_label['decay'])
        reward_enamel_removal = np.sum(burr_occupancy == self._state_label['enamel'])
        reward = 10 * reward_decay_removal - reward_enamel_removal

        # state
        self._states[self._agent_location[0], self._agent_location[1], self._agent_location[2]:] = 0

        # termination
        terminated = ~np.any(self._states == self._state_label['decay'])  # no more decay
        # if terminated:
        #   reward = reward + 10

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            fig = plt.figure()
            self.window = fig.add_subplot(projection='3d')
            self.window.set_xlabel('x')
            self.window.set_ylabel('y')
            self.window.set_zlabel('z')

        alpha = 0.8
        self.window.clear()
        burr = np.zeros_like(self._states, dtype=bool)
        burr[self._agent_location[0], self._agent_location[1], self._agent_location[2]:] = True
        self.window.voxels(burr, facecolors=[0, 0, 1], edgecolors='grey')
        self.window.voxels(self._states == self._state_label['decay'], facecolors=[1, 0, 0, alpha], edgecolors='grey')
        self.window.voxels(self._states == self._state_label['enamel'], facecolors=[0, 1, 0, alpha], edgecolors='grey')

        if self.render_mode == "human":
            plt.draw()
            plt.pause(1 / self.metadata["render_fps"])
        # else:
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        #     )

    def close(self):
        if self.window is not None:
            plt.close()
