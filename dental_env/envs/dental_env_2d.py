import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class DentalEnv2D(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "states": spaces.MultiDiscrete(3 * np.ones((self.size, self.size))),
            }
        )
        self._state_label = {
            "empty": 0,
            "decay": 1,
            "enamel": 2,
        }

        self.action_space = spaces.Discrete(8)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([1, 1]),
            2: np.array([0, 1]),
            3: np.array([-1, 1]),
            4: np.array([-1, 0]),
            5: np.array([-1, -1]),
            6: np.array([0, -1]),
            7: np.array([1, -1]),
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

        self._agent_location = np.array([0, np.ceil(self.size / 2) - 1], dtype=int)  # start from top center
        self._states = self.np_random.integers(1, 3, size=(self.size, self.size))
        self._states[:1, :] = 0  # empty space

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
        burr_occupancy = self._states[:self._agent_location[0] + 1, self._agent_location[1]]
        reward_decay_removal = np.sum(burr_occupancy == self._state_label['decay'])
        reward_enamel_removal = np.sum(burr_occupancy == self._state_label['enamel'])
        reward = 10 * reward_decay_removal - reward_enamel_removal

        # state
        self._states[:self._agent_location[0] + 1, self._agent_location[1]] = 0

        # termination
        terminated = ~np.any(self._states == self._state_label['decay'])  # no more decay

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
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = int(self.window_size / self.size)

        # draw agent
        pygame.draw.rect(
            canvas,
            (0, 0, 255, 50),
            pygame.Rect(
                (0, pix_square_size * self._agent_location[1]),
                (pix_square_size * (self._agent_location[0] + 1), pix_square_size),
            )
        )

        # draw state
        for index, state in np.ndenumerate(self._states):
            if state == self._state_label['decay']:
                pygame.draw.rect(
                    canvas,
                    (255, 0, 0),
                    pygame.Rect(
                        pix_square_size * np.array(index),
                        (pix_square_size, pix_square_size),
                    ),
                )
            elif state == self._state_label['enamel']:
                pygame.draw.rect(
                    canvas,
                    (0, 255, 0),
                    pygame.Rect(
                        pix_square_size * np.array(index),
                        (pix_square_size, pix_square_size),
                    ),
                )

        # draw lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
