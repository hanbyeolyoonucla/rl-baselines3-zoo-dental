import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

class DentalEnv5D(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "mesh"], "render_fps": 4}

    def __init__(self, render_mode=None, size=11):
        self.size = size
        self.window_size = 512
        self.burr_init = trimesh.load('dental_env/cad/burr.stl')
        self.burr_init.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        # self.burr_init.apply_scale(2.5)
        self.burr = self.burr_init.copy()

        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(0, self.size - 1, shape=(3,), dtype=int),
                "agent_ori": spaces.Box(-90, 90, shape=(2,), dtype=int),  # (Z)YX euler angle
                "states": spaces.MultiDiscrete(4 * np.ones((self.size, self.size, self.size), dtype=np.int32)),
            }
        )
        self._state_label = {
            "empty": 0,
            "decay": 1,
            "enamel": 2,
            "adjacent": 3,
        }

        self.action_space = spaces.Box(-1, 1, shape=(5,), dtype=int)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent_pos": self._agent_location[:3], "agent_ori": self._agent_location[3:], "states": self._states}

    def _get_info(self):
        return {
            "decay_remained": np.sum(self._states == self._state_label['decay'])
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_idx = 0

        self._agent_location = np.append(self.np_random.integers(0, self.size, size=2),
                                         [self.size - 1, 0, 0]).astype(int)  # start from random
        # state initialization
        self._states = np.ones((self.size, self.size, self.size), dtype=int) * 2
        decay_position = self.np_random.integers(low=[1, 1, 0], high=[self.size - 1, self.size - 1, self.size - 1],
                                                 size=(5, 3))
        decay_size = self.np_random.integers(low=[1, 1, 1],
                                             high=[self.size * 2 // 3, self.size * 2 // 3, self.size * 2 // 3],
                                             size=(5, 3))

        # decay 1
        aa = np.append(1, decay_position[0, 1:])
        bb = np.clip(aa + decay_size[0, :], 0, self.size - 1)
        self._states[aa[0]:bb[0], aa[1]:bb[1], aa[2]:bb[2]] = 1
        # decay 2
        aa = np.append(self.size - 2, decay_position[1, 1:])
        bb = np.clip(aa - decay_size[1, :], 0, self.size - 1)
        self._states[aa[0]:bb[0]:-1, aa[1]:bb[1]:-1, aa[2]:bb[2]:-1] = 1
        # decay 3
        aa = np.array([decay_position[2, 0], 1, decay_position[2, 2]])
        bb = np.clip(aa + decay_size[2, :], 0, self.size - 1)
        self._states[aa[0]:bb[0], aa[1]:bb[1], aa[2]:bb[2]] = 1
        # decay 4
        aa = np.array([decay_position[3, 0], self.size - 2, decay_position[3, 2]])
        bb = np.clip(aa - decay_size[3, :], 0, self.size - 1)
        self._states[aa[0]:bb[0]:-1, aa[1]:bb[1]:-1, aa[2]:bb[2]:-1] = 1
        # decay 5
        aa = np.append(decay_position[4, 0:2], self.size - 2)
        bb = np.clip(aa - decay_size[4, :], 0, self.size - 1)
        self._states[aa[0]:bb[0]:-1, aa[1]:bb[1]:-1, aa[2]:bb[2]:-1] = 1

        self._states[:, :, -1] = 0  # empty space
        self._states[:, 0, :] = 0  # empty space
        self._states[:, -1, :] = 0  # empty space
        self._states[0, 1:-1, 0:-1] = 3  # adjacent
        self._states[-1, 1:-1, 0:-1] = 3  # adjacent

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "mesh":
            self._render_frame()

        return observation, info

    def step(self, action):


        self.step_idx = self.step_idx + 1
        # action
        action[3:] = action[3:] * 5  # denormalize angle
        self._agent_location = np.clip(
            self._agent_location + action, [0, 0, 0, -90, -90], [self.size - 1, self.size - 1, self.size - 1, 90, 90]
        )

        # burr pose update
        burr_position = self._agent_location[:3] + np.array([0.5, 0.5, 0])  # voxel position correction
        burr_zyx_euler = np.append(0, self._agent_location[3:]) * np.pi / 180
        burr_rotation = trimesh.transformations.euler_matrix(burr_zyx_euler[0], burr_zyx_euler[1], burr_zyx_euler[2], 'rzyx')
        self.burr = self.burr_init.copy()
        self.burr.apply_transform(burr_rotation)
        self.burr.apply_translation(burr_position)
        volume_pcs = trimesh.sample.volume_mesh(self.burr, 1000)
        surface_pcs = trimesh.sample.sample_surface(self.burr, 1000)[0]
        burr_pcs = np.concatenate((volume_pcs, np.array(surface_pcs)))

        # burr occupancy
        burr_occupancy = np.zeros((self.size, self.size, self.size), dtype=bool)
        occupancy_idx = np.array([idx for idx in burr_pcs.astype(int) if np.all(idx < self.size) and np.all(idx >= 0)])
        if occupancy_idx.size > 0:
            burr_occupancy[occupancy_idx[:, 0], occupancy_idx[:, 1], occupancy_idx[:, 2]] = True

        # reward
        reward_decay_removal = np.sum(burr_occupancy & (self._states == self._state_label['decay']))
        reward_enamel_removal = np.sum(burr_occupancy & (self._states == self._state_label['enamel']))
        reward_adjacent_removal = np.sum(burr_occupancy & (self._states == self._state_label['adjacent']))
        reward = 10 * reward_decay_removal - 3 * reward_enamel_removal - 10 * reward_adjacent_removal - 1

        # state
        self._states[burr_occupancy] = 0

        # termination
        terminated = ~np.any(self._states == self._state_label['decay'])  # no more decay
        # if terminated:
        #   reward = reward + 10

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "mesh":
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

        if self.window is None and self.render_mode == "mesh":
            self.window = trimesh.Scene()
            self.window.camera_transform = trimesh.transformations.compose_matrix(angles= [np.pi/6, 0, np.pi/4], translate=[self.size*2, -self.size, self.size*4])

        burr_position = self._agent_location[:3] + np.array([0.5, 0.5, 0])  # voxel position correction
        burr_zyx_euler = np.append(0, self._agent_location[3:]) * np.pi / 180
        burr_rotation = trimesh.transformations.euler_matrix(burr_zyx_euler[0], burr_zyx_euler[1],
                                                             burr_zyx_euler[2], 'rzyx')
        self.burr = self.burr_init.copy()
        self.burr.apply_transform(burr_rotation)
        self.burr.apply_translation(burr_position)

        if self.render_mode == "human":
            vertices = self.burr.vertices
            faces = self.burr.faces

            alpha = 0.7
            self.window.clear()
            self.window.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], alpha=0.3, triangles=faces,
                                     color='gray')
            self.window.voxels(self._states == self._state_label['decay'], facecolors=[1, 0, 0, alpha],
                               edgecolors='gray')
            self.window.voxels(self._states == self._state_label['enamel'], facecolors=[0, 1, 0, alpha],
                               edgecolors='gray')
            self.window.voxels(self._states == self._state_label['adjacent'], facecolors=[1, 0.7, 0, alpha],
                               edgecolors='gray')

            plt.draw()
            plt.pause(1 / self.metadata["render_fps"])
            # plt.savefig('logs/episodes/figure_step_%d' % self.step_idx)

        if self.render_mode == "mesh":
            alpha = 0.7
            self.window.geometry.clear()
            decay_voxel = trimesh.voxel.base.VoxelGrid(self._states == self._state_label['decay']).as_boxes([1,0,0, alpha])
            enamel_voxel = trimesh.voxel.base.VoxelGrid(self._states == self._state_label['enamel']).as_boxes([0,1,0, alpha])
            adjacent_voxel = trimesh.voxel.base.VoxelGrid(self._states == self._state_label['adjacent']).as_boxes([0,0,1, alpha])
            self.window.add_geometry([self.burr, decay_voxel, enamel_voxel, adjacent_voxel])

            self.window.show()
            # self.window.save_image()


    def close(self):
        if self.window is not None:
            plt.close()