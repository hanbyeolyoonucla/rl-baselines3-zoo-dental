import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import logging
import open3d as o3d
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


class DentalEnv3D(gym.Env):
    metadata = {"render_modes": ["pyplot", "open3d", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=11, channel=4):
        self.size = size
        self.window_size = 512

        self._state_label = {
            "empty": 0,
            "decay": 1,
            "enamel": 2,
            "adjacent": 3,
        }
        self.channel = len(self._state_label)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(3,), dtype=int),
                "states": spaces.Box(0, 1, shape=(self.channel, self.size, self.size, self.size), dtype=bool),
            }
        )

        self.action_space = spaces.Discrete(26)
        self._action_to_direction = {
            0: np.array([1, 0, 0]), 1: np.array([1, 1, 0]), 2: np.array([0, 1, 0]), 3: np.array([-1, 1, 0]),
            4: np.array([-1, 0, 0]), 5: np.array([-1, -1, 0]), 6: np.array([0, -1, 0]), 7: np.array([1, -1, 0]),
            8: np.array([1, 0, 1]), 9: np.array([1, 1, 1]), 10: np.array([0, 1, 1]), 11: np.array([-1, 1, 1]),
            12: np.array([-1, 0, 1]), 13: np.array([-1, -1, 1]), 14: np.array([0, -1, 1]), 15: np.array([1, -1, 1]),
            16: np.array([1, 0, -1]), 17: np.array([1, 1, -1]), 18: np.array([0, 1, -1]), 19: np.array([-1, 1, -1]),
            20: np.array([-1, 0, -1]), 21: np.array([-1, -1, -1]), 22: np.array([0, -1, -1]), 23: np.array([1, -1, -1]),
            24: np.array([0, 0, 1]), 25: np.array([0, 0, -1])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "states": self._states}

    def _get_info(self):
        return {
            "decay_remained": np.sum(self._states[self._state_label['decay']])
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # agent initialization
        self._agent_location = np.append(self.np_random.integers(0, self.size, size=2),
                                         self.size - 1).astype(int)  # start from random
        # state initialization
        self._states = np.zeros((self.channel, self.size, self.size, self.size), dtype=bool)
        self._decay = np.zeros((self.size, self.size, self.size), dtype=bool)
        self._empty = np.zeros((self.size, self.size, self.size), dtype=bool)
        self._enamel = np.zeros((self.size, self.size, self.size), dtype=bool)
        self._adjacent = np.zeros((self.size, self.size, self.size), dtype=bool)

        # burr initialization
        self._burr_occupancy = np.zeros((self.size, self.size, self.size), dtype=bool)
        self._burr_occupancy[self._agent_location[0], self._agent_location[1], self._agent_location[2]:] = True

        # random initialize decay
        decay_position = self.np_random.integers(low=[1, 1, 0], high=[self.size - 1, self.size - 1, self.size - 1],
                                                 size=(5, 3))
        decay_size = self.np_random.integers(low=[1, 1, 1],
                                             high=[self.size * 2 // 3, self.size * 2 // 3, self.size * 2 // 3],
                                             size=(5, 3))
        # decay 1: yz plane
        aa = np.append(1, decay_position[0, 1:])
        bb = np.clip(aa + decay_size[0, :], 0, self.size - 1)
        self._decay[aa[0]:bb[0], aa[1]:bb[1], aa[2]:bb[2]] = 1
        # decay 2: yz plane
        aa = np.append(self.size - 2, decay_position[1, 1:])
        bb = np.clip(aa - decay_size[1, :], 0, self.size - 1)
        self._decay[aa[0]:bb[0]:-1, aa[1]:bb[1]:-1, aa[2]:bb[2]:-1] = 1
        # decay 3
        aa = np.array([decay_position[2, 0], 1, decay_position[2, 2]])
        bb = np.clip(aa + decay_size[2, :], 0, self.size - 1)
        self._decay[aa[0]:bb[0], aa[1]:bb[1], aa[2]:bb[2]] = 1
        # decay 4
        aa = np.array([decay_position[3, 0], self.size - 2, decay_position[3, 2]])
        bb = np.clip(aa - decay_size[3, :], 0, self.size - 1)
        self._decay[aa[0]:bb[0]:-1, aa[1]:bb[1]:-1, aa[2]:bb[2]:-1] = 1
        # decay 5
        aa = np.append(decay_position[4, 0:2], self.size - 2)
        bb = np.clip(aa - decay_size[4, :], 0, self.size - 1)
        self._decay[aa[0]:bb[0]:-1, aa[1]:bb[1]:-1, aa[2]:bb[2]:-1] = 1

        self._empty[:, :, -1] = 1  # empty space
        self._empty[:, 0, :] = 1  # empty space
        self._empty[:, -1, :] = 1  # empty space
        self._adjacent[0, 1:-1, 0:-1] = 1  # adjacent
        self._adjacent[-1, 1:-1, 0:-1] = 1  # adjacent
        self._decay[self._empty & self._adjacent] = 0  # make sure no decay on empty and adjacent
        self._enamel[~self._decay & ~self._empty & ~self._adjacent] = 1  # enamel

        self._states[self._state_label['empty']] = self._empty
        self._states[self._state_label['decay']] = self._decay
        self._states[self._state_label['enamel']] = self._enamel
        self._states[self._state_label['adjacent']] = self._adjacent

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "pyplot" or self.render_mode == "open3d":
            self._render_frame()

        return observation, info

    def step(self, action):
        # action
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # reward
        self._burr_occupancy = np.zeros((self.size, self.size, self.size), dtype=bool)
        self._burr_occupancy[self._agent_location[0], self._agent_location[1], self._agent_location[2]:] = True
        burr_decay_occupancy = self._states[self._state_label['decay'], self._burr_occupancy]
        burr_enamel_occupancy = self._states[self._state_label['enamel'], self._burr_occupancy]
        burr_adjacent_occupancy = self._states[self._state_label['adjacent'], self._burr_occupancy]
        reward_decay_removal = np.sum(burr_decay_occupancy)
        reward_enamel_removal = np.sum(burr_enamel_occupancy)
        reward_adjacent_removal = np.sum(burr_adjacent_occupancy)
        reward = 30 * reward_decay_removal - 3 * reward_enamel_removal - 10 * reward_adjacent_removal - 1

        # state update
        self._states[self._state_label['decay'], self._burr_occupancy] = 0
        self._states[self._state_label['enamel'], self._burr_occupancy] = 0
        self._states[self._state_label['adjacent'], self._burr_occupancy] = 0
        self._states[self._state_label['empty'], self._burr_occupancy] = 1

        # termination
        terminated = ~np.any(self._states[self._state_label['decay']]) or reward_adjacent_removal > 0  # no more decay
        # if terminated:
        #     reward = reward + 50

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "pyplot" or self.render_mode == "open3d":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "pyplot":
            fig = plt.figure()
            self.window = fig.add_subplot(projection='3d')
            self.window.set_xlabel('x')
            self.window.set_ylabel('y')
            self.window.set_zlabel('z')

        if self.render_mode == "pyplot":
            alpha = 0.7
            self.window.clear()
            burr = np.zeros((self.size, self.size, self.size*2), dtype=bool)
            burr[self._agent_location[0], self._agent_location[1], self._agent_location[2]:self._agent_location[2]+self.size] = True
            self.window.voxels(burr, facecolors=[0, 0, 1], edgecolors='grey')
            self.window.voxels(self._states[self._state_label['decay']], facecolors=[1, 0, 0, 1], edgecolors='grey')
            self.window.voxels(self._states[self._state_label['enamel']], facecolors=[0, 1, 0, alpha], edgecolors='grey')
            self.window.voxels(self._states[self._state_label['adjacent']], facecolors=[1, 0.7, 0, alpha], edgecolors='grey')
            plt.draw()
            plt.pause(1 / self.metadata["render_fps"])

        if self.window is None and self.render_mode == "open3d":
            self.window = o3d.visualization.Visualizer()
            self.window.create_window(window_name='Open3D', width=540, height=540, left=50, top=50, visible=True)
            self._states_voxel = self._np_to_voxels(self._states)
            self._burr_voxel = self._np_to_burr_voxels(self._burr_occupancy)
            self.window.add_geometry(self._states_voxel)
            self.window.add_geometry(self._burr_voxel)

        if self.render_mode == "open3d":
            self._burr_voxel.clear()
            self._burr_voxel.voxel_size = 1
            for idx in np.argwhere(self._burr_occupancy):
                self._states_voxel.remove_voxel(idx)
                voxel = o3d.geometry.Voxel()
                voxel.grid_index = idx
                voxel.color = np.array([0, 0, 1])
                self._burr_voxel.add_voxel(voxel)
            self.window.update_geometry(self._states_voxel)
            self.window.update_geometry(self._burr_voxel)
            self.window.poll_events()
            self.window.update_renderer()

    def _np_to_voxels(self, state):
        voxel_grid = o3d.geometry.VoxelGrid()
        voxel_grid.voxel_size = 1
        for z in range(state.shape[3]):
            for y in range(state.shape[2]):
                for x in range(state.shape[1]):
                    if state[self._state_label['empty'],x,y,z] == 1:
                        continue
                    voxel = o3d.geometry.Voxel()
                    if state[self._state_label['decay'],x,y,z] == 1:
                        voxel.color = np.array([1, 0, 0])
                    elif state[self._state_label['enamel'],x,y,z] == 1:
                        voxel.color = np.array([0, 1, 0])
                    elif state[self._state_label['adjacent'],x,y,z] == 1:
                        voxel.color = np.array([1, 0.7, 0])
                    voxel.grid_index = np.array([x,y,z])
                    voxel_grid.add_voxel(voxel)
        return voxel_grid

    def _np_to_burr_voxels(self, burr):
        voxel_grid = o3d.geometry.VoxelGrid()
        voxel_grid.voxel_size = 1
        for z in range(burr.shape[2]):
            for y in range(burr.shape[1]):
                for x in range(burr.shape[1]):
                    if burr[x, y, z] == 0:
                        continue
                    voxel = o3d.geometry.Voxel()
                    voxel.color = np.array([0, 0, 1])
                    voxel.grid_index = np.array([x, y, z])
                    voxel_grid.add_voxel(voxel)
        return voxel_grid

    def close(self):
        if self.window is not None and self.render_mode == "pyplot":
            plt.close()
        if self.window is not None and self.render_mode == "open3d":
            self.window.close()
            self.window = None


class DentalEnv3DSTL(gym.Env):
    metadata = {"render_modes": ["pyplot", "open3d", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=11):
        self.size = size
        self.window_size = 512

        self.burr_vis_init = o3d.io.read_triangle_mesh('dental_env/cad/burr.stl')
        self.burr_vis_init.transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        self.burr_init = trimesh.load('dental_env/cad/burr.stl')
        self.burr_init.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        # self.burr_init.apply_scale(2.5)
        self.burr = self.burr_init.copy()

        self._state_label = {
            "empty": 0,
            "decay": 1,
            "enamel": 2,
            "adjacent": 3,
        }
        self.channel = len(self._state_label)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(3,), dtype=int),
                "states": spaces.Box(0, 1, shape=(self.channel, self.size, self.size, self.size), dtype=bool),
            }
        )

        self.action_space = spaces.Discrete(26)
        self._action_to_direction = {
            0: np.array([1, 0, 0]), 1: np.array([1, 1, 0]), 2: np.array([0, 1, 0]), 3: np.array([-1, 1, 0]),
            4: np.array([-1, 0, 0]), 5: np.array([-1, -1, 0]), 6: np.array([0, -1, 0]), 7: np.array([1, -1, 0]),
            8: np.array([1, 0, 1]), 9: np.array([1, 1, 1]), 10: np.array([0, 1, 1]), 11: np.array([-1, 1, 1]),
            12: np.array([-1, 0, 1]), 13: np.array([-1, -1, 1]), 14: np.array([0, -1, 1]), 15: np.array([1, -1, 1]),
            16: np.array([1, 0, -1]), 17: np.array([1, 1, -1]), 18: np.array([0, 1, -1]), 19: np.array([-1, 1, -1]),
            20: np.array([-1, 0, -1]), 21: np.array([-1, -1, -1]), 22: np.array([0, -1, -1]), 23: np.array([1, -1, -1]),
            24: np.array([0, 0, 1]), 25: np.array([0, 0, -1])
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

        # agent initialization
        self._agent_location = np.append(self.np_random.integers(0, self.size, size=2),
                                         self.size - 1).astype(int)  # start from random
        # state initialization
        self._states = np.zeros((self.channel, self.size, self.size, self.size), dtype=bool)
        self._decay = np.zeros((self.size, self.size, self.size), dtype=bool)
        self._empty = np.zeros((self.size, self.size, self.size), dtype=bool)
        self._enamel = np.zeros((self.size, self.size, self.size), dtype=bool)
        self._adjacent = np.zeros((self.size, self.size, self.size), dtype=bool)

        # burr initialization
        self._burr_occupancy = np.zeros((self.size, self.size, self.size), dtype=bool)
        # self._burr_occupancy[self._agent_location[0], self._agent_location[1], self._agent_location[2]:] = True

        # random initialize decay
        decay_position = self.np_random.integers(low=[1, 1, 0], high=[self.size - 1, self.size - 1, self.size - 1],
                                                 size=(5, 3))
        decay_size = self.np_random.integers(low=[1, 1, 1],
                                             high=[self.size * 2 // 3, self.size * 2 // 3, self.size * 2 // 3],
                                             size=(5, 3))
        # decay 1: yz plane
        aa = np.append(1, decay_position[0, 1:])
        bb = np.clip(aa + decay_size[0, :], 0, self.size - 1)
        self._decay[aa[0]:bb[0], aa[1]:bb[1], aa[2]:bb[2]] = 1
        # decay 2: yz plane
        aa = np.append(self.size - 2, decay_position[1, 1:])
        bb = np.clip(aa - decay_size[1, :], 0, self.size - 1)
        self._decay[aa[0]:bb[0]:-1, aa[1]:bb[1]:-1, aa[2]:bb[2]:-1] = 1
        # decay 3
        aa = np.array([decay_position[2, 0], 1, decay_position[2, 2]])
        bb = np.clip(aa + decay_size[2, :], 0, self.size - 1)
        self._decay[aa[0]:bb[0], aa[1]:bb[1], aa[2]:bb[2]] = 1
        # decay 4
        aa = np.array([decay_position[3, 0], self.size - 2, decay_position[3, 2]])
        bb = np.clip(aa - decay_size[3, :], 0, self.size - 1)
        self._decay[aa[0]:bb[0]:-1, aa[1]:bb[1]:-1, aa[2]:bb[2]:-1] = 1
        # decay 5
        aa = np.append(decay_position[4, 0:2], self.size - 2)
        bb = np.clip(aa - decay_size[4, :], 0, self.size - 1)
        self._decay[aa[0]:bb[0]:-1, aa[1]:bb[1]:-1, aa[2]:bb[2]:-1] = 1

        self._empty[:, :, -1] = 1  # empty space
        self._empty[:, 0, :] = 1  # empty space
        self._empty[:, -1, :] = 1  # empty space
        self._adjacent[0, 1:-1, 0:-1] = 1  # adjacent
        self._adjacent[-1, 1:-1, 0:-1] = 1  # adjacent
        self._decay[self._empty & self._adjacent] = 0  # make sure no decay on empty and adjacent
        self._enamel[~self._decay & ~self._empty & ~self._adjacent] = 1  # enamel

        self._states[self._state_label['empty']] = self._empty
        self._states[self._state_label['decay']] = self._decay
        self._states[self._state_label['enamel']] = self._enamel
        self._states[self._state_label['adjacent']] = self._adjacent

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "pyplot" or self.render_mode == "open3d":
            self._render_frame()

        return observation, info

    def step(self, action):
        # action
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # burr pose update
        self.burr = self.burr_init.copy()
        self.burr.apply_translation(self._agent_location + [0.5, 0.5, 0])
        volume_pcs = trimesh.sample.volume_mesh(self.burr, 1000)
        surface_pcs = trimesh.sample.sample_surface(self.burr, 1000)[0]
        burr_pcs = np.concatenate((volume_pcs, np.array(surface_pcs)))

        # burr occupancy
        self._burr_occupancy = np.zeros((self.size, self.size, self.size), dtype=bool)
        occupancy_idx = np.array([idx for idx in burr_pcs.astype(int) if np.all(idx < self.size) and np.all(idx >= 0)])
        if occupancy_idx.size > 0:
            self._burr_occupancy[occupancy_idx[:,0],occupancy_idx[:,1],occupancy_idx[:,2]] = True

        # reward

        burr_decay_occupancy = self._states[self._state_label['decay'], self._burr_occupancy]
        burr_enamel_occupancy = self._states[self._state_label['enamel'], self._burr_occupancy]
        burr_adjacent_occupancy = self._states[self._state_label['adjacent'], self._burr_occupancy]
        reward_decay_removal = np.sum(burr_decay_occupancy)
        reward_enamel_removal = np.sum(burr_enamel_occupancy)
        reward_adjacent_removal = np.sum(burr_adjacent_occupancy)
        reward = 30 * reward_decay_removal - 3 * reward_enamel_removal - 10 * reward_adjacent_removal - 1

        # state
        self._states[self._state_label['decay'], self._burr_occupancy] = 0
        self._states[self._state_label['enamel'], self._burr_occupancy] = 0
        self._states[self._state_label['adjacent'], self._burr_occupancy] = 0
        self._states[self._state_label['empty'], self._burr_occupancy] = 1

        # termination
        terminated = ~np.any(self._states == self._state_label['decay']) or reward_adjacent_removal > 0 # no more decay
        # if terminated:
        #     reward = reward + 50

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "pyplot" or self.render_mode == "open3d":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "pyplot":
            fig = plt.figure()
            self.window = fig.add_subplot(projection='3d')
            self.window.set_xlabel('x')
            self.window.set_ylabel('y')
            self.window.set_zlabel('z')

        if self.render_mode == "pyplot":
            alpha = 0.7
            self.window.clear()

            self.burr = self.burr_init.copy()
            self.burr.apply_translation(self._agent_location + [0.5, 0.5, 0])
            vertices = self.burr.vertices
            faces = self.burr.faces
            self.window.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], alpha=0.3, triangles=faces, color='gray')
            self.window.voxels(self._states[self._state_label['decay']], facecolors=[1, 0, 0, alpha], edgecolors='gray')
            self.window.voxels(self._states[self._state_label['enamel']], facecolors=[0, 1, 0, alpha], edgecolors='gray')
            self.window.voxels(self._states[self._state_label['adjacent']], facecolors=[1, 0.7, 0, alpha], edgecolors='gray')

            plt.draw()
            plt.pause(1 / self.metadata["render_fps"])

        if self.window is None and self.render_mode == "open3d":
            self.window = o3d.visualization.Visualizer()
            self.window.create_window(window_name='Cut Path Episode', width=540, height=540, left=50, top=50, visible=True)
            self._states_voxel = self._np_to_voxels(self._states)
            # self._burr_voxel = self._np_to_burr_voxels(self._burr_occupancy)
            self.burr_vis = self.burr_vis_init
            self.burr_vis.translate(self._agent_location + [0.5, 0.5, 0], relative=False)
            self.window.add_geometry(self._states_voxel)
            self.window.add_geometry(self.burr_vis)

        if self.render_mode == "open3d":
            for idx in np.argwhere(self._burr_occupancy):
                self._states_voxel.remove_voxel(idx)
            self.burr_vis.translate(self._agent_location + [0.5, 0.5, 0], relative=False)
            self.window.update_geometry(self._states_voxel)
            self.window.update_geometry(self.burr_vis)
            self.window.poll_events()
            self.window.update_renderer()

    def _np_to_voxels(self, state):
        voxel_grid = o3d.geometry.VoxelGrid()
        voxel_grid.voxel_size = 1
        for z in range(state.shape[3]):
            for y in range(state.shape[2]):
                for x in range(state.shape[1]):
                    if state[self._state_label['empty'],x,y,z] == 1:
                        continue
                    voxel = o3d.geometry.Voxel()
                    if state[self._state_label['decay'],x,y,z] == 1:
                        voxel.color = np.array([1, 0, 0])
                    elif state[self._state_label['enamel'],x,y,z] == 1:
                        voxel.color = np.array([0, 1, 0])
                    elif state[self._state_label['adjacent'],x,y,z] == 1:
                        voxel.color = np.array([1, 0.7, 0])
                    voxel.grid_index = np.array([x,y,z])
                    voxel_grid.add_voxel(voxel)
        return voxel_grid

    def close(self):
        if self.window is not None and self.render_mode == "pyplot":
            plt.close()
        if self.window is not None and self.render_mode == "open3d":
            self.window.close()
            self.window = None

class DentalEnv3DSTLALL(gym.Env):
    metadata = {"render_modes": ["pyplot", "open3d", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=11):
        self.size = size
        self.window_size = 512
        dim = self.size - 1
        self.resolution = 10 / dim

        cary_mesh = trimesh.load('dental_env/cad/cary.stl')
        enamel_mesh = trimesh.load('dental_env/cad/enamel.stl')
        self.cary_voxel = trimesh.voxel.creation.local_voxelize(cary_mesh, [0, 0, 0], self.resolution, dim // 2)
        self.enamel_voxel = trimesh.voxel.creation.local_voxelize(enamel_mesh, [0, 0, 0], self.resolution, dim // 2)

        self.burr_vis_init = o3d.io.read_triangle_mesh('dental_env/cad/burr.stl')
        self.burr_vis_init.transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        self.burr_vis_init.scale(scale=1 / self.resolution,center=[0,0,0])
        self.burr_init = trimesh.load('dental_env/cad/burr.stl')
        self.burr_init.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        self.burr_init.apply_scale(1/self.resolution)
        self.burr = self.burr_init.copy()

        self._state_label = {
            "empty": 0,
            "decay": 1,
            "enamel": 2,
            "adjacent": 3,
        }
        self.channel = len(self._state_label)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(3,), dtype=int),
                "states": spaces.Box(0, 1, shape=(self.channel, self.size, self.size, self.size), dtype=bool),
            }
        )

        self.action_space = spaces.Discrete(26)
        self._action_to_direction = {
            0: np.array([1, 0, 0]), 1: np.array([1, 1, 0]), 2: np.array([0, 1, 0]), 3: np.array([-1, 1, 0]),
            4: np.array([-1, 0, 0]), 5: np.array([-1, -1, 0]), 6: np.array([0, -1, 0]), 7: np.array([1, -1, 0]),
            8: np.array([1, 0, 1]), 9: np.array([1, 1, 1]), 10: np.array([0, 1, 1]), 11: np.array([-1, 1, 1]),
            12: np.array([-1, 0, 1]), 13: np.array([-1, -1, 1]), 14: np.array([0, -1, 1]), 15: np.array([1, -1, 1]),
            16: np.array([1, 0, -1]), 17: np.array([1, 1, -1]), 18: np.array([0, 1, -1]), 19: np.array([-1, 1, -1]),
            20: np.array([-1, 0, -1]), 21: np.array([-1, -1, -1]), 22: np.array([0, -1, -1]), 23: np.array([1, -1, -1]),
            24: np.array([0, 0, 1]), 25: np.array([0, 0, -1])
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

        # agent initialization
        self._agent_location = np.append(self.np_random.integers(0, self.size, size=2),
                                         self.size - 1).astype(int)  # start from random
        # state initialization
        self._states = np.zeros((self.channel, self.size, self.size, self.size), dtype=bool)
        self._decay = self.cary_voxel.matrix
        self._enamel = self.enamel_voxel.matrix
        self._empty = ~self._decay & ~self._enamel
        self._states[self._state_label['empty']] = self._empty
        self._states[self._state_label['decay']] = self._decay
        self._states[self._state_label['enamel']] = self._enamel
        # self._states[self._state_label['adjacent']] = self._adjacent

        # burr initialization
        self._burr_occupancy = np.zeros((self.size, self.size, self.size), dtype=bool)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "pyplot" or self.render_mode == "open3d":
            self._render_frame()

        return observation, info

    def step(self, action):
        # action
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # burr pose update
        self.burr = self.burr_init.copy()
        self.burr.apply_translation(self._agent_location + [0.5, 0.5, 0])
        volume_pcs = trimesh.sample.volume_mesh(self.burr, 1000)
        surface_pcs = trimesh.sample.sample_surface(self.burr, 1000)[0]
        burr_pcs = np.concatenate((volume_pcs, np.array(surface_pcs)))

        # burr occupancy
        self._burr_occupancy = np.zeros((self.size, self.size, self.size), dtype=bool)
        occupancy_idx = np.array([idx for idx in burr_pcs.astype(int) if np.all(idx < self.size) and np.all(idx >= 0)])
        if occupancy_idx.size > 0:
            self._burr_occupancy[occupancy_idx[:,0],occupancy_idx[:,1],occupancy_idx[:,2]] = True

        # reward

        burr_decay_occupancy = self._states[self._state_label['decay'], self._burr_occupancy]
        burr_enamel_occupancy = self._states[self._state_label['enamel'], self._burr_occupancy]
        burr_adjacent_occupancy = self._states[self._state_label['adjacent'], self._burr_occupancy]
        reward_decay_removal = np.sum(burr_decay_occupancy)
        reward_enamel_removal = np.sum(burr_enamel_occupancy)
        reward_adjacent_removal = np.sum(burr_adjacent_occupancy)
        reward = 30 * reward_decay_removal - 3 * reward_enamel_removal - 10 * reward_adjacent_removal - 1

        # state
        self._states[self._state_label['decay'], self._burr_occupancy] = 0
        self._states[self._state_label['enamel'], self._burr_occupancy] = 0
        self._states[self._state_label['adjacent'], self._burr_occupancy] = 0
        self._states[self._state_label['empty'], self._burr_occupancy] = 1

        # termination
        terminated = ~np.any(self._states == self._state_label['decay']) or reward_adjacent_removal > 0 # no more decay
        # if terminated:
        #     reward = reward + 50

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "pyplot" or self.render_mode == "open3d":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "pyplot":
            fig = plt.figure()
            self.window = fig.add_subplot(projection='3d')
            self.window.set_xlabel('x')
            self.window.set_ylabel('y')
            self.window.set_zlabel('z')

        if self.render_mode == "pyplot":
            alpha = 0.7
            self.window.clear()

            self.burr = self.burr_init.copy()
            self.burr.apply_translation(self._agent_location + [0.5, 0.5, 0])
            vertices = self.burr.vertices
            faces = self.burr.faces
            self.window.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], alpha=0.3, triangles=faces, color='gray')
            self.window.voxels(self._states[self._state_label['decay']], facecolors=[1, 0, 0, alpha], edgecolors='gray')
            self.window.voxels(self._states[self._state_label['enamel']], facecolors=[0, 1, 0, alpha], edgecolors='gray')
            self.window.voxels(self._states[self._state_label['adjacent']], facecolors=[1, 0.7, 0, alpha], edgecolors='gray')

            plt.draw()
            plt.pause(1 / self.metadata["render_fps"])

        if self.window is None and self.render_mode == "open3d":
            self.window = o3d.visualization.Visualizer()
            self.window.create_window(window_name='Cut Path Episode', width=540, height=540, left=50, top=50, visible=True)
            self._states_voxel = self._np_to_voxels(self._states)
            # self._burr_voxel = self._np_to_burr_voxels(self._burr_occupancy)
            self.burr_vis = self.burr_vis_init
            self.burr_vis.translate(self._agent_location + [0.5, 0.5, 0], relative=False)
            self.window.add_geometry(self._states_voxel)
            self.window.add_geometry(self.burr_vis)

        if self.render_mode == "open3d":
            for idx in np.argwhere(self._burr_occupancy):
                self._states_voxel.remove_voxel(idx)
            self.burr_vis.translate(self._agent_location + [0.5, 0.5, 0], relative=False)
            self.window.update_geometry(self._states_voxel)
            self.window.update_geometry(self.burr_vis)
            self.window.poll_events()
            self.window.update_renderer()

    def _np_to_voxels(self, state):
        voxel_grid = o3d.geometry.VoxelGrid()
        voxel_grid.voxel_size = 1
        for z in range(state.shape[3]):
            for y in range(state.shape[2]):
                for x in range(state.shape[1]):
                    if state[self._state_label['empty'],x,y,z] == 1:
                        continue
                    voxel = o3d.geometry.Voxel()
                    if state[self._state_label['decay'],x,y,z] == 1:
                        voxel.color = np.array([1, 0, 0])
                    elif state[self._state_label['enamel'],x,y,z] == 1:
                        voxel.color = np.array([0, 1, 0])
                    elif state[self._state_label['adjacent'],x,y,z] == 1:
                        voxel.color = np.array([1, 0.7, 0])
                    voxel.grid_index = np.array([x,y,z])
                    voxel_grid.add_voxel(voxel)
        return voxel_grid

    def close(self):
        if self.window is not None and self.render_mode == "pyplot":
            plt.close()
        if self.window is not None and self.render_mode == "open3d":
            self.window.close()
            self.window = None