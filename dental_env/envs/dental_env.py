
import copy
import logging

import numpy as np
import trimesh
import open3d as o3d
import nibabel as nib
from spatialmath import SO3, SE3, UnitQuaternion
from scipy.ndimage import affine_transform

import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


class DentalEnvBase(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, down_sample=10):

        # Define settings
        self._ds = down_sample
        self._window_size = 512
        self._original_resolution = 0.034  # 34 micron per voxel
        self._resolution = self._original_resolution * self._ds  # resolution of each voxel

        # Initialize segmentations
        self._state_init = nib.load('dental_env/labels/tooth_2.nii.gz').get_fdata()  # may go reset function
        # data specific alignment - using affine transform
        # shape = self._state_init.shape
        # center = np.array(shape) / 2
        # transform = SE3.Trans(center[::-1])*SE3.Rt(SO3.RPY(0, -np.pi/2, 0), [0, 0, 0])*SE3.Trans(-center)
        # transform = transform.inv()
        # self._state_init = affine_transform(self._state_init, transform.A, order=0,
        #                                     output_shape=(shape[2], shape[1], shape[0]))
        self._state_init = self._state_init[::self._ds, ::self._ds, ::self._ds]  # down-sampling
        # data specific alignment - using simple numpy rot90
        self._state_init = np.rot90(self._state_init, k=1, axes=(0, 2))
        self._state_label = {
            "empty": 0,
            "decay": 1,
            "enamel": 2,
            "dentin": 3,
            "burr": 4,
        }
        self._channel = len(self._state_label)

        # Initialize burr
        self._burr_vis_init = o3d.io.read_triangle_mesh('dental_env/cad/burr.stl')
        self._burr_vis_init.transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))  # burr pointing -z
        self._burr_vis_init.scale(scale=1/self._resolution, center=[0, 0, 0])  # burr stl to match with voxel resolution
        self._burr_init = trimesh.load('dental_env/cad/burr.stl')
        self._burr_init.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        self._ee_vis_init = o3d.io.read_triangle_mesh('dental_env/cad/hand_piece_no_bur.stl')
        self._ee_vis_init.transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        self._ee_vis_init.translate((0, 0, 10))  # translate by burr length 10 mm
        self._ee_vis_init.scale(scale=1 / self._resolution, center=[0, 0, 0])

        # Define obs and action space
        # TODO: agent position in tooth base reference frame ranging from -1 to 1
        # TODO: change states representation from box to multibinary
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0, 0, 0]), high=np.array(self._state_init.shape)-1, dtype=np.int32),
                "states": spaces.Box(0, 1, shape=(self._channel, self._state_init.shape[0], self._state_init.shape[1],
                                                  self._state_init.shape[2]), dtype=np.int32),
            }
        )
        # TODO: change action representation from Discrete to Box(-1, 1, shape=(3,), dtype=np.int32)
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
            "decay_remained": np.sum(self._states[self._state_label['decay']]),
            "is_success": np.sum(self._states[self._state_label['decay']]) == 0
        }

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # agent initialization
        # self._agent_location = np.array([self._state_init.shape[0]//2, self._state_init.shape[1]//2,
        #                                  self._state_init.shape[2]-5]).astype(int)  # start from random
        self._agent_location = self.np_random.integers(low=[0, 0, int(self._state_init.shape[2]*4/5)],
                                                       high=self._state_init.shape, dtype=np.int32)  # start from random
        # state initialization
        self._states = np.zeros((self._channel, self._state_init.shape[0], self._state_init.shape[1],
                                 self._state_init.shape[2]), dtype=np.int32)
        self._states[self._state_label['empty']] = self._state_init == self._state_label['empty']
        self._states[self._state_label['decay']] = self._state_init == self._state_label['decay']
        self._states[self._state_label['enamel']] = self._state_init == self._state_label['enamel']
        self._states[self._state_label['dentin']] = self._state_init == self._state_label['dentin']

        # burr initialization
        self._burr = self._burr_init.copy()
        position = (self._agent_location - np.array(self._state_init.shape)//2) * self._resolution
        self._burr.apply_translation(position)
        self._burr_voxel = trimesh.voxel.creation.local_voxelize(self._burr, [0, 0, 0], self._resolution,
                                                                 int(np.max(self._state_init.shape)))
        self._burr_occupancy = self.crop_center(self._burr_voxel.matrix, self._state_init.shape[0],
                                                self._state_init.shape[1], self._state_init.shape[2])
        self._states[self._state_label['burr']] = self._burr_occupancy

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # action
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, a_min=0, a_max=self._state_init.shape
        )

        # burr pose update
        self._burr = self._burr_init.copy()
        position = (self._agent_location - np.array(self._state_init.shape)//2) * self._resolution
        self._burr.apply_translation(position)
        self._burr_voxel = trimesh.voxel.creation.local_voxelize(self._burr, [0, 0, 0], self._resolution,
                                                                 int(np.max(self._state_init.shape)))
        self._burr_occupancy = self.crop_center(self._burr_voxel.matrix, self._state_init.shape[0],
                                                self._state_init.shape[1], self._state_init.shape[2])

        # reward
        burr_decay_occupancy = self._states[self._state_label['decay'], self._burr_occupancy]
        burr_enamel_occupancy = self._states[self._state_label['enamel'], self._burr_occupancy]
        burr_dentin_occupancy = self._states[self._state_label['dentin'], self._burr_occupancy]
        reward_decay_removal = np.sum(burr_decay_occupancy)
        reward_enamel_removal = np.sum(burr_enamel_occupancy)
        reward_dentin_removal = np.sum(burr_dentin_occupancy)
        reward = 1000 * reward_decay_removal - 10 * reward_enamel_removal - 100 * reward_dentin_removal - 1

        # state
        self._states[self._state_label['decay'], self._burr_occupancy] = 0
        self._states[self._state_label['enamel'], self._burr_occupancy] = 0
        self._states[self._state_label['dentin'], self._burr_occupancy] = 0
        self._states[self._state_label['empty'], self._burr_occupancy] = 1
        self._states[self._state_label['burr']] = self._burr_occupancy

        # termination
        terminated = ~np.any(self._states[self._state_label['decay']])  # or reward_dentin_removal > 0  # no more decay

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
            self.window = o3d.visualization.Visualizer()
            self.window.create_window(window_name='Cut Path Episode', width=1080, height=1080, left=50, top=50, visible=True)
            self._states_voxel = self._np_to_voxels(self._states)
            self._ee_vis = copy.deepcopy(self._ee_vis_init)
            self._burr_vis = copy.deepcopy(self._burr_vis_init)
            self._burr_center = self._burr_vis.get_center()
            self._ee_center = self._ee_vis.get_center()
            # self._burr_voxel = o3d.geometry.VoxelGrid()
            # self._np_to_burr_voxels(self._burr_occupancy, self._burr_voxel)
            # print(self.burr_vis.get_center())
            self._burr_vis.translate(self._burr_center+self._agent_location + [0.5, 0.5, 0.5], relative=False)
            self._ee_vis.translate(self._ee_center+self._agent_location + [0.5, 0.5, 0.5], relative=False)
            self.window.add_geometry(self._states_voxel)
            self.window.add_geometry(self._burr_vis)
            self.window.add_geometry(self._ee_vis)
            self.window.add_geometry(self._bounding_box())
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1/self._resolution)
            self.window.add_geometry(frame)
            # self.window.add_geometry(self._burr_voxel)
            ctr = self.window.get_view_control()
            ctr.rotate(0, -200)

        if self.render_mode == "human":
            for idx in np.argwhere(self._burr_occupancy):
                self._states_voxel.remove_voxel(idx)
            self._burr_vis.translate(self._burr_center+self._agent_location + [0.5, 0.5, 0.5], relative=False)
            self._ee_vis.translate(self._ee_center+self._agent_location + [0.5, 0.5, 0.5], relative=False)
            self.window.update_geometry(self._states_voxel)
            self.window.update_geometry(self._burr_vis)
            self.window.update_geometry(self._ee_vis)
            # self._np_to_burr_voxels(self._burr_occupancy, self._burr_voxel)
            # self.window.update_geometry(self._burr_voxel)
            self.window.poll_events()
            self.window.update_renderer()

    @staticmethod
    def crop_center(voxel, cropx, cropy, cropz):
        # local voxelize function can voxelize burr into cube, so we need to crop it for smaller dimension
        x, y, z = voxel.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        startz = z // 2 - (cropz // 2)
        return voxel[startx:startx + cropx, starty:starty + cropy, startz:startz + cropz]

    def _bounding_box(self):
        x, y, z = self._state_init.shape
        points = np.array([
            [0, 0, 0],
            [x, 0, 0],
            [0, y, 0],
            [x, y, 0],
            [0, 0, z],
            [x, 0, z],
            [0, y, z],
            [x, y, z],
        ])
        lines = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        colors = [[1, 0, 0] for i in range(len(lines))]
        box = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        box.colors = o3d.utility.Vector3dVector(colors)
        return box

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
                    elif state[self._state_label['dentin'],x,y,z] == 1:
                        voxel.color = np.array([1, 0.7, 0])
                    voxel.grid_index = np.array([x,y,z])
                    voxel_grid.add_voxel(voxel)
        return voxel_grid

    def _np_to_burr_voxels(self, burr, voxel_grid):
        # voxel_grid = o3d.geometry.VoxelGrid()
        voxel_grid.clear()
        voxel_grid.voxel_size = 1
        for z in range(burr.shape[2]):
            for y in range(burr.shape[1]):
                for x in range(burr.shape[0]):
                    if burr[x, y, z] == 0:
                        continue
                    voxel = o3d.geometry.Voxel()
                    voxel.color = np.array([0, 0, 1])
                    voxel.grid_index = np.array([x, y, z])
                    voxel_grid.add_voxel(voxel)
        # return voxel_grid

    def close(self):
        if self.window is not None and self.render_mode == "human":
            self.window.close()
            self.window = None


class DentalEnv5D(DentalEnvBase):
    def __int__(self, render_mode=None, down_sample=10):
        super().__init__(render_mode=render_mode, down_sample=down_sample)

        # Define obs and action space
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(low=np.array([0, 0, 0]),
                                        high=np.array(self._state_init.shape) - 1, dtype=np.int32),
                "agent_rot": spaces.Box(low=np.array([-1, -1, -1, -1]),
                                        high=np.array([1, 1, 1, 1]), dtype=np.float32),
                "states": spaces.MultiBinary([self._channel, self._state_init.shape[0], self._state_init.shape[1],
                                              self._state_init.shape[2]])
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.int32)

    def _get_obs(self):
        return {"agent_pos": self._agent_location, "agent_rot": self._agent_rotation, "states": self._states}

    def reset(self, seed=None, options=None):

        # agent initialization
        self._agent_location = self.np_random.integers(low=[0, 0, int(self._state_init.shape[2]*4/5)],
                                                       high=self._state_init.shape, dtype=np.int32)  # start from random
        self._agent_rotation = np.array([1, 0, 0, 0])

        # state initialization
        self._states = np.zeros((self._channel, self._state_init.shape[0], self._state_init.shape[1],
                                 self._state_init.shape[2]), dtype=np.int32)
        self._states[self._state_label['empty']] = self._state_init == self._state_label['empty']
        self._states[self._state_label['decay']] = self._state_init == self._state_label['decay']
        self._states[self._state_label['enamel']] = self._state_init == self._state_label['enamel']
        self._states[self._state_label['dentin']] = self._state_init == self._state_label['dentin']

        # burr initialization
        self._burr = self._burr_init.copy()
        position = (self._agent_location - np.array(self._state_init.shape)//2) * self._resolution
        self._burr.apply_translation(position)
        self._burr.apply_transform(trimesh.transformations.quaternion_matrix(self._agent_rotation))
        self._burr_voxel = trimesh.voxel.creation.local_voxelize(self._burr, [0, 0, 0], self._resolution,
                                                                 int(np.max(self._state_init.shape)))
        self._burr_occupancy = self.crop_center(self._burr_voxel.matrix, self._state_init.shape[0],
                                                self._state_init.shape[1], self._state_init.shape[2])
        self._states[self._state_label['burr']] = self._burr_occupancy

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # action
        self._agent_location = np.clip(
            self._agent_location + action[:3], a_min=0, a_max=self._state_init.shape
        )
        self._agent_rotation = (UnitQuaternion(self._agent_location)
                                * UnitQuaternion(SO3.RPY(0, action[3], action[4]))).A

        # burr pose update
        self._burr = self._burr_init.copy()
        position = (self._agent_location - np.array(self._state_init.shape)//2) * self._resolution
        self._burr.apply_translation(position)
        self._burr.apply_transform(trimesh.transformations.quaternion_matrix(self._agent_rotation))
        self._burr_voxel = trimesh.voxel.creation.local_voxelize(self._burr, [0, 0, 0], self._resolution,
                                                                 int(np.max(self._state_init.shape)))
        self._burr_occupancy = self.crop_center(self._burr_voxel.matrix, self._state_init.shape[0],
                                                self._state_init.shape[1], self._state_init.shape[2])

        # reward
        burr_decay_occupancy = self._states[self._state_label['decay'], self._burr_occupancy]
        burr_enamel_occupancy = self._states[self._state_label['enamel'], self._burr_occupancy]
        burr_dentin_occupancy = self._states[self._state_label['dentin'], self._burr_occupancy]
        reward_decay_removal = np.sum(burr_decay_occupancy)
        reward_enamel_removal = np.sum(burr_enamel_occupancy)
        reward_dentin_removal = np.sum(burr_dentin_occupancy)
        reward = 1000 * reward_decay_removal - 10 * reward_enamel_removal - 100 * reward_dentin_removal - 1

        # state
        self._states[self._state_label['decay'], self._burr_occupancy] = 0
        self._states[self._state_label['enamel'], self._burr_occupancy] = 0
        self._states[self._state_label['dentin'], self._burr_occupancy] = 0
        self._states[self._state_label['empty'], self._burr_occupancy] = 1
        self._states[self._state_label['burr']] = self._burr_occupancy

        # termination
        terminated = ~np.any(self._states[self._state_label['decay']])  # or reward_dentin_removal > 0  # no more decay

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            self.window = o3d.visualization.Visualizer()
            self.window.create_window(window_name='Cut Path Episode', width=1080, height=1080, left=50, top=50, visible=True)
            self._states_voxel = self._np_to_voxels(self._states)
            self._ee_vis = copy.deepcopy(self._ee_vis_init)
            self._burr_vis = copy.deepcopy(self._burr_vis_init)
            self._burr_center = self._burr_vis.get_center()
            self._ee_center = self._ee_vis.get_center()
            # self._burr_voxel = o3d.geometry.VoxelGrid()
            # self._np_to_burr_voxels(self._burr_occupancy, self._burr_voxel)
            # print(self.burr_vis.get_center())
            self._burr_vis.translate(self._burr_center+self._agent_location + [0.5, 0.5, 0.5], relative=False)
            self._burr_vis.rotate(self._burr_vis.get_rotation_matrix_from_quaternion(self._agent_rotation))
            self._ee_vis.translate(self._ee_center+self._agent_location + [0.5, 0.5, 0.5], relative=False)
            self._ee_vis.rotate(self._ee_vis.get_rotation_matrix_from_quaternion(self._agent_rotation))
            self.window.add_geometry(self._states_voxel)
            self.window.add_geometry(self._burr_vis)
            self.window.add_geometry(self._ee_vis)
            self.window.add_geometry(self._bounding_box())
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1/self._resolution)
            self.window.add_geometry(frame)
            # self.window.add_geometry(self._burr_voxel)
            ctr = self.window.get_view_control()
            ctr.rotate(0, -200)

        if self.render_mode == "human":
            for idx in np.argwhere(self._burr_occupancy):
                self._states_voxel.remove_voxel(idx)

            self._ee_vis = copy.deepcopy(self._ee_vis_init)
            self._burr_vis = copy.deepcopy(self._burr_vis_init)
            self._burr_vis.translate(self._burr_center+self._agent_location + [0.5, 0.5, 0.5], relative=False)
            self._burr_vis.rotate(self._burr_vis.get_rotation_matrix_from_quaternion(self._agent_rotation))
            self._ee_vis.translate(self._ee_center+self._agent_location + [0.5, 0.5, 0.5], relative=False)
            self._ee_vis.rotate(self._ee_vis.get_rotation_matrix_from_quaternion(self._agent_rotation))
            self.window.update_geometry(self._states_voxel)
            self.window.update_geometry(self._burr_vis)
            self.window.update_geometry(self._ee_vis)
            # self._np_to_burr_voxels(self._burr_occupancy, self._burr_voxel)
            # self.window.update_geometry(self._burr_voxel)
            self.window.poll_events()
            self.window.update_renderer()
