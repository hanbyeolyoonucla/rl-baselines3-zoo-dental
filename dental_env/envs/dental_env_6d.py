
import copy
import logging

import numpy as np
import trimesh
import fcl
import open3d as o3d
import nibabel as nib
from spatialmath import SO3, SE3, UnitQuaternion
from scipy.ndimage import affine_transform

import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


class DentalEnv6D(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, down_sample=10, angle_res=3, coffset=True, collision_check=True,
                 tooth='tooth_4_1.0_0_0_0_0_0_0'):

        # Define settings
        self._ds = down_sample
        self._coffset = 0.5 if coffset else 0
        self._angle_resolution = angle_res  # burr orientation resolution 3 deg
        self._window_size = 512
        self._original_resolution = 0.034  # 34 micron per voxel
        self._resolution = self._original_resolution * self._ds  # resolution of each voxel
        self._col_check = collision_check
        self._collision = False

        # Initialize segmentations
        # self._state_init = nib.load('dental_env/labels/tooth_2.nii.gz').get_fdata()  # may go reset function
        # self._state_init = self._state_init[::self._ds, ::self._ds, ::self._ds]  # down-sampling
        # # data specific alignment - using simple numpy rot90
        # self._state_init = np.rot90(self._state_init, k=1, axes=(0, 2))
        self._state_init = np.load(f'dental_env/labels_augmented/{tooth}.npy')
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

        # Initialize end effector
        self._ee_vis_init = o3d.io.read_triangle_mesh('dental_env/cad/end_effector_no_bur.stl')
        self._ee_vis_init.transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
        self._ee_vis_init.scale(scale=1000 / self._resolution, center=[0, 0, 0])
        self._ee_vis_init.compute_vertex_normals()

        # Initialize jaw
        # self._jaw = o3d.io.read_triangle_mesh('dental_env/cad/combined_jaw.stl')
        self._jaw = o3d.io.read_triangle_mesh('dental_env/cad/maxilla_with_stent.stl')
        self._jaw.scale(scale=1/self._resolution, center=[0, 0, 0])
        self._jaw.translate(np.array(self._state_init.shape)//2)  # + np.array([0, 0, 1])*self._state_init.shape[2]//4
        self._jaw.compute_vertex_normals()

        # Define collision object
        if self._col_check:
            ee = fcl.BVHModel()
            ee.beginModel(len(self._ee_vis_init.vertices), len(self._ee_vis_init.triangles))
            ee.addSubModel(self._ee_vis_init.vertices, self._ee_vis_init.triangles)
            ee.endModel()
            self._ee_col = fcl.CollisionObject(ee)
            jaw = fcl.BVHModel()
            jaw.beginModel(len(self._jaw.vertices), len(self._jaw.triangles))
            jaw.addSubModel(self._jaw.vertices, self._jaw.triangles)
            jaw.endModel()
            self._jaw_col = fcl.CollisionObject(jaw)

        # Define obs and action space
        self.observation_space = spaces.Dict(
            {
                "burr_pos": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float64),
                "burr_rot": spaces.Box(low=np.array([-1, -1, -1, -1]),
                                        high=np.array([1, 1, 1, 1]), dtype=np.float64),
                "voxel": spaces.MultiBinary([self._channel, self._state_init.shape[0], self._state_init.shape[1],
                                              self._state_init.shape[2]]),
            }
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.int32)
        # self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.window_col = None
        self.clock = None

    def _get_obs(self):
        return {"burr_pos": self._agent_location_normalized, "burr_rot": self._agent_rotation, "voxel": self._states}

    def _get_info(self):
        return {
            "decay_remained": np.sum(self._states[self._state_label['decay']]),
            "is_success": np.sum(self._states[self._state_label['decay']]) == 0
        }

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # agent initialization
        self._agent_location = np.array([self._state_init.shape[0]//2,
                                         self._state_init.shape[1]//2,
                                         self._state_init.shape[2]-1])
        # self._agent_location = self.np_random.integers(low=[0, 0, int(self._state_init.shape[2]*2/4)],
        #                                                high=self._state_init.shape, dtype=np.int32)  # start from random
        self._agent_location_normalized = (self._agent_location - np.array(self._state_init.shape)//2) / \
                                          (np.array(self._state_init.shape)//2)
        self._agent_rotation = np.array([1, 0, 0, 0], dtype=np.float64)

        # state initialization
        self._states = np.zeros((self._channel, self._state_init.shape[0], self._state_init.shape[1],
                                 self._state_init.shape[2]), dtype=np.bool_)
        self._states[self._state_label['empty']] = self._state_init == self._state_label['empty']
        self._states[self._state_label['decay']] = self._state_init == self._state_label['decay']
        self._states[self._state_label['enamel']] = self._state_init == self._state_label['enamel']
        self._states[self._state_label['dentin']] = self._state_init == self._state_label['dentin']

        # burr initialization
        self._burr = self._burr_init.copy()
        position = (self._agent_location - np.array(self._state_init.shape)//2) * self._resolution
        self._burr.apply_transform(trimesh.transformations.quaternion_matrix(self._agent_rotation))
        self._burr.apply_translation(position)
        burr_voxel = trimesh.voxel.creation.local_voxelize(self._burr, np.ones(3)*self._resolution/2, self._resolution,
                                                           int(np.max(self._state_init.shape)//2))
        self._burr_occupancy = self.crop_center(burr_voxel.matrix, self._state_init.shape[0],
                                                self._state_init.shape[1], self._state_init.shape[2])
        self._states[self._state_label['burr']] = self._burr_occupancy
        self._states[self._state_label['empty']] = (self._states[self._state_label['empty']] &
                                                    ~self._burr_occupancy)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # action
        # action = action - 1  # [0 1 2] to [-1 0 1]
        self._agent_location = np.clip(
            self._agent_location + action[:3],
            a_min=0 + self._coffset, a_max=np.array(self._state_init.shape) - np.ones(3)*self._coffset
        )
        self._agent_location_normalized = (self._agent_location - np.array(self._state_init.shape)//2) / \
                                          (np.array(self._state_init.shape)//2)
        agent_rotation = (UnitQuaternion(self._agent_rotation)
                          * UnitQuaternion(SO3.RPY(self._angle_resolution*action[3], self._angle_resolution*action[4],
                                                   self._angle_resolution*action[5], unit='deg')))
        if agent_rotation.angvec()[0] >= np.pi/2:
            self._agent_rotation = UnitQuaternion.AngVec(np.pi/2, agent_rotation.angvec()[1]).A
        else:
            self._agent_rotation = agent_rotation.A

        # burr pose update
        self._burr = self._burr_init.copy()
        position = (self._agent_location - np.array(self._state_init.shape)//2) * self._resolution
        self._burr.apply_transform(trimesh.transformations.quaternion_matrix(self._agent_rotation))
        self._burr.apply_translation(position)
        burr_voxel = trimesh.voxel.creation.local_voxelize(self._burr, np.ones(3)*self._resolution/2, self._resolution,
                                                           int(np.max(self._state_init.shape)//2))
        self._burr_occupancy = self.crop_center(burr_voxel.matrix, self._state_init.shape[0],
                                                self._state_init.shape[1], self._state_init.shape[2])

        # collision check
        self._collision = False
        if self._col_check:
            self._ee_col.setTransform(fcl.Transform(self._agent_rotation, self._agent_location))
            request = fcl.CollisionRequest()
            result = fcl.CollisionResult()
            ret = fcl.collide(self._ee_col, self._jaw_col, request, result)
            if ret > 0:
                self._collision = True

        # reward
        burr_decay_occupancy = self._states[self._state_label['decay'], self._burr_occupancy]
        burr_enamel_occupancy = self._states[self._state_label['enamel'], self._burr_occupancy]
        burr_dentin_occupancy = self._states[self._state_label['dentin'], self._burr_occupancy]
        reward_decay_removal = np.sum(burr_decay_occupancy)
        reward_enamel_removal = np.sum(burr_enamel_occupancy)
        reward_dentin_removal = np.sum(burr_dentin_occupancy)
        reward = 1000*reward_decay_removal - 10*reward_enamel_removal - 100*reward_dentin_removal - 1\
                 - 100*self._collision

        # state
        self._states[self._state_label['decay'], self._burr_occupancy] = 0
        self._states[self._state_label['enamel'], self._burr_occupancy] = 0
        self._states[self._state_label['dentin'], self._burr_occupancy] = 0
        self._states[self._state_label['burr']] = self._burr_occupancy
        self._states[self._state_label['empty']] = (~self._states[self._state_label['decay']] &
                                                    ~self._states[self._state_label['enamel']] &
                                                    ~self._states[self._state_label['dentin']] &
                                                    ~self._states[self._state_label['burr']])

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
            self.window.create_window(window_name='Cut Path Episode', width=1080, height=1080,
                                      left=50, top=50, visible=True)

            self._states_voxel = o3d.geometry.VoxelGrid()
            self._burr_voxel = o3d.geometry.VoxelGrid()
            self._states_voxel.voxel_size = 1
            self._burr_voxel.voxel_size = 1
            self._initialize_state_voxels()
            self._update_burr_voxels()

            self._ee_vis = copy.deepcopy(self._ee_vis_init)
            self._burr_vis = copy.deepcopy(self._burr_vis_init)
            self._burr_center = self._burr_vis.get_center()
            self._ee_center = self._ee_vis.get_center()

            self._burr_vis.translate(self._burr_center+self._agent_location, relative=False)
            self._burr_vis.rotate(self._burr_vis.get_rotation_matrix_from_quaternion(self._agent_rotation),
                                  center=self._agent_location)
            self._ee_vis.translate(self._ee_center+self._agent_location, relative=False)
            self._ee_vis.rotate(self._ee_vis.get_rotation_matrix_from_quaternion(self._agent_rotation),
                                center=self._agent_location)

            self.window.add_geometry(self._states_voxel)
            # self.window.add_geometry(self._burr_voxel)
            self.window.add_geometry(self._burr_vis)

            self._burr_vis.rotate(self._burr_vis.get_rotation_matrix_from_quaternion(self._agent_rotation).transpose(),
                                  center=self._agent_location)
            self._ee_vis.rotate(self._ee_vis.get_rotation_matrix_from_quaternion(self._agent_rotation).transpose(),
                                center=self._agent_location)

            self.window.add_geometry(self._bounding_box())
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1/self._resolution)
            self.window.add_geometry(frame)

            ctr = self.window.get_view_control()
            ctr.rotate(0, -200)

            if self._col_check and self.window_col is None:
                self.window_col = o3d.visualization.Visualizer()
                self.window_col.create_window(window_name='Cut Path Episode - Collision Status',
                                              width=1080, height=1080, left=1130, top=50, visible=True)

                self.window_col.add_geometry(self._burr_vis)
                self.window_col.add_geometry(self._ee_vis)
                self.window_col.add_geometry(self._jaw)
                self.window_col.add_geometry(self._bounding_box())
                self.window_col.add_geometry(frame)

                ctr_col = self.window_col.get_view_control()
                ctr_col.rotate(0, -300)

        if self.render_mode == "human":

            for idx in np.argwhere(self._burr_occupancy):
                self._states_voxel.remove_voxel(idx)
            self._update_burr_voxels()

            self._burr_vis.translate(self._burr_center+self._agent_location, relative=False)
            self._burr_vis.rotate(self._burr_vis.get_rotation_matrix_from_quaternion(self._agent_rotation),
                                  center=self._agent_location)
            self._ee_vis.translate(self._ee_center+self._agent_location, relative=False)
            self._ee_vis.rotate(self._ee_vis.get_rotation_matrix_from_quaternion(self._agent_rotation),
                                center=self._agent_location)

            self.window.update_geometry(self._states_voxel)
            # self.window.update_geometry(self._burr_voxel)
            self.window.update_geometry(self._burr_vis)

            self.window.poll_events()
            self.window.update_renderer()

            if self._col_check:

                if self._collision:
                    self._ee_vis.paint_uniform_color(np.array([1, 0, 0]))
                else:
                    self._ee_vis.paint_uniform_color(np.array([0.7, 0.7, 0.7]))

                self.window_col.update_geometry(self._burr_vis)
                self.window_col.update_geometry(self._ee_vis)

                self.window_col.poll_events()
                self.window_col.update_renderer()

            self._burr_vis.rotate(self._burr_vis.get_rotation_matrix_from_quaternion(self._agent_rotation).transpose(),
                                  center=self._agent_location)
            self._ee_vis.rotate(self._ee_vis.get_rotation_matrix_from_quaternion(self._agent_rotation).transpose(),
                                center=self._agent_location)

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

    def _initialize_state_voxels(self):
        for z in range(self._states.shape[3]):
            for y in range(self._states.shape[2]):
                for x in range(self._states.shape[1]):
                    if self._states[self._state_label['empty'],x,y,z] == 1:
                        continue
                    voxel = o3d.geometry.Voxel()
                    if self._states[self._state_label['decay'],x,y,z] == 1:
                        voxel.color = np.array([1, 0, 0])
                    elif self._states[self._state_label['enamel'],x,y,z] == 1:
                        voxel.color = np.array([0, 1, 0])
                    elif self._states[self._state_label['dentin'],x,y,z] == 1:
                        voxel.color = np.array([1, 0.7, 0])
                    # elif state[self._state_label['burr'],x,y,z] == 1:  # not updating i.e. this function called once
                    #     voxel.color = np.array([0, 0, 1])
                    voxel.grid_index = np.array([x,y,z])
                    self._states_voxel.add_voxel(voxel)

    def _update_burr_voxels(self):
        self._burr_voxel.clear()
        self._burr_voxel.voxel_size = 1
        for z in range(self._burr_occupancy.shape[2]):
            for y in range(self._burr_occupancy.shape[1]):
                for x in range(self._burr_occupancy.shape[0]):
                    if self._burr_occupancy[x, y, z] == 0:
                        continue
                    voxel = o3d.geometry.Voxel()
                    voxel.color = np.array([0, 0, 1])
                    voxel.grid_index = np.array([x, y, z])
                    self._burr_voxel.add_voxel(voxel)

    def close(self):
        if self.window is not None and self.render_mode == "human":
            self.window.close()
            self.window = None

        if self.window_col is not None and self.render_mode == "human":
            self.window_col.close()
            self.window_col = None