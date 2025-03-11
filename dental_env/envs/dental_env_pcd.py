
import copy
import logging
import os

import numpy as np
import trimesh
import fcl
import open3d as o3d
import nibabel as nib
from spatialmath import SO3, SE3, UnitQuaternion
from scipy.ndimage import affine_transform
from itertools import product

import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


class DentalEnvPCD(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, collision_check=True, tooth=None):

        # Define settings
        self._jaw_offset = np.array([3, 3, 4.5])
        self._pos_resolution = 1  # action: burr position resolution 1000 micron
        self._angle_resolution = 1  # action: burr orientation resolution 1 deg
        self._resolution = 0.102  # resolution of each voxel: 102 micron
        self._col_check = collision_check
        self._collision = False
        self._tooth = tooth

        # Initialize segmentations
        if self._tooth:
            self._state_init = np.load(f'dental_env/labels_augmented/{self._tooth}.npy')
        self._tooth_dir = f'dental_env/labels_augmented/'
        self._state_shape = np.array([60, 60, 60])
        self._state_label = {"decay": 1, "enamel": 2, "dentin": 3}
        self._channel = len(self._state_label)

        # Initialize burr
        self._burr_init = o3d.io.read_triangle_mesh('dental_env/cad/burr.stl')
        self._burr_init.compute_vertex_normals()
        self._burr_center = self._burr_init.get_center()

        # Initialize end effector
        self._ee_vis_init = o3d.io.read_triangle_mesh('dental_env/cad/end_effector_no_bur.stl')
        self._ee_vis_init.compute_vertex_normals()

        # Initialize jaw
        self._jaw = o3d.io.read_triangle_mesh('dental_env/cad/jaw.stl')
        self._jaw.translate(self._jaw_offset)  # can be modified for different target tooth number
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
                "burr_pos": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                "burr_rot": spaces.Box(low=np.array([-1, -1, -1, -1]),
                                        high=np.array([1, 1, 1, 1]), dtype=np.float32),
                "voxel": spaces.MultiBinary([self._channel,
                                             self._state_shape[0], self._state_shape[1], self._state_shape[2]]),
            }
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.window_col = None
        self.clock = None

    def _get_obs(self):
        return {"burr_pos": self._agent_location_normalized, "burr_rot": self._agent_rotation, "voxel": self._states}

    def _get_info(self):
        curr_num_decay = len(self._decay_points)
        curr_num_enamel = len(self._enamel_points)
        curr_num_dentin = len(self._dentin_points)
        processed_cavity = (self._init_num_decay - curr_num_decay) + (self._init_num_enamel - curr_num_enamel) + (self._init_num_dentin - curr_num_dentin)
        return {
            "position": self._agent_location,
            "rotation": self._agent_rotation,
            "tooth": self._tooth,
            "decay_remained": curr_num_decay,
            "decay_removal": (self._init_num_decay - curr_num_decay) / self._init_num_decay,
            "enamel_damage": (self._init_num_enamel - curr_num_enamel),  # / self._init_num_enamel
            "dentin_damage": (self._init_num_dentin - curr_num_dentin),  #  / self._init_num_dentin
            "initial_caries": self._init_num_decay,
            "processed_cavity": processed_cavity,
            "CRE": curr_num_decay / self._init_num_decay,
            "MIP": processed_cavity / self._init_num_decay,
            "is_collision": self._collision,
            "is_success": curr_num_decay == 0
        }

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # agent initialization
        if not self._tooth:
            dirlist = os.listdir(self._tooth_dir)
            self._tooth = dirlist[np.random.randint(0, len(dirlist))]
            self._state_init = np.load(self._tooth_dir+self._tooth)
            self._tooth = self._tooth[:-4]  # remove .npy
        # check top left right and initialize agent accordingly
        if 'top' in self._tooth:
            self._agent_location = np.array([self._state_init.shape[0]/2 * self._resolution,
                                             self._state_init.shape[1]/2 * self._resolution,
                                             self._state_init.shape[2] * self._resolution], dtype=np.float32)
            self._agent_rotation = np.array([1, 0, 0, 0], dtype=np.float32)
        elif 'left' in self._tooth:
            self._agent_location = np.array([self._state_init.shape[0]/2 * self._resolution,
                                             self._state_init.shape[1] * self._resolution,
                                             self._state_init.shape[2]/2 * self._resolution], dtype=np.float32)
            self._agent_rotation = UnitQuaternion(SO3.RPY(-90, 0, 0, unit='deg')).A.astype(np.float32)
        else:
            self._agent_location = np.array([self._state_init.shape[0]/2 * self._resolution,
                                             0 * self._resolution,
                                             self._state_init.shape[2]/2 * self._resolution], dtype=np.float32)
            self._agent_rotation = UnitQuaternion(SO3.RPY(90, 0, 0, unit='deg')).A.astype(np.float32)

        # normalize location
        self._agent_location_normalized = ((self._agent_location - np.array(self._state_init.shape)/2 * self._resolution) /
                                           (np.array(self._state_init.shape)/2 * self._resolution)).astype(np.float32)

        # burr initialization
        self._burr = copy.deepcopy(self._burr_init)
        self._burr.translate(self._burr_center+self._agent_location, relative=False)
        self._burr.rotate(UnitQuaternion(self._agent_rotation).SO3().A,
                          center=self._agent_location)

        # pcd initialization
        self._decay_points = ((np.argwhere(self._state_init == self._state_label['decay']) + 1 / 2) * self._resolution).astype(np.float32)
        self._enamel_points = ((np.argwhere(self._state_init == self._state_label['enamel']) + 1 / 2) * self._resolution).astype(np.float32)
        self._dentin_points = ((np.argwhere(self._state_init == self._state_label['dentin']) + 1 / 2) * self._resolution).astype(np.float32)
        self._decay_pcd = o3d.geometry.PointCloud()
        self._enamel_pcd = o3d.geometry.PointCloud()
        self._dentin_pcd = o3d.geometry.PointCloud()
        self._decay_pcd.points = o3d.utility.Vector3dVector(self._decay_points)
        self._enamel_pcd.points = o3d.utility.Vector3dVector(self._enamel_points)
        self._dentin_pcd.points = o3d.utility.Vector3dVector(self._dentin_points)
        self._decay_pcd.paint_uniform_color([0.3, 0.3, 0.3])
        self._enamel_pcd.paint_uniform_color([0.95, 0.95, 0.99])
        self._dentin_pcd.paint_uniform_color([0.95, 0.95, 0.95])

        # state initialization
        decay_idx = (self._decay_points / self._resolution - 1/2).astype(int)
        enamel_idx = (self._enamel_points / self._resolution - 1/2).astype(int)
        dentin_idx = (self._dentin_points / self._resolution - 1/2).astype(int)
        self._states = np.zeros((self._channel, self._state_init.shape[0], self._state_init.shape[1],
                                 self._state_init.shape[2]), dtype=np.bool_)
        self._states[self._state_label['decay']-1, decay_idx[:, 0], decay_idx[:, 1], decay_idx[:, 2]] = 1
        self._states[self._state_label['enamel']-1, enamel_idx[:, 0], enamel_idx[:, 1], enamel_idx[:, 2]] = 1
        self._states[self._state_label['dentin']-1, dentin_idx[:, 0], dentin_idx[:, 1], dentin_idx[:, 2]] = 1

        # initial voxel counts
        self._init_num_decay = len(self._decay_points)
        self._init_num_enamel = len(self._enamel_points)
        self._init_num_dentin = len(self._dentin_points)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        # update agent position and orientation
        self._agent_location = np.clip(
            self._agent_location + self._pos_resolution * action[:3],
            a_min=0, a_max=np.array(self._state_init.shape) * self._resolution
        )
        self._agent_location_normalized = ((self._agent_location - np.array(self._state_init.shape)/2 * self._resolution) /
                                           (np.array(self._state_init.shape)/2 * self._resolution)).astype(np.float32)
        # agent_rotation = UnitQuaternion(self._agent_rotation) * UnitQuaternion(action[3:])
        agent_rotation = (UnitQuaternion(self._agent_rotation)
                          * UnitQuaternion(SO3.RPY(self._angle_resolution*action[3], self._angle_resolution*action[4],
                                                   self._angle_resolution*action[5], unit='deg')))
        if agent_rotation.angvec()[0] >= np.pi/2:
            self._agent_rotation = UnitQuaternion.AngVec(np.pi/2, agent_rotation.angvec()[1]).A.astype(np.float32)
        else:
            self._agent_rotation = agent_rotation.A.astype(np.float32)

        # burr pose update
        self._burr = copy.deepcopy(self._burr_init)
        self._burr.translate(self._burr_center+self._agent_location, relative=False)
        self._burr.rotate(UnitQuaternion(self._agent_rotation).SO3().A,
                          center=self._agent_location)

        # removal
        burr_geom = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy=self._burr)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(burr_geom)
        removal_caries = scene.compute_occupancy(self._decay_points).numpy().astype(bool)
        removal_enamel = scene.compute_occupancy(self._enamel_points).numpy().astype(bool)
        removal_dentin = scene.compute_occupancy(self._dentin_points).numpy().astype(bool)
        self._decay_points = self._decay_points[~removal_caries]
        self._enamel_points = self._enamel_points[~removal_enamel]
        self._dentin_points = self._dentin_points[~removal_dentin]
        self._decay_pcd.points = o3d.utility.Vector3dVector(self._decay_points)
        self._enamel_pcd.points = o3d.utility.Vector3dVector(self._enamel_points)
        self._dentin_pcd.points = o3d.utility.Vector3dVector(self._dentin_points)
        self._decay_pcd.paint_uniform_color([0.3, 0.3, 0.3])
        self._enamel_pcd.paint_uniform_color([0.95, 0.95, 0.99])
        self._dentin_pcd.paint_uniform_color([0.95, 0.95, 0.95])

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
        reward_decay_removal = np.sum(removal_caries)
        reward_enamel_removal = np.sum(removal_enamel)
        reward_dentin_removal = np.sum(removal_dentin)
        reward = (1000*reward_decay_removal
                  - 10*reward_enamel_removal
                  - 100*reward_dentin_removal
                  - 100*self._collision)*0.001
        # reward = 1000*reward_decay_removal - 1*reward_enamel_removal - 1*reward_dentin_removal\
        #          - 1*self._collision
        # reward = reward_decay_removal

        # state
        decay_idx = (self._decay_points / self._resolution - 1/2).astype(int)
        enamel_idx = (self._enamel_points / self._resolution - 1/2).astype(int)
        dentin_idx = (self._dentin_points / self._resolution - 1/2).astype(int)
        self._states = np.zeros((self._channel, self._state_init.shape[0], self._state_init.shape[1],
                                 self._state_init.shape[2]), dtype=np.bool_)
        self._states[self._state_label['decay']-1, decay_idx[:, 0], decay_idx[:, 1], decay_idx[:, 2]] = 1
        self._states[self._state_label['enamel']-1, enamel_idx[:, 0], enamel_idx[:, 1], enamel_idx[:, 2]] = 1
        self._states[self._state_label['dentin']-1, dentin_idx[:, 0], dentin_idx[:, 1], dentin_idx[:, 2]] = 1

        # termination
        terminated = ~np.any(self._states[self._state_label['decay']-1])  # or reward_dentin_removal > 0  # no more decay

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

            self._ee_vis = copy.deepcopy(self._ee_vis_init)
            self._burr_vis = copy.deepcopy(self._burr_init)
            self._burr_center = self._burr_vis.get_center()
            self._ee_center = self._ee_vis.get_center()

            self._burr_vis.translate(self._burr_center+self._agent_location, relative=False)
            self._burr_vis.rotate(UnitQuaternion(self._agent_rotation).SO3().A,
                                  center=self._agent_location)
            self._ee_vis.translate(self._ee_center+self._agent_location, relative=False)
            self._ee_vis.rotate(UnitQuaternion(self._agent_rotation).SO3().A,
                                center=self._agent_location)

            self.window.add_geometry(self._burr_vis)
            self.window.add_geometry(self._decay_pcd)
            self.window.add_geometry(self._enamel_pcd)
            self.window.add_geometry(self._dentin_pcd)

            self._burr_vis.rotate(UnitQuaternion(self._agent_rotation).SO3().A.transpose(),
                                  center=self._agent_location)
            self._ee_vis.rotate(UnitQuaternion(self._agent_rotation).SO3().A.transpose(),
                                center=self._agent_location)

            self.window.add_geometry(self._bounding_box(res=self._resolution))
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            self.window.add_geometry(frame)
            self._frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=self._agent_location)
            self.window.add_geometry(self._frame)

            ctr = self.window.get_view_control()
            ctr.rotate(0, -200)

            if self._col_check and self.window_col is None:
                self.window_col = o3d.visualization.Visualizer()
                self.window_col.create_window(window_name='Cut Path Episode - Collision Status',
                                              width=1080, height=1080, left=1130, top=50, visible=True)

                self.window_col.add_geometry(self._burr_vis)
                self.window_col.add_geometry(self._ee_vis)
                self.window_col.add_geometry(self._jaw)
                self.window_col.add_geometry(self._bounding_box(res=self._resolution))
                self.window_col.add_geometry(frame)

                ctr_col = self.window_col.get_view_control()
                ctr_col.rotate(0, -300)

        if self.render_mode == "human":

            self._burr_vis.translate(self._burr_center+self._agent_location, relative=False)
            self._burr_vis.rotate(UnitQuaternion(self._agent_rotation).SO3().A,
                                  center=self._agent_location)
            self._ee_vis.translate(self._ee_center+self._agent_location, relative=False)
            self._ee_vis.rotate(UnitQuaternion(self._agent_rotation).SO3().A,
                                center=self._agent_location)

            self.window.update_geometry(self._burr_vis)
            self.window.update_geometry(self._decay_pcd)
            self.window.update_geometry(self._enamel_pcd)
            self.window.update_geometry(self._dentin_pcd)

            self._frame.translate(self._agent_location, relative=False)
            self._frame.rotate(UnitQuaternion(self._agent_rotation).SO3().A,
                               center=self._agent_location)
            self.window.update_geometry(self._frame)

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

            self._burr_vis.rotate(UnitQuaternion(self._agent_rotation).SO3().A.transpose(),
                                  center=self._agent_location)
            self._ee_vis.rotate(UnitQuaternion(self._agent_rotation).SO3().A.transpose(),
                                center=self._agent_location)
            self._frame.rotate(UnitQuaternion(self._agent_rotation).SO3().A.transpose(),
                               center=self._agent_location)

    @staticmethod
    def crop_center(voxel, cropx, cropy, cropz):
        # local voxelize function can voxelize burr into cube, so we need to crop it for smaller dimension
        x, y, z = voxel.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        startz = z // 2 - (cropz // 2)
        return voxel[startx:startx + cropx, starty:starty + cropy, startz:startz + cropz]

    def _bounding_box(self, res):
        x, y, z = self._state_init.shape
        x *= res
        y *= res
        z *= res
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


    def close(self):
        if self.window is not None and self.render_mode == "human":
            self.window.close()
            self.window = None

        if self.window_col is not None and self.render_mode == "human":
            self.window_col.close()
            self.window_col = None
