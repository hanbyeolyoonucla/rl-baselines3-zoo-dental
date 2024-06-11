import dental_env
import trimesh
import numpy as np
import open3d as o3d
import time

# print(torch.__version__)
# print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")

def _np_to_voxels(state):
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = 1
    for z in range(state.shape[3]):
        for y in range(state.shape[2]):
            for x in range(state.shape[1]):
                if state[2, x, y, z] == 1:
                    continue
                voxel = o3d.geometry.Voxel()
                if state[1, x, y, z] == 1:
                    voxel.color = np.array([1, 0, 0])
                elif state[0, x, y, z] == 1:
                    voxel.color = np.array([0, 1, 0])
                # elif state[self._state_label['adjacent'], x, y, z] == 1:
                #     voxel.color = np.array([1, 0.7, 0])
                voxel.grid_index = np.array([x, y, z])
                voxel_grid.add_voxel(voxel)
    return voxel_grid

def _np_to_burr_voxels(burr):
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


burr = trimesh.load('dental_env/cad/burr.stl')
cary = trimesh.load('dental_env/cad/cary.stl')
enamel = trimesh.load('dental_env/cad/enamel.stl')

burr.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))

# scene = trimesh.Scene()
# scene.add_geometry(burr)
# scene.add_geometry(cary)
# scene.add_geometry(enamel)
# scene.add_geometry(trimesh.creation.axis())
# scene.show()

resolution = 0.1
dim = 100
cary_voxel = trimesh.voxel.creation.voxelize(cary, resolution)
enamel_voxel = trimesh.voxel.creation.voxelize(enamel, resolution)
burr_voxel = trimesh.voxel.creation.voxelize(burr, resolution)
local_cary_voxel = trimesh.voxel.creation.local_voxelize(cary, [0, 0, 0], resolution, dim//2)
local_enamel_voxel = trimesh.voxel.creation.local_voxelize(enamel, [0, 0, 0], resolution, dim//2)
curr = time.time()
local_burr_voxel = trimesh.voxel.creation.local_voxelize(burr, [0, 0, 0], resolution, dim//2)
print(time.time() - curr)

burr_states = local_burr_voxel.matrix
burr_occ_voxel = _np_to_burr_voxels(burr_states)

channel = 3
size = 101
states = np.zeros((channel,size,size,size),dtype=bool)
states[0] = local_enamel_voxel.matrix & ~local_burr_voxel.matrix
states[1] = local_cary_voxel.matrix & ~local_burr_voxel.matrix
states[2] = ~states[0] & ~states[1]
states_voxel = _np_to_voxels(states)


occ = local_burr_voxel.matrix & local_cary_voxel.matrix
o3d.visualization.draw_geometries([states_voxel])

