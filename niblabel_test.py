import dental_env
import open3d as o3d
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
from spatialmath import SO3, SE3

def np_to_voxels(dict, arr):
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = 1
    for z in range(arr.shape[2]):
        for y in range(arr.shape[1]):
            for x in range(arr.shape[0]):
                if arr[x, y, z] == dict['empty']:
                    continue
                voxel = o3d.geometry.Voxel()
                if arr[x, y, z] == dict['decay']:
                    voxel.color = np.array([1, 0, 0])
                elif arr[x, y, z] == dict['enamel']:
                    voxel.color = np.array([0, 1, 0])
                elif arr[x, y, z] == dict['dentin']:
                    voxel.color = np.array([1, 0.7, 0])
                voxel.grid_index = np.array([x, y, z])
                voxel_grid.add_voxel(voxel)
    return voxel_grid

img = nib.load('dental_env/labels/tooth_2.nii.gz')
nparr = img.get_fdata()
nparr = nparr[::5,::5,::5]
# nparr = np.rot90(nparr, k=1, axes=(0, 2))
center = np.array(nparr.shape) / 2
state = {
            "empty": 0,
            "decay": 1,
            "enamel": 2,
            "dentin": 3,
        }
transform = SE3.Trans(center) * SE3.Rt(SO3.Rz(0, "deg"), [0, 0, 0]) * SE3.Trans(-center)
nparr_transformed = affine_transform(nparr, transform.A, order=0)

states_voxel = np_to_voxels(state, nparr)
states_voxel_transformed = np_to_voxels(state, nparr_transformed)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
o3d.visualization.draw_geometries([states_voxel, states_voxel_transformed, frame])