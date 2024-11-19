import dental_env
import open3d as o3d
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform, zoom
from spatialmath import SO3, SE3
from itertools import product
from tqdm import tqdm

def np_to_voxels_trans(dict, arr):
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
                voxel.grid_index = np.array([x+arr.shape[0], y, z])
                voxel_grid.add_voxel(voxel)
    return voxel_grid
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

def bounding_box(state):
    x, y, z = state.shape
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

def downsample_state(state, ds=10):
    w, h, d = state.shape
    dw, dh, dd = w//ds, h//ds, d//ds
    # Initialize the downsampled matrix
    downsampled_matrix = np.zeros((dw, dh, dd), dtype=int)

    # Iterate over each 10x10x10 block in the original matrix
    for i in range(dw):
        for j in range(dh):
            for k in range(dd):
                # Define the boundaries of the 10x10x10 block
                x_start, x_end = i * 10, (i + 1) * 10
                y_start, y_end = j * 10, (j + 1) * 10
                z_start, z_end = k * 10, (k + 1) * 10

                # Extract the sub-block
                sub_block = state[x_start:x_end, y_start:y_end, z_start:z_end]

                # Set downsampled matrix to 1 if there's at least one caries (value 1) in the sub-block
                if np.any(sub_block == 1):
                    downsampled_matrix[i, j, k] = 1
                else:
                    # Otherwise, use the most common value within the block for smoother downsampling
                    downsampled_matrix[i, j, k] = np.argmax(np.bincount(sub_block.flatten()))
    return downsampled_matrix


if __name__ == "__main__":

    vis_only = True
    # vis_only = False
    alignment_check = True
    # alignment_check = False
    tnum = 5

    state = {
        "empty": 0,
        "decay": 1,
        "enamel": 2,
        "dentin": 3,
    }

    if vis_only:
        scale, rz, ry, rx, tx, ty, tz = 1.0, 0, 0, 0, 0, 0, 0
        data = np.load(f'dental_env/labels_augmented/tooth_{tnum}_{scale}_{rx}_{ry}_{rz}_{tx}_{ty}_{tz}.npy')
        states_voxel = np_to_voxels(state, data)
        bbox = bounding_box(data)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        o3d.visualization.draw_geometries([states_voxel, bbox, frame])
    else:
        img = nib.load(f'dental_env/labels/tooth_{tnum}.nii.gz')
        nparr_init = img.get_fdata().astype(int)
        # nparr = nparr_init[::10,::10,::10]
        nparr = downsample_state(nparr_init, 10)
        # tooth 2: 1 (0,2), tooth3: -1 (0,2), tooth4: -1 (1,2) 1 (0, 1), tooth5: -1 (1,2) 1 (0, 1)
        # nparr = np.rot90(nparr, k=1, axes=(0, 2))
        # nparr = np.rot90(nparr, k=-1, axes=(0, 2))
        nparr = np.rot90(nparr, k=-1, axes=(1, 2))
        nparr = np.rot90(nparr, k=1, axes=(0, 1))

        input_shape = nparr.shape
        output_shape = nparr.shape
        input_center = np.array(input_shape) / 2
        output_center = np.array(output_shape) / 2

        # # tooth2 x20, tooth3 0, tooth4 x20, tooth5 0
        transform = SE3.Trans(output_center) * SE3.Rt(SO3.RPY([0, 0, 0], unit="deg"), [0, 0, 0]) * SE3.Trans(-input_center)
        transform = transform.inv()
        nparr = affine_transform(nparr, transform.A, order=0, output_shape=output_shape)
        original_shape = np.array(nparr.shape)
        if alignment_check:
            states_voxel = np_to_voxels(state, nparr)
            bbox = bounding_box(nparr)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
            o3d.visualization.draw_geometries([states_voxel, bbox, frame])

        scales = [0.9, 1.0, 1.1]
        rotations_z = [0, 45, 90, 135, 180, 225, 270, 315]
        rotations_y = [-10, 0, 10]
        rotations_x = [-10, 0, 10]
        translations_x = [-5, 0, 5]
        translations_y = [-5, 0, 5]
        translations_z = [-10, -5, 0]

        combinations = list(product(scales, rotations_z, rotations_y, rotations_x,
                                    translations_x, translations_y, translations_z))

        for combo in tqdm(combinations):
            scale, rz, ry, rx, tx, ty, tz = combo

            # scale
            nparr_scaled_temp = zoom(nparr, scale, order=0)
            scaled_shape = np.array(nparr_scaled_temp.shape)
            pad_size = (original_shape - scaled_shape) // 2
            if pad_size[0] > 0:
                nparr_scaled = np.pad(nparr_scaled_temp,
                                      pad_width=(
                                      (pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2])))
            elif pad_size[0] < 0:
                slice_size = -np.copy(pad_size)
                nparr_scaled = nparr_scaled_temp[slice_size[0]:scaled_shape[0] - slice_size[0],
                               slice_size[1]:scaled_shape[1] - slice_size[1],
                               slice_size[2]:scaled_shape[2] - slice_size[2]]
            else:
                nparr_scaled = np.copy(nparr_scaled_temp)

            # affine transform
            transform = SE3.Trans(output_center) * SE3.Rt(SO3.RPY([rx, ry, rz], unit="deg"), [tx, ty, tz]) * SE3.Trans(
                -input_center)
            transform = transform.inv()
            nparr_transformed = affine_transform(nparr_scaled, transform.A, order=0, output_shape=output_shape)
            np.save(f'dental_env/labels_augmented/tooth_{tnum}_{scale}_{rx}_{ry}_{rz}_{tx}_{ty}_{tz}.npy',
                    nparr_transformed)

        scale, rz, ry, rx, tx, ty, tz = 1.0, 0, 0, 0, 0, 0, 0
        data = np.load(f'dental_env/labels_augmented/tooth_{tnum}_{scale}_{rx}_{ry}_{rz}_{tx}_{ty}_{tz}.npy')
        states_voxel = np_to_voxels(state, data)
        bbox = bounding_box(data)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        o3d.visualization.draw_geometries([states_voxel, bbox, frame])

