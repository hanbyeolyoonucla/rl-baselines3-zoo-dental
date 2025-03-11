import dental_env
import open3d as o3d
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform, zoom
from spatialmath import SO3, SE3
from itertools import product
from tqdm import tqdm
from sklearn.cluster import HDBSCAN
import os, random


def visualize_pcd(arr, res=17e-3 * 2):
    caries = np.argwhere(arr == 1)
    enamel = np.argwhere(arr == 2)
    dentin = np.argwhere(arr == 3)
    caries = (np.asarray(caries) + 1 / 2) * res
    enamel = (np.asarray(enamel) + 1 / 2) * res
    dentin = (np.asarray(dentin) + 1 / 2) * res
    caries_pcd = o3d.geometry.PointCloud()
    caries_pcd.points = o3d.utility.Vector3dVector(caries)
    tooth_pcd = o3d.geometry.PointCloud()
    tooth_pcd.points = o3d.utility.Vector3dVector(np.concatenate((enamel, dentin)))
    ref = o3d.geometry.TriangleMesh.create_coordinate_frame()
    caries_pcd.paint_uniform_color([0.3, 0.3, 0.3])
    tooth_pcd.paint_uniform_color([0.9, 0.9, 0.95])
    bbox = bounding_box(arr, res)
    o3d.visualization.draw_geometries([ref, bbox, caries_pcd, tooth_pcd])


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


def bounding_box(state, res):
    x, y, z = state.shape
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
                x_start, x_end = i * ds, (i + 1) * ds
                y_start, y_end = j * ds, (j + 1) * ds
                z_start, z_end = k * ds, (k + 1) * ds

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

    # vis_only = True
    vis_only = False
    # alignment_check = True
    # alignment_check = False
    state = {
        "empty": 0,
        "decay": 1,
        "enamel": 2,
        "dentin": 3,
    }

    if vis_only:
        # tnum = 5
        # scale = 1.0
        # flip_axis, label = None, 1
        # idx = [124, 333, 413]
        # data = np.load(
        #     f'dental_env/labels_augmented_crop/tooth_{tnum}_{scale}_{flip_axis}_{label}_{idx[0]}_{idx[1]}_{idx[2]}.npy')
        dir = f'dental_env/labels_augmented/'
        fname = random.choice(os.listdir(dir))
        print(fname)
        data = np.load(dir+fname)
        visualize_pcd(data, res=17e-3 * 2 * 3)
        # data = np.load(f'dental_env/labels/tooth_4.npy')
        # visualize_pcd(data, res=34e-3)
    else:

        # data augmentation
        tnums = [2, 3, 4, 5]
        scales = [0.9, 1.0, 1.1]
        flip_axes = [None, 0, 1]

        combinations = list(product(tnums, scales, flip_axes))

        for combo in tqdm(combinations):
            tnum, scale, flip_axis = combo

            # load tooth
            nparr = np.load(f'dental_env/labels/tooth_{tnum}.npy')

            # scale
            nparr_scaled = zoom(nparr, scale, order=0)
            # print(scale)
            # visualize_pcd(nparr_scaled)

            # flip
            if flip_axis == 0 or flip_axis == 1:
                nparr_scaled = np.flip(nparr_scaled, axis=flip_axis)
                # print(flip_axis)
                # visualize_pcd(nparr_scaled)

            # crop: cluster > crop for each cluster > save
            crop_size = 180
            pad_size = crop_size//2
            nparr_pad = np.pad(nparr_scaled, ((pad_size,pad_size),(pad_size,pad_size),(pad_size,pad_size)))
            caries = np.argwhere(nparr_pad == 1)
            hdb = HDBSCAN(min_cluster_size=5, cluster_selection_epsilon=0, allow_single_cluster=True)
            hdb.fit(caries)
            labels = hdb.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            for label in range(n_clusters_):
                cmax = caries[labels == label].max(axis=0)
                cmin = caries[labels == label].min(axis=0)
                c = (cmax + cmin) // 2
                a = c - crop_size // 2
                b = c + crop_size // 2
                aa = (a+cmin) //2
                bb = (b+cmax) //2
                # check top / left / right
                top = np.sum(nparr_pad[c[0], c[1], c[2]:] == 2) + np.sum(nparr_pad[c[0], c[1], c[2]:] == 3)
                left = np.sum(nparr_pad[c[0], c[1]:, c[2]] == 2) + np.sum(nparr_pad[c[0], c[1]:, c[2]] == 3)
                right = np.sum(nparr_pad[c[0], :c[1], c[2]] == 2) + np.sum(nparr_pad[c[0], :c[1], c[2]] == 3)
                if top <= left and top <= right:
                    cut_type = 'top'
                elif left <= top and left <= right:
                    cut_type = 'left'
                else:
                    cut_type = 'right'
                nparr_new = nparr_pad.copy()
                convert_caries = caries[labels != label]
                nparr_new[convert_caries[:, 0], convert_caries[:, 1], convert_caries[:, 2]] = 0
                idx1 = [a[0], aa[0], bb[0]-crop_size]
                idx2 = [a[1], aa[1], bb[1]-crop_size]
                idx3 = [a[2], aa[2], bb[2]-crop_size]
                for idx in product(idx1, idx2, idx3):
                    crop_state = nparr_new[idx[0]:idx[0]+crop_size, idx[1]:idx[1]+crop_size, idx[2]:idx[2]+crop_size]
                    crop_state = downsample_state(crop_state, 3)
                    # visualize_pcd(crop_state, res=17e-3 * 2 * 3)
                    # save data
                    np.save(f'dental_env/labels_augmented/tooth_{tnum}_{scale}_{flip_axis}_{cut_type}_{label}_{idx[0]}_{idx[1]}_{idx[2]}.npy',
                            crop_state)

        data = np.load(f'dental_env/labels_augmented_crop/tooth_{tnum}_{scale}_{flip_axis}_{cut_type}_{label}_{idx[0]}_{idx[1]}_{idx[2]}.npy')
        visualize_pcd(data, res=17e-3 * 2 * 3)
        # states_voxel = np_to_voxels(state, data)
        # bbox = bounding_box(data)
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        # o3d.visualization.draw_geometries([states_voxel, bbox, frame])

