import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import HDBSCAN
import nibabel as nib
import os
from spatialmath import UnitQuaternion, SO3
from tqdm import tqdm


def generate_path(tlr, keep, axis1, axis2, axis3, enamel_max, offset, q):
    """
    :param t:
    :param keep: what to keep 'first' or 'last'
    :param axis1: zigzag direction e.g. 'y'
    :param axis2: overall direction e.g. 'x'
    :param max_margin: e.g. enamel_max_z
    :param offset: e.g. 10
    :return:
    """
    tlr_path = pd.DataFrame(columns=['x','y','z','qw','qx','qy','qz'])
    if axis3 == 'x':
        max_margin = enamel_max[0]
    elif axis3 == 'y':
        max_margin = enamel_max[1]
    elif axis3 == 'z':
        max_margin = enamel_max[2]

    for caries_ in tlr:
        df = pd.DataFrame(caries_)
        df.columns = ['x', 'y', 'z']

        # sorting df and drop duplicates
        df.sort_values(by=['x', 'y', 'z'], inplace=True, ignore_index=True)
        df.drop_duplicates(subset=[axis1, axis2], keep=keep, inplace=True, ignore_index=True)

        # sorting df into zigzag pattern
        overall_dir = df[axis2].unique()
        for i in range(math.ceil(len(overall_dir) / 2)):
            temp = df.loc[df[axis2] == overall_dir[2 * i]]
            df.loc[df[axis2] == overall_dir[2 * i]] = np.array(temp.sort_values(by=axis1, ascending=False))

        # add start and end point considering max margin and offset
        start_point = pd.DataFrame([df.iloc[0]], columns=['x', 'y', 'z'])
        end_point = pd.DataFrame([df.iloc[-1]], columns=['x', 'y', 'z'])
        start_point[axis3] = max_margin + offset
        end_point[axis3] = max_margin + offset
        df = pd.concat([start_point, df], ignore_index=True)
        df = pd.concat([df, end_point], ignore_index=True)
        df[['qw']] = q[0]
        df[['qx']] = q[1]
        df[['qy']] = q[2]
        df[['qz']] = q[3]
        tlr_path = pd.concat([tlr_path, df], ignore_index=True)

    return tlr_path


# Function to interpolate between two points
def interpolate_points(start, end):
    # Calculate the difference in x, y, z
    t_start, t_end = start[:3], end[:3]
    q_start, q_end = start[3:], end[3:]
    diff = t_end - t_start
    angle_diff = (q_end - q_start)/3
    assert np.count_nonzero(angle_diff) <= 1

    # Find the number of steps needed for each dimension to ensure increments are <= 1 voxel
    num_steps = int(np.max(np.abs(np.append(diff, angle_diff))))  # Get the largest distance to cover
    # Generate interpolated points
    interpolated_points = [
        t_start + step * (diff / num_steps) for step in range(1, num_steps + 1)
    ]
    interpolated_R = [
        (q_start + step * (angle_diff*3 / num_steps)) for step in range(1, num_steps + 1)
    ]
    return np.concatenate((interpolated_points, interpolated_R), axis=1) if num_steps else None


# Function to process the entire DataFrame
def interpolate_path(df):
    # List to store all points
    full_path = [df.iloc[0].values]  # Start with the first point

    # Iterate over each pair of consecutive waypoints
    for i in range(len(df) - 1):
        start = df.iloc[i].values
        end = df.iloc[i + 1].values
        # Interpolate between start and end
        interpolated_points = interpolate_points(start, end)
        if interpolated_points is not None:
            full_path.extend(interpolated_points)  # Add interpolated points to path

    # Convert to DataFrame
    interpolated_df = pd.DataFrame(full_path, columns=['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
    return interpolated_df  # Round to integer for voxel indices
    # return interpolated_df.round(0).astype(int)  # Round to integer for voxel indices


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


if __name__ == '__main__':

    # params
    tnum = 5
    ds = 4
    res = 0.102
    res_ds = res * ds
    visualize = False
    # voxel_size = 60

    # np load npy
    tooth_dir = f'dental_env/labels_augmented/'
    dirlist = os.listdir(tooth_dir)
    # fname = dirlist[np.random.randint(0, len(dirlist))]
    for fname in tqdm(dirlist):
        tooth = np.load(tooth_dir+fname)
        tooth = downsample_state(tooth, ds)
        voxel_size = tooth.shape[0]
        caries = np.argwhere(tooth == 1)
        enamel = np.argwhere(tooth == 2)

        # Enamel spatial info statistics
        offset = 0
        enamel_g = np.mean(np.append(enamel, caries, axis=0), axis=0)
        enamel_max = np.max(enamel, axis=0)
        enamel_min = np.min(enamel, axis=0)

        # Define path for each caries
        if 'top' in fname:
            init_pos = np.array([voxel_size/2, voxel_size/2, voxel_size])
            init_quat = UnitQuaternion()
            cutpath = pd.DataFrame([np.concatenate((init_pos, init_quat.A))],
                                   columns=['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
            t_path = generate_path([caries], keep='first', axis1='y', axis2='x', axis3='z',
                                   enamel_max=enamel_max, offset=offset, q=init_quat.A)
            t_path[['x', 'y']] += 1/2
            cutpath = pd.concat([cutpath, t_path], ignore_index=True)
        elif 'left' in fname:
            init_pos = np.array([voxel_size/2, voxel_size, voxel_size/2])
            init_quat = UnitQuaternion(SO3.RPY(-90, 0, 0, unit='deg'))
            cutpath = pd.DataFrame([np.concatenate((init_pos, init_quat.A))],
                                   columns=['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
            l_path = generate_path([caries], keep='first', axis1='z', axis2='x', axis3='y',
                                   enamel_max=enamel_max, offset=offset, q=init_quat.A)
            l_path[['x', 'z']] += 1/2
            cutpath = pd.concat([cutpath, l_path], ignore_index=True)
        else:
            init_pos = np.array([voxel_size/2, 0, voxel_size/2])
            init_quat = UnitQuaternion(SO3.RPY(90, 0, 0, unit='deg'))
            cutpath = pd.DataFrame([np.concatenate((init_pos, init_quat.A))],
                                    columns=['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
            r_path = generate_path([caries], keep='last', axis1='z', axis2='x', axis3='y',
                                   enamel_max=enamel_min, offset=-offset, q=init_quat.A)
            r_path[['y']] += 1
            r_path[['x', 'z']] += 1/2
            cutpath = pd.concat([cutpath, r_path], ignore_index=True)

        # interpolate
        cutpath[['x','y','z']] *= ds
        cutpath_ = interpolate_path(cutpath)
        # convert unit to mm
        cutpath[['x', 'y', 'z']] *= res
        cutpath_[['x', 'y', 'z']] *= res
        caries = (caries + 1/2) * res_ds
        enamel = (enamel + 1/2) * res_ds

        # plot of center of voxels
        if visualize:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            ax1.scatter(caries[:, 0], caries[:, 1], caries[:, 2], alpha=0.3)
            ax1.scatter(enamel[:, 0], enamel[:, 1], enamel[:, 2], alpha=0.1)
            ax1.set_xlabel('X [voxel]')
            ax1.set_ylabel('Y [voxel]')
            ax1.set_zlabel('Z [voxel]')

            ax3 = fig.add_subplot(1, 3, 2, projection='3d')
            ax3.scatter(caries[:, 0], caries[:, 1], caries[:, 2], alpha=0.3)
            ax3.scatter(enamel[:, 0], enamel[:, 1], enamel[:, 2], alpha=0.1)
            ax3.plot(cutpath['x'].values, cutpath['y'].values, cutpath['z'].values, color='red')
            ax3.set_xlabel('X [voxel]')
            ax3.set_ylabel('Y [voxel]')
            ax3.set_zlabel('Z [voxel]')

            ax4 = fig.add_subplot(1, 3, 3, projection='3d')
            ax4.scatter(caries[:, 0], caries[:, 1], caries[:, 2], alpha=0.3)
            ax4.scatter(enamel[:, 0], enamel[:, 1], enamel[:, 2], alpha=0.1)
            ax4.plot(cutpath_['x'].values, cutpath_['y'].values, cutpath_['z'].values, color='red')
            ax4.set_xlabel('X [voxel]')
            ax4.set_ylabel('Y [voxel]')
            ax4.set_zlabel('Z [voxel]')

            plt.show()

        cutpath_.to_csv(f'dental_env/demos_augmented/coverage/{fname[:-4]}.csv', index=False, header=False, sep=' ')

