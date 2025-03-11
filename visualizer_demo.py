import numpy as np
import open3d as o3d
import copy
import pandas as pd
from scipy.ndimage import affine_transform, zoom
from spatialmath import SO3, SE3, UnitQuaternion

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

def bounding_box_offset(state, res, offset):
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
    points += offset
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


def keyboard_to_action(key):
    act = np.array([0, 0, 0, 0, 0, 0])
    keymap = {'4': np.array([-1, 0, 0, 0, 0, 0]), '6': np.array([1, 0, 0, 0, 0, 0]),
              '1': np.array([0, -1, 0, 0, 0, 0]), '9': np.array([0, 1, 0, 0, 0, 0]),
              '2': np.array([0, 0, -1, 0, 0, 0]), '8': np.array([0, 0, 1, 0, 0, 0]),
              'a': np.array([0, 0, 0, -1, 0, 0]), 'd': np.array([0, 0, 0, 1, 0, 0]),
              'z': np.array([0, 0, 0, 0, -1, 0]), 'e': np.array([0, 0, 0, 0, 1, 0]),
              'x': np.array([0, 0, 0, 0, 0, -1]), 'w': np.array([0, 0, 0, 0, 0, 1]),
              }
    for c in key:
        act += keymap.get(c, np.array([0,0,0,0,0,0]))
    return act


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

    # np load npy
    use_rotation = True
    visualize = True
    ds = 3
    cut_type = ['occ', 'ling', 'bucc'][0]
    demo_type = ['heuristic', 'traction'][1]
    tnum, voxel_size = 5, 180
    # TOOTH 5
    # label, a = 0, np.array([182, 159, 399])  # occ
    # label, a = 1, np.array([28, 90, 394])  # occ
    label, a = 2, np.array([69, 151, 397])  # occ
    # TOOTH 4
    # label, a = 0, np.array([169, 81, 350])  # bucc
    # label, a = 1, np.array([52, 259, 389])  # occ
    # label, a = 2, np.array([107, 205, 404])  # occ
    # label, a = 3, np.array([100, 139, 412])  # occ
    tooth = np.load(f'dental_env/labels_crop/tooth_{tnum}_{label}_{voxel_size}_{a[0]}_{a[1]}_{a[2]}.npy')
    tooth = downsample_state(tooth, ds)
    tooth_original = np.load(f'dental_env/labels/tooth_{tnum}.npy')

    # argwhere and pcd + 1/2
    caries = np.argwhere(tooth == 1) + 1 / 2
    enamel = np.argwhere(tooth == 2) + 1 / 2
    dentin = np.argwhere(tooth == 3) + 1 / 2
    caries_original = np.argwhere(tooth_original == 1) + 1 / 2
    enamel_original = np.argwhere(tooth_original == 2) + 1 / 2
    dentin_original = np.argwhere(tooth_original == 3) + 1 / 2

    # convert to point clouds
    res = 17e-3 * 2
    res_ds = res * ds
    caries_pcd = o3d.geometry.PointCloud()
    tooth_pcd = o3d.geometry.PointCloud()
    caries_pcd.points = o3d.utility.Vector3dVector(np.asarray(caries) * res_ds)
    tooth_pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(enamel), np.asarray(dentin))) * res_ds)
    caries_pcd_original = o3d.geometry.PointCloud()
    tooth_pcd_original = o3d.geometry.PointCloud()
    caries_pcd_original.points = o3d.utility.Vector3dVector(np.asarray(caries_original) * res)
    tooth_pcd_original.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(enamel_original),
                                                                           np.asarray(dentin_original))) * res)

    # check visualization (crop vs original)
    ref = o3d.geometry.TriangleMesh.create_coordinate_frame()
    caries_pcd.paint_uniform_color([0.3, 0.3, 0.3])
    tooth_pcd.paint_uniform_color([0.95, 0.95, 0.99])
    bbox = bounding_box(tooth, res_ds)
    o3d.visualization.draw_geometries([ref, bbox, caries_pcd, tooth_pcd])
    caries_pcd_original.paint_uniform_color([0.3, 0.3, 0.3])
    tooth_pcd_original.paint_uniform_color([0.95, 0.95, 0.99])
    bbox_original = bounding_box(tooth_original, res)
    bbox_offset = bounding_box_offset(tooth, res, a * res)
    o3d.visualization.draw_geometries([ref, bbox_original, bbox_offset, caries_pcd_original, tooth_pcd_original])

    #############################################################
    # traction path planning
    #############################################################
    # bur initialize - translate/rotate to top/left/right center
    burr = o3d.io.read_triangle_mesh(f'dental_env/cad/burr.stl')
    burr.compute_vertex_normals()
    if cut_type == 'ling':
        init_pos = np.array([voxel_size // 2, voxel_size, voxel_size//2]) * res
        init_quat = UnitQuaternion(SO3.RPY(90, 180, 0, unit='deg'))
    elif cut_type == 'bucc':
        init_pos = np.array([voxel_size // 2, 0, voxel_size//2]) * res
        init_quat = UnitQuaternion(SO3.RPY(-90, 180, 0, unit='deg'))
    else:
        init_pos = np.array([voxel_size // 2, voxel_size // 2, voxel_size]) * res
        init_quat = UnitQuaternion(SO3.RPY(0, 180, 0, unit='deg'))

    burr.rotate(burr.get_rotation_matrix_from_quaternion(init_quat), center=np.zeros(3))
    burr_center = burr.get_center()
    burr.translate(init_pos + burr_center, relative=False)
    o3d.visualization.draw_geometries([ref, bbox, caries_pcd, tooth_pcd, burr])

    # initial pos, quat, caries, tooth
    curr_pos = init_pos
    curr_quat = UnitQuaternion()
    cutpath = [np.concatenate((curr_pos, (init_quat*curr_quat).A))]
    initial_caries = len(caries_pcd.points)
    initial_tooth = len(tooth_pcd.points)
    update_caries_pcd = copy.deepcopy(caries_pcd)
    update_tooth_pcd = copy.deepcopy(tooth_pcd)
    traverse_length = 0
    traverse_angle = 0

    if visualize:
        window1 = o3d.visualization.Visualizer()
        window1.create_window(window_name='Cut Path Episode', width=1080, height=1080,
                             left=1130, top=50, visible=True)
        window1.add_geometry(ref)
        window1.add_geometry(bbox)
        window1.add_geometry(update_caries_pcd)
        window1.add_geometry(update_tooth_pcd)
        window1.add_geometry(burr)
        ctr1 = window1.get_view_control()
        ctr1.set_up([0, 0, 1])
        ctr1.set_front([0, -1, 1])
        ctr1.set_lookat(curr_pos)
        window11 = o3d.visualization.Visualizer()
        window11.create_window(window_name='Cut Path Episode', width=1080, height=1080,
                             left=1130, top=1130, visible=True)
        window11.add_geometry(ref)
        window11.add_geometry(bbox)
        window11.add_geometry(update_caries_pcd)
        window11.add_geometry(update_tooth_pcd)
        window11.add_geometry(burr)
        ctr11 = window11.get_view_control()
        ctr11.set_up([0, 0, 1])
        ctr11.set_front([0, -1, 0])
        ctr11.set_lookat(curr_pos)
        ctr11.set_zoom(0.1)
        window2 = o3d.visualization.Visualizer()
        window2.create_window(window_name='Cut Path Episode', width=1080, height=1080,
                             left=50, top=50, visible=True)
        window2.add_geometry(ref)
        window2.add_geometry(bbox)
        window2.add_geometry(update_caries_pcd)
        window2.add_geometry(update_tooth_pcd)
        window2.add_geometry(burr)
        ctr2 = window2.get_view_control()
        ctr2.set_up([0, 0, 1])
        ctr2.set_front([-1, 0, 1])
        ctr2.set_lookat(curr_pos)
        window22 = o3d.visualization.Visualizer()
        window22.create_window(window_name='Cut Path Episode', width=1080, height=1080,
                             left=50, top=1130, visible=True)
        window22.add_geometry(ref)
        window22.add_geometry(bbox)
        window22.add_geometry(update_caries_pcd)
        window22.add_geometry(update_tooth_pcd)
        window22.add_geometry(burr)
        ctr22 = window22.get_view_control()
        ctr22.set_up([0, 0, 1])
        ctr22.set_front([-1, 0, 0])
        ctr22.set_lookat(curr_pos)
        ctr22.set_zoom(0.1)

    user_input = True
    while user_input != 'n':
        # calculate disgned distance vector and update pcd
        burr_geom = o3d.t.geometry.TriangleMesh.from_legacy(burr)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(burr_geom)  # we do not need the geometry ID for mesh

        # occupancy
        occ_caries = scene.compute_occupancy(np.asarray(update_caries_pcd.points, dtype=np.float32)).numpy().astype(bool)
        occ_tooth = scene.compute_occupancy(np.asarray(update_tooth_pcd.points, dtype=np.float32)).numpy().astype(bool)
        update_caries_pcd.points = o3d.utility.Vector3dVector(
            np.asarray(update_caries_pcd.points, dtype=np.float32)[~occ_caries])
        update_tooth_pcd.points = o3d.utility.Vector3dVector(
            np.asarray(update_tooth_pcd.points, dtype=np.float32)[~occ_tooth])
        update_caries_pcd.paint_uniform_color([0.3, 0.3, 0.3])
        update_tooth_pcd.paint_uniform_color([0.95, 0.95, 0.99])

        # performance metric
        processed_cavity = initial_caries + initial_tooth - len(update_caries_pcd.points) - len(update_tooth_pcd.points)
        print(f'IC: {initial_caries * (res ** 3)}')
        print(f'RC: {len(update_caries_pcd.points) * (res ** 3)}')
        print(f'PC: {processed_cavity * (res ** 3)}')
        print(f'CRE: {len(update_caries_pcd.points) / initial_caries}')
        print(f'MIP: {processed_cavity / initial_caries}')
        print(f'Traverse Length: {traverse_length}')
        print(f'Traverse Angle: {traverse_angle}')

        if visualize:
            window1.update_geometry(update_caries_pcd)
            window1.update_geometry(update_tooth_pcd)
            window1.update_geometry(burr)
            window1.poll_events()
            window1.update_renderer()
            ctr1.set_lookat(curr_pos)
            window11.update_geometry(update_caries_pcd)
            window11.update_geometry(update_tooth_pcd)
            window11.update_geometry(burr)
            window11.poll_events()
            window11.update_renderer()
            ctr11.set_lookat(curr_pos)
            window2.update_geometry(update_caries_pcd)
            window2.update_geometry(update_tooth_pcd)
            window2.update_geometry(burr)
            window2.poll_events()
            window2.update_renderer()
            ctr2.set_lookat(curr_pos)
            window22.update_geometry(update_caries_pcd)
            window22.update_geometry(update_tooth_pcd)
            window22.update_geometry(burr)
            window22.poll_events()
            window22.update_renderer()
            ctr22.set_lookat(curr_pos)
        else:
            o3d.visualization.draw_geometries([ref, bbox, update_caries_pcd, update_tooth_pcd, burr])

        # user keyboard input
        user_input = input("Keyboard input (n to stop): ")
        action = keyboard_to_action(user_input)

        # update bur mesh
        burr.rotate(burr.get_rotation_matrix_from_quaternion(curr_quat).transpose(), center=curr_pos)
        curr_pos += action[:3] * 0.1
        burr.translate(burr_center + curr_pos, relative=False)
        curr_quat = curr_quat * UnitQuaternion.RPY(action[3:], unit='deg')
        # traverse_angle += prev_quat.angdist(UnitQuaternion.AngVec(wp[3], wp[4:]))
        burr.rotate(burr.get_rotation_matrix_from_quaternion(curr_quat), center=curr_pos)

        # traverse
        cutpath.append(np.concatenate((curr_pos, (init_quat*curr_quat).A)))
        traverse_length += np.abs(action[:3]).sum() * 0.1
        traverse_angle += np.abs(action[3:]).sum() * 1

    np.savetxt(f'dental_env/demo_crop/tooth_{tnum}_{label}_human_cutpath.csv', cutpath)
    if visualize:
        window1.close()
        window11.close()
        window2.close()
        window22.close()


