import numpy as np
import open3d as o3d
import copy
from scipy.ndimage import affine_transform, zoom
from spatialmath import SO3, SE3, UnitQuaternion
import os
from tqdm import tqdm
import pandas as pd


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


if __name__ == "__main__":

    # np load npy
    use_rotation = True
    visualize = False
    res = 0.102
    thres = 0.0
    max_length = 50
    tnum, voxel_size = 5, 60
    tooth_dir = f'dental_env/labels_augmented/'
    dirlist = os.listdir(tooth_dir)
    fname = dirlist[np.random.randint(0, len(dirlist))]
    stats = pd.DataFrame(columns=['tooth', 'IC', 'RC', 'PC', 'CRE', 'MIP', 'Traverse Length', 'Traverse Angle'])
    for fname in tqdm(dirlist):
        tooth = np.load(tooth_dir+fname)

        # argwhere and pcd + 1/2
        caries_points = ((np.argwhere(tooth == 1) + 1 / 2) * res).astype(np.float32)
        enamel_points = ((np.argwhere(tooth == 2) + 1 / 2) * res).astype(np.float32)
        dentin_points = ((np.argwhere(tooth == 3) + 1 / 2) * res).astype(np.float32)
        tooth_points = np.concatenate((enamel_points, dentin_points)).astype(np.float32)

        # convert to point clouds
        caries_pcd = o3d.geometry.PointCloud()
        tooth_pcd = o3d.geometry.PointCloud()
        caries_pcd.points = o3d.utility.Vector3dVector(caries_points)
        tooth_pcd.points = o3d.utility.Vector3dVector(tooth_points)

        # check visualization (crop vs original)
        ref = o3d.geometry.TriangleMesh.create_coordinate_frame()
        caries_pcd.paint_uniform_color([0.3, 0.3, 0.3])
        tooth_pcd.paint_uniform_color([0.95, 0.95, 0.99])
        bbox = bounding_box(tooth, res)
        # o3d.visualization.draw_geometries([ref, bbox, caries_pcd, tooth_pcd])

        #############################################################
        # traction path planning
        #############################################################
        # bur initialize - translate/rotate to top/left/right center
        burr = o3d.io.read_triangle_mesh(f'dental_env/cad/burr.stl')
        burr.compute_vertex_normals()
        if 'top' in fname:
            init_pos = np.array([voxel_size/2, voxel_size/2, voxel_size]) * res
            init_quat = UnitQuaternion()
        elif 'left' in fname:
            init_pos = np.array([voxel_size/2, voxel_size, voxel_size/2]) * res
            init_quat = UnitQuaternion(SO3.RPY(-90, 0, 0, unit='deg'))
        else:
            init_pos = np.array([voxel_size/2, 0, voxel_size/2]) * res
            init_quat = UnitQuaternion(SO3.RPY(90, 0, 0, unit='deg'))

        burr_center = burr.get_center()
        burr.translate(init_pos + burr_center, relative=False)
        burr.rotate(burr.get_rotation_matrix_from_quaternion(init_quat), center=init_pos)
        # o3d.visualization.draw_geometries([ref, bbox, caries_pcd, tooth_pcd, burr])

        # initial pos, quat, caries, tooth
        curr_pos = init_pos
        curr_quat = init_quat
        cutpath = [np.concatenate((curr_pos, curr_quat.A))]
        initial_caries = len(caries_pcd.points)
        initial_tooth = len(tooth_pcd.points)
        traverse_length = 0
        traverse_angle = 0

        # initialize visualization
        points = np.array([curr_pos, curr_pos, curr_pos, curr_pos])
        lines = np.array([[0, 1], [0, 2], [0, 3]])
        colors = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        actions = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        actions.colors = o3d.utility.Vector3dVector(colors)

        if visualize:
            window = o3d.visualization.Visualizer()
            window.create_window(window_name='Cut Path Episode', width=1080, height=1080,
                                 left=50, top=50, visible=True)
            window.add_geometry(ref)
            window.add_geometry(bbox)
            window.add_geometry(caries_pcd)
            window.add_geometry(tooth_pcd)
            window.add_geometry(burr)
            window.add_geometry(actions)
            window.get_render_option().line_width = 100
            ctr = window.get_view_control()
            ctr.rotate(0, -200)

        while len(caries_pcd.points)/initial_caries > thres and traverse_length <= max_length:
            # calculate disgned distance vector and update pcd
            burr_geom = o3d.t.geometry.TriangleMesh.from_legacy(burr)
            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(burr_geom)  # we do not need the geometry ID for mesh

            # occupancy
            removal_caries = scene.compute_occupancy(caries_points).numpy().astype(bool)
            removal_tooth = scene.compute_occupancy(tooth_points).numpy().astype(bool)
            caries_points = caries_points[~removal_caries]
            tooth_points = tooth_points[~removal_tooth]
            caries_pcd.points = o3d.utility.Vector3dVector(caries_points)
            tooth_pcd.points = o3d.utility.Vector3dVector(tooth_points)
            caries_pcd.paint_uniform_color([0.3, 0.3, 0.3])
            tooth_pcd.paint_uniform_color([0.95, 0.95, 0.99])

            # mask for tooth pcd near bur
            # bound_r = 1.5
            # tooth_pcd_np = tooth_pcd_np[np.linalg.norm(tooth_pcd_np - curr_pos, axis=1) <= bound_r]

            # signed distance vector
            eps = 1e-5
            bur_caries = scene.compute_closest_points(caries_points)['points'].numpy()
            bur_tooth = scene.compute_closest_points(tooth_points)['points'].numpy()
            caries_moment_arm = bur_caries - curr_pos
            tooth_moment_arm = bur_tooth - curr_pos
            caries_to_bur = bur_caries - caries_points
            tooth_to_bur = bur_tooth - tooth_points

            bound = 0.5
            tooth_mask = np.linalg.norm(tooth_to_bur, axis=1) <= bound
            tooth_to_bur = tooth_to_bur[tooth_mask]
            tooth_moment_arm = tooth_moment_arm[tooth_mask]
            caries_mask = np.linalg.norm(caries_to_bur, axis=1) <= bound

            mc = np.cross(caries_moment_arm[caries_mask], -caries_to_bur[caries_mask]).mean(axis=0) if caries_to_bur[caries_mask].any() else np.zeros(3)
            mt = np.cross(tooth_moment_arm, tooth_to_bur).mean(axis=0) if tooth_to_bur.any() else np.zeros(3)
            min_dist = np.linalg.norm(caries_to_bur, axis=1).min() if caries_to_bur.any() else np.zeros(3)
            caries_to_bur = (caries_to_bur / ((caries_to_bur ** 2).sum(axis=1, keepdims=True) +eps)).mean(axis=0) if caries_to_bur.any() else np.zeros(3)
            tooth_to_bur = (tooth_to_bur / ((tooth_to_bur ** 2).sum(axis=1, keepdims=True) +eps)).mean(axis=0) if tooth_to_bur.any() else np.zeros(3)

            # visualize
            weight = 5
            action = -weight * caries_to_bur + tooth_to_bur
            action = action / np.linalg.norm(action) * 0.1  # max(min(min_dist, 0.1), 0.01)
            traverse_length += 0.1
            rotation = mc + mt
            rotation = rotation / np.linalg.norm(rotation) if rotation.any() else np.zeros(3)
            points = np.array([curr_pos, curr_pos - caries_to_bur, curr_pos + tooth_to_bur, curr_pos + action*10])
            actions.points = o3d.utility.Vector3dVector(points)
            # actions.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([ref, bbox, caries_pcd, tooth_pcd, burr, actions])

            # translate bur and update distance vector
            if visualize:
                window.update_geometry(caries_pcd)
                window.update_geometry(tooth_pcd)
                window.update_geometry(burr)
                window.update_geometry(actions)
                window.poll_events()
                window.update_renderer()
            # print(f'curr_pos: {curr_pos}')
            # print(f'action: {action}')
            # print(f'rotation: {rotation}')
            # print(f'removal_action: {-caries_to_bur}')
            # print(f'avoidance_action: {tooth_to_bur}')

            # update bur mesh
            burr.rotate(burr.get_rotation_matrix_from_quaternion(curr_quat).transpose(), center=curr_pos)
            curr_pos += action
            burr.translate(curr_pos + burr_center, relative=False)
            if use_rotation:
                traverse_angle += 0.1 if rotation.any() else 0
                curr_quat = UnitQuaternion.AngVec(0.1, rotation, unit='deg')*curr_quat if rotation.any() else curr_quat
                burr.rotate(burr.get_rotation_matrix_from_quaternion(curr_quat), center=curr_pos)
            cutpath.append(np.concatenate((curr_pos, curr_quat.A)))

            # performance metric
            processed_cavity = initial_caries + initial_tooth - len(caries_pcd.points) - len(tooth_pcd.points)

        np.savetxt(f'dental_env/demos_augmented/traction/{fname[:-4]}.csv', cutpath)
        stat = {
            'tooth': fname[:-4],
            'IC': initial_caries*(res**3),
            'RC': len(caries_pcd.points)*(res**3),
            'PC': processed_cavity*(res**3),
            'CRE': len(caries_pcd.points)/initial_caries,
            'MIP': processed_cavity/initial_caries,
            'Traverse Length': traverse_length,
            'Traverse Angle': traverse_angle
        }
        stat = pd.DataFrame([stat])
        # stat[['tooth']] = fname[:-4]
        # stat[['IC']] = initial_caries*(res**3)
        # stat[['RC']] = len(caries_pcd.points)*(res**3)
        # stat[['PC']] = processed_cavity*(res**3)
        # stat[['CRE']] = len(caries_pcd.points)/initial_caries
        # stat[['MIP']] = processed_cavity/initial_caries
        # stat[['Traverse Length']] = traverse_length
        # stat[['Traverse Angle']] = traverse_angle
        stats = pd.concat([stats, stat], ignore_index=True)
        stats.to_csv(f'dental_env/demos_augmented/traction/00_stats.csv', index=False)
        print(stat)

        if visualize:
            window.close()
