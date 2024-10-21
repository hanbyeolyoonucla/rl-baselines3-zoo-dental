import dental_env
import open3d as o3d
import numpy as np
import trimesh


def np_to_burr_voxels(burr, voxel_grid):
    # voxel_grid = o3d.geometry.VoxelGrid()
    # voxel_grid.clear()
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


def crop_center(voxel, cropx, cropy, cropz):
    # local voxelize function can voxelize burr into cube, so we need to crop it for smaller dimension
    x, y, z = voxel.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    startz = z // 2 - (cropz // 2)
    return voxel[startx:startx + cropx, starty:starty + cropy, startz:startz + cropz]

cary_mesh = o3d.io.read_triangle_mesh('dental_env/cad/cary.stl')
enamel_mesh = o3d.io.read_triangle_mesh('dental_env/cad/enamel.stl')
burr_mesh = o3d.io.read_triangle_mesh('dental_env/cad/burr.stl')
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

cary_voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(cary_mesh, voxel_size=0.1)
enamel_voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(enamel_mesh, voxel_size=0.1)
burr_voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(burr_mesh, voxel_size=0.1)

o3d.visualization.draw_geometries([cary_voxel, enamel_voxel, burr_voxel])

o3d.visualization.draw_geometries([frame, burr_mesh])
print(cary_mesh.get_center())
agent_rotation = np.array([1, 1, 0, 0], dtype=np.float64)
burr_mesh.translate(cary_mesh.get_center(), relative=False)
# burr_mesh.rotate(burr_mesh.get_rotation_matrix_from_quaternion(agent_rotation), [0, 0, 0])
o3d.visualization.draw_geometries([frame, burr_mesh])
burr_mesh.rotate(burr_mesh.get_rotation_matrix_from_quaternion(agent_rotation).transpose())
agent_rotation = np.array([1, 0, 0, 1], dtype=np.float64)
burr_mesh.rotate(burr_mesh.get_rotation_matrix_from_quaternion(agent_rotation))
o3d.visualization.draw_geometries([frame, burr_mesh])



# pcd test
points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
aabb = pcd.get_axis_aligned_bounding_box()
aabb.color = (1,0,0)

o3d.visualization.draw_geometries([pcd, aabb])


voxel = o3d.geometry.VoxelGrid()
voxel.clear()



