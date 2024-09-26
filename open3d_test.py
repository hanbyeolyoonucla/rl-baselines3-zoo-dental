import dental_env
import open3d as o3d
import numpy as np


cary_mesh = o3d.io.read_triangle_mesh('dental_env/cad/cary.stl')
enamel_mesh = o3d.io.read_triangle_mesh('dental_env/cad/enamel.stl')
burr_mesh = o3d.io.read_triangle_mesh('dental_env/cad/burr.stl')

cary_voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(cary_mesh, voxel_size=0.1)
enamel_voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(enamel_mesh, voxel_size=0.1)
burr_voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(burr_mesh, voxel_size=0.1)

o3d.visualization.draw_geometries([cary_voxel, enamel_voxel, burr_voxel])

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
