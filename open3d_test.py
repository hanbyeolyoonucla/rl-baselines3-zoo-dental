import dental_env
import open3d as o3d

# print(torch.__version__)
# print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")

cary_mesh = o3d.io.read_triangle_mesh('dental_env/cad/cary.stl')
enamel_mesh = o3d.io.read_triangle_mesh('dental_env/cad/enamel.stl')
burr_mesh = o3d.io.read_triangle_mesh('dental_env/cad/burr.stl')

cary_voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(cary_mesh, voxel_size=0.1)
enamel_voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(enamel_mesh, voxel_size=0.1)
burr_voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(burr_mesh, voxel_size=0.1)

o3d.visualization.draw_geometries([cary_voxel, enamel_voxel, burr_voxel])
