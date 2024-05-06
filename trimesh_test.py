import dental_env
import trimesh

# print(torch.__version__)
# print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")


burr = trimesh.load('dental_env/cad/burr.stl')
cary = trimesh.load('dental_env/cad/cary.stl')
enamel = trimesh.load('dental_env/cad/enamel.stl')

# scene = trimesh.Scene()
# scene.add_geometry(burr)
# scene.add_geometry(cary)
# scene.add_geometry(enamel)
# scene.add_geometry(trimesh.creation.axis())
# scene.show()


burr_voxel = trimesh.voxel.creation.voxelize(burr, pitch=0.1)
cary_voxel = trimesh.voxel.creation.voxelize(cary, pitch=0.1)
enamel_voxel = trimesh.voxel.creation.voxelize(enamel, pitch=0.1)

scene = trimesh.Scene()
scene.add_geometry(burr_voxel.marching_cubes)
scene.add_geometry(cary_voxel.marching_cubes)
scene.add_geometry(enamel_voxel.marching_cubes)
scene.add_geometry(trimesh.creation.axis())
scene.show()