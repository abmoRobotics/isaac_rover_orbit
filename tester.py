import pymeshlab
from pxr import Usd

stage: Usd.Stage = Usd.Stage.Open("/home/anton/1._University/0._Master_Project/Workspace/terrain_generation/terrains/mars1/terrain_merged.usd")
mesh_prim = stage.GetPrimAtPath("/BIG_1_001_423_generated/Mesh_766")

print("ok")
points = mesh_prim.GetAttribute("points").Get()
face_vertex_counts = mesh_prim.GetAttribute("faceVertexCounts").Get()
face_vertex_indices = mesh_prim.GetAttribute("faceVertexIndices").Get()
print(type(points))

vertices = [(point[0], point[1], point[2]) for point in points]
faces = [(face_vertex_indices[i], face_vertex_indices[i+1], face_vertex_indices[i+2]) for i in range(0, len(face_vertex_indices), 3)]

mesh = pymeshlab.Mesh(vertices, faces)
mesh_set = pymeshlab.MeshSet()
mesh_set.add_mesh(mesh)
mesh_set.save_current_mesh("/home/anton/1._University/0._Master_Project/Workspace/terrain_generation/terrains/mars1/rocks_merged123456.obj")
#ms.Mesh(vertices, faces)
print("ok2")
