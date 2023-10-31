from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from pxr import PhysxSchema
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.prims import XFormPrim

import numpy as np

def trimesh_to_usd(vertices: np.ndarray, faces: np.ndarray, position = None, orientation = None):
    """ Convert trimesh to USD 
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertices of the mesh
    faces : np.ndarray
        Faces of the mesh
    """
    
    # Get current stage 
    stage: Usd.Stage = get_current_stage()  

    # Define terrain mesh
    mesh_prim = stage.DefinePrim("/World/terrain", "Mesh")
    mesh_prim.GetAttribute("points").Set(vertices)
    mesh_prim.GetAttribute("faceVertexIndices").Set(faces.flatten())
    mesh_prim.GetAttribute("faceVertexCounts").Set(np.asarray([3] * faces.shape[0])) # 3 vertices per face 

    terrain_prim = XFormPrim(
                prim_path="/World/terrain",
                name="terrain",
                position=position,
                orientation=orientation)
    
    UsdPhysics.CollisionAPI.Apply(terrain_prim.prim)
    
    physx_collision_api: PhysxSchema._physxSchema.PhysxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(terrain_prim.prim)
    physx_collision_api.GetContactOffsetAttr().Set(0.04)
    physx_collision_api.GetRestOffsetAttr().Set(0.02)