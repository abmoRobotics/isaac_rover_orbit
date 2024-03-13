import re

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.orbit.assets import Articulation
from pxr import PhysxSchema, Sdf, Usd, UsdGeom


class RoverArticulation(Articulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_contact_sensors()

    def prepare_contact_sensors(self):
        stage = get_current_stage()
        pattern = "/World/envs/env_.*/Robot/.*(Drive|Steer|Boogie|Bogie|Body)$"
        matching_prims = []
        prim: Usd.Prim
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Xform):
                prim_path: Sdf.Path = prim.GetPath()
                if re.match(pattern, prim_path.pathString):
                    matching_prims.append(prim_path)

        for prim in matching_prims:
            contact_api: PhysxSchema._physxSchema.PhysxContactReportAPI = \
                PhysxSchema._physxSchema.PhysxContactReportAPI.Get(stage, prim)
            contact_api.CreateReportPairsRel().AddTarget("/World/terrain/obstacles/obstacles")
