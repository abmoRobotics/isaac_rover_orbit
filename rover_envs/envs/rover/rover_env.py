import asyncio
import re
from typing import List

import omni
import torch
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.orbit.envs.rl_task_env import RLTaskEnv
from omni.isaac.orbit.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.orbit.terrains import TerrainImporter
from pxr import PhysxSchema, Sdf, Usd, UsdGeom

from .rover_env_cfg import RoverEnvCfg, RoverSceneCfg


def change_non_root_decorator(func):
    def wrapper(self, *args, **kwargs):
        omni.timeline.get_timeline_interface().pause()
        result = func(self, *args, **kwargs)
        omni.timeline.get_timeline_interface().play()
        return result
    return wrapper

class RoverEnv(RLTaskEnv):
    """ Rover environment.

    Note:
        This is a placeholder class for the rover environment. That is, this class is not yet implemented."""
    def __init__(self, cfg: RoverEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        env_ids = torch.arange(self.num_envs, device=self.device)


        # Get the terrain and change the origin
        terrain: TerrainImporter = self.scene.terrain
        terrain.env_origins[env_ids,0] += 100
        terrain.env_origins[env_ids,1] += 100

        # Change the contact sensor at each body to only report contact with the obstacle to improve performance:
        # if "contact_sensor" in self.scene.sensors:
        #     self.prepare_contact_sensors(cfg)

    #@change_non_root_decorator
    def prepare_contact_sensors(self, cfg: RoverEnvCfg):
        """
        Prepare the contact sensors.

        This function add contact report pairs to the contact sensor API for each body with a contact sensor.
        This is done to avoid reporting contact with the ground, and thus increase performce.

        Args:
            cfg (RoverEnvCfg): The rover environment config.
        """
        def find_contact_sensor_bodies(contact_sensor_cfg: ContactSensorCfg) -> List[str]:
            """
            Find the bodies that the contact sensor should report contact for.

            Args:
                contact_sensor_cfg (ContactSensorCfg): The contact sensor config.

            Returns:
                List[str]: A list of strings containing the names of the bodies that the contact sensor should report contact for.
            """
            # Define a pattern to search
            pattern = r"\(([^)]+)\)"
            # Using re.search() to find the match
            match: str = re.findall(pattern, contact_sensor_cfg.prim_path)[0]

            # Split the match into a list of strings
            return match.split("|")

        stage = get_current_stage()

        contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        contact_sensor_cfg = cfg.scene.contact_sensor

        target_prim = contact_sensor_cfg.filter_prim_paths_expr[0]
        robot_prims = [stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot") for i in range(self.num_envs)]

        for robot_prim in robot_prims:
            prim = robot_prim#stage.GetPrimAtPath(robot_prim)

            link: Usd.Prim
            for link in prim.GetChildren():
                contact_reporter_body_keywords = find_contact_sensor_bodies(contact_sensor_cfg)

                # Check if the child prim is one of the bodies we want to report contact for
                if any(keyword in link.GetName() for keyword in contact_reporter_body_keywords):

                    # Get the contact reporter API
                    contact_report_api: PhysxSchema._physxSchema.PhysxContactReportAPI = PhysxSchema._physxSchema.PhysxContactReportAPI.Get(stage, link.GetPrimPath())
                    contact_report_api.CreateReportPairsRel().AddTarget(target_prim)
