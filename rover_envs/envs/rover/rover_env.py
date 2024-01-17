
import torch
from omni.isaac.orbit.envs.rl_task_env import RLTaskEnv
from omni.isaac.orbit.terrains import TerrainImporter

from .rover_env_cfg import RoverEnvCfg


class RoverEnv(RLTaskEnv):
    """ Rover environment.

    Note:
        This is a placeholder class for the rover environment. That is, this class is not yet implemented."""

    def __init__(self, cfg: RoverEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        env_ids = torch.arange(self.num_envs, device=self.device)

        # Get the terrain and change the origin
        terrain: TerrainImporter = self.scene.terrain
        terrain.env_origins[env_ids, 0] += 100
        terrain.env_origins[env_ids, 1] += 100
