import torch
from omni.isaac.orbit.envs.rl_task_env import RLTaskEnv

from .rover_env_cfg import RoverEnvCfg
from .utils.terrain_utils.terrain_utils import TerrainManager


class RoverEnv(RLTaskEnv):
    def __init__(self, cfg: RoverEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._terrainManager = TerrainManager(self.device)
        #self._reset_rover_state()


    # def _reset_rover_state(self, env_ids):
    #     reset_state = self.scene.robot.get_default_root_state(env_ids=env_ids)

    #     spawn_locations = self._terrainManager.spawn_locations

    #     spawn_index = torch.randperm(len(spawn_locations), device=self.device)[:len(env_ids)]

    #     # Set Position
    #     self.envs
