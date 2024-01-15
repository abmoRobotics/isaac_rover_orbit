import torch
from omni.isaac.orbit.terrains import TerrainImporter, TerrainImporterCfg

from .terrain_utils import TerrainManager


class RoverTerrainImporter(TerrainImporter):
    def __init__(self, cfg: TerrainImporterCfg):
        super().__init__(cfg)
        self._cfg = cfg
        self._terrainManager = TerrainManager(num_envs=self._cfg.num_envs, device=self.device)

    def sample_new_targets(self, env_ids):
        # We need to keep track of the original env_ids, because we need to resample some of them
        original_env_ids = env_ids

        # Initialize the target position
        target_position = torch.zeros(self._cfg.num_envs, 3, device=self.device)

        # Sample new targets
        reset_buf_len = len(env_ids)
        while (reset_buf_len > 0):
            # sample new random targets
            target_position[env_ids] = self.generate_random_targets(env_ids, target_position)

            # Here we check if the target is valid, and if not, we resample a new random target
            env_ids, reset_buf_len = self._terrainManager.check_if_target_is_valid(env_ids, target_position[env_ids, 0:2], device=self.device)

        # Adjust the height of the target, so that it matches the terrain
        target_position[original_env_ids, 2] = self._terrainManager._heightmap_manager.get_height_at(target_position[original_env_ids, 0:2])

        return target_position[original_env_ids]

    def generate_random_targets(self, env_ids, target_position):
        """
        This function generates random targets for the rover to navigate to.
        The targets are generated in a circle around the environment origin.

        Args:
            env_ids: The ids of the environments for which we need to generate targets.
            target_position: The target position buffer.
        """
        radius = 9
        theta = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi

        # set the target x and y positions
        target_position[env_ids, 0] = torch.cos(theta) * radius + self.env_origins[env_ids, 0]
        target_position[env_ids, 1] = torch.sin(theta) * radius + self.env_origins[env_ids, 1]

        return target_position[env_ids]

    def get_spawn_locations(self):
        """
        This function returns valid spawn locations, that avoids spawning the rover on top of obstacles.

        Returns:
            spawn_locations: The spawn locations buffer. Shape (num_envs, 3).
        """
        return self._terrainManager.spawn_locations
