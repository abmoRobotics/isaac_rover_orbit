from typing import Sequence

import torch
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.envs import BaseEnv
from omni.isaac.orbit.managers import CommandTerm
from omni.isaac.orbit.markers import VisualizationMarkers
# TODO (anton): Remove the following import since they were changed in the Orbit API
# from omni.isaac.orbit.envs.mdp.commands.commands_cfg import TerrainBasedPositionCommandCfg
# from omni.isaac.orbit.envs.mdp.commands.position_command import TerrainBasedPositionCommand
from omni.isaac.orbit.markers.config import CUBOID_MARKER_CFG
from omni.isaac.orbit.terrains import TerrainImporter, TerrainImporterCfg
from omni.isaac.orbit.utils.math import quat_rotate_inverse, wrap_to_pi, yaw_quat

from .terrain_utils import TerrainManager


# TODO: THIS IS A TEMPORARY FIX, since terrain based command is changed in the Orbit API
class TerrainBasedPositionCommand(CommandTerm):
    """Command generator that generates position commands based on the terrain.

    The position commands are sampled from the terrain mesh and the heading commands are either set
    to point towards the target or are sampled uniformly.
    """

    """Configuration for the command generator."""

    def __init__(self, cfg, env: BaseEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        # -- terrain
        self.terrain: TerrainImporter = env.scene.terrain

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "TerrainBasedPositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base position in base frame. Shape is (num_envs, 3)."""
        return self.pos_command_b

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets from the terrain
        # TODO: need to add that here directly
        self.pos_command_w[env_ids] = self.terrain.sample_new_targets(env_ids)
        # offset the position command by the current root position
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_heading = wrap_to_pi(target_direction + torch.pi)
            self.heading_command_w[env_ids] = torch.where(
                wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
                < wrap_to_pi(flipped_heading - self.robot.data.heading_w[env_ids]).abs(),
                target_direction,
                flipped_heading,
            )
        else:
            # random heading command
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root position and heading."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos"] = torch.norm(self.pos_command_w - self.robot.data.root_pos_w[:, :3], dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "box_goal_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_goal"
                marker_cfg.markers["cuboid"].scale = (0.1, 0.1, 0.1)
                self.box_goal_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.box_goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "box_goal_visualizer"):
                self.box_goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.box_goal_visualizer.visualize(self.pos_command_w)


class RoverTerrainImporter(TerrainImporter):
    def __init__(self, cfg: TerrainImporterCfg):
        super().__init__(cfg)
        self._cfg = cfg
        self._terrainManager = TerrainManager(num_envs=self._cfg.num_envs, device=self.device)
        self.target_distance = 9.0

    def sample_new_targets(self, env_ids):
        # We need to keep track of the original env_ids, because we need to resample some of them
        original_env_ids = env_ids

        # Initialize the target position
        target_position = torch.zeros(self._cfg.num_envs, 3, device=self.device)

        # Sample new targets
        reset_buf_len = len(env_ids)
        while (reset_buf_len > 0):
            # sample new random targets
            # print(reset_buf_len)
            # print(f'env_ids: {env_ids}')
            target_position[env_ids] = self.generate_random_targets(env_ids, target_position)

            # Here we check if the target is valid, and if not, we resample a new random target
            env_ids, reset_buf_len = self._terrainManager.check_if_target_is_valid(
                env_ids, target_position[env_ids, 0:2], device=self.device)

        # Adjust the height of the target, so that it matches the terrain
        target_position[original_env_ids, 2] = self._terrainManager._heightmap_manager.get_height_at(
            target_position[original_env_ids, 0:2])

        return target_position[original_env_ids]

    def generate_random_targets(self, env_ids, target_position):
        """
        This function generates random targets for the rover to navigate to.
        The targets are generated in a circle around the environment origin.

        Args:
            env_ids: The ids of the environments for which we need to generate targets.
            target_position: The target position buffer.
        """
        radius = self.target_distance
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


# class TerrainBasedPositionCommandCustom(TerrainBasedPositionCommand):

#     cfg: TerrainBasedPositionCommandCfg

#     def __init__(self, cfg: TerrainBasedPositionCommandCfg, env):
#         super().__init__(cfg, env)

#     def _set_debug_vis_impl(self, debug_vis: bool):
#         # create markers if necessary for the first tome
#         if debug_vis:
#             if not hasattr(self, "box_goal_visualizer"):
#                 marker_cfg = VisualizationMarkersCfg(
#                     prim_path="/Visuals/Command/position_goal",
#                     markers={
#                         "sphere": sim_utils.SphereCfg(
#                             radius=0.2,
#                             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
#                         ),
#                     },
#                 )
#                 self.box_goal_visualizer = VisualizationMarkers(marker_cfg)
#             # set their visibility to true
#             self.box_goal_visualizer.set_visibility(True)
#         else:
#             if hasattr(self, "box_goal_visualizer"):
#                 self.box_goal_visualizer.set_visibility(False)

#     def _debug_vis_callback(self, event):
#         self.box_goal_visualizer.visualize(translations=self.pos_command_w, marker_indices=[0])
