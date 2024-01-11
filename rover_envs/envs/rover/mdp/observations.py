from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import RayCaster
#from omni.isaac.orbit.command_generators import UniformPoseCommandGenerator
from omni.isaac.orbit.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def angle_to_target_observation(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Calculate the angle to the target."""
    rover_asset: RigidObject = env.scene[asset_cfg.name]
    _, _, yaw = euler_xyz_from_quat(rover_asset.data.root_state_w[:, 3:7])
    rover_position = rover_asset.data.root_state_w[:, :3]

    # Calculating the rover's heading direction
    direction_vector = torch.zeros([env.num_envs, 2], device=env.device)
    direction_vector[:, 0] = torch.cos(yaw)
    direction_vector[:, 1] = torch.sin(yaw)

    # Accessing the target's position and calculating the direction vector
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]
    target_vector = target_position[:,:2] - rover_position[:,:2]

    # Calculating the angle and applying the penalty
    cross_product = direction_vector[:, 1] * target_vector[:, 0] - direction_vector[:, 0] * target_vector[:, 1]
    dot_product = direction_vector[:, 0] * target_vector[:, 0] + direction_vector[:, 1] * target_vector[:, 1]
    angle = torch.atan2(cross_product, dot_product)
    #print(env.scene.terrain)
    return angle.unsqueeze(-1)

def distance_to_target_euclidean(env: RLTaskEnv, command_name: str):
    """Calculate the euclidean distance to the target."""
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]
    distance: torch.Tensor = torch.norm(target_position, p=2, dim=-1)
    return distance.unsqueeze(-1)

def height_scan_rover(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Calculate the height scan of the rover.

    This function uses a ray caster to generate a height scan of the rover's surroundings.
    The height scan is normalized by the maximum range of the ray caster.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - 0.26878 (0.26878 is the distance between the sensor and the rover's base)
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.26878
