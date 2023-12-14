from __future__ import annotations

from typing import TYPE_CHECKING

import torch
# Importing necessary modules from the omni.isaac.orbit package
from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

def is_success(env: RLTaskEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float) -> torch.Tensor:
    """
    Determine whether the target has been reached.

    This function checks if the rover is within a certain threshold distance from the target.
    If the target is reached, a scaled reward is returned based on the remaining time steps.
    """
    # Accessing the rover's position
    rover_asset: RigidObject = env.scene[asset_cfg.name]
    rover_position = rover_asset.data.root_state_w[:, 3]

    # Accessing the target's position
    target = env.command_manager.get_command(command_name)
    target_position = target[:, 3]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position - rover_position, p=2, dim=-1)

    return torch.where(distance < threshold, 1, 0)


def far_from_target(env: RLTaskEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float) -> torch.Tensor:
    """
    Determine whether the target has been reached.

    This function checks if the rover is within a certain threshold distance from the target.
    If the target is reached, a scaled reward is returned based on the remaining time steps.
    """
    # Accessing the rover's position
    rover_asset: RigidObject = env.scene[asset_cfg.name]
    rover_position = rover_asset.data.root_state_w[:, 3]

    # Accessing the target's position
    target = env.command_manager.get_command(command_name)
    target_position = target[:, 3]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position - rover_position, p=2, dim=-1)

    return torch.where(distance > threshold, 1, 0)

# def collision_with_rock(env: RLTaskEnv, asset_cfg: SceneEntityCfg, sensor_name: str) -> torch.Tensor:
#     """
#     Determine whether the target has been reached.

#     This function checks if the rover is within a certain threshold distance from the target.
#     If the target is reached, a scaled reward is returned based on the remaining time steps.
#     """
#     # Accessing the rover's position
#     rover_asset: RigidObject = env.scene[asset_cfg.name]
#     rover_position = rover_asset.data.root_state_w[:, 3]

#     # Accessing the target's position
#     sensor = env.sensor_manager.get_sensor(sensor_name)
#     sensor_data = sensor.data
#     sensor_data = sensor_data.reshape(sensor_data.shape[0], -1, 3)
#     sensor_data = sensor_data[:, :, 2]
#     sensor_data = sensor_data.reshape(sensor_data.shape[0], -1)
#     sensor_data = torch.sum(sensor_data, dim=1)

#     return torch.where(sensor_data > 0, 1, 0)
