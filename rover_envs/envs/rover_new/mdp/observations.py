from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
#from omni.isaac.orbit.command_generators import UniformPoseCommandGenerator
from omni.isaac.orbit.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def angle_to_target_observation(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    rover_asset: RigidObject = env.scene[asset_cfg.name]
    _, _, yaw = euler_xyz_from_quat(rover_asset.data.root_state_w[:, 3:7])
    rover_position = rover_asset.data.root_state_w[:, :3]
    # print(f'rover_position: {rover_position}')
    # print(f'rover_rotation_euler: {rover_rotation_euler}')
    # print(f'rover_asset.data.root_state_w: {rover_asset.data.root_state_w}')
    # print(f'rover_asset.data.root_state_w[:, :3]: {rover_asset.data.root_state_w[:, :3]}')
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

def distance_to_target_euclidean(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg):
    rover_asset: RigidObject = env.scene[asset_cfg.name]
    rover_position = rover_asset.data.root_state_w[:, :3]
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :3]
    distance: torch.Tensor = torch.norm(target_position - rover_position, p=2, dim=-1)
    return distance.unsqueeze(-1)
