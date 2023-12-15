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

def distance_to_target(env: RLTaskEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """
    Calculate and return the distance to the target.

    This function computes the Euclidean distance between the rover and the target.
    It then calculates a reward based on this distance, which is inversely proportional
    to the squared distance. The reward is also normalized by the maximum episode length.
    """
    # Accessing the rover object and its position
    rover_asset: RigidObject = env.scene[asset_cfg.name]
    rover_position = rover_asset.data.root_state_w[:, 3]

    # Accessing the target's position through the command manage
    target = env.command_manager.get_command(command_name)
    target_position = target[:, 3]

    # Calculating the distance and the reward
    distance = torch.norm(target_position - rover_position, p=2, dim=-1)
    reward = (1.0 / (1.0 + (0.11 * distance * distance))) / env.max_episode_length

    return reward

def reached_target(env: RLTaskEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float) -> torch.Tensor:
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
    time_steps_to_goal = env.max_episode_length - env.episode_length_buf
    reward_scale = time_steps_to_goal / env.max_episode_length

    reward = torch.where(distance < threshold, 1.0 * reward_scale, 0)

    return reward

def oscillation_penalty(env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate the oscillation penalty.

    This function penalizes the rover for oscillatory movements by comparing the difference
    in consecutive actions. If the difference exceeds a threshold, a squared penalty is applied.
    """
    # Accessing the rover's actions
    rover_asset: RigidObject = env.scene[asset_cfg.name]
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    # Calculating differences between consecutive actions
    linear_diff = action[:, 1] - prev_action[:, 1]
    angular_diff = action[:, 0] - prev_action[:, 0]

    # TODO combine these 5 lines into two lines
    angular_penalty = torch.where(angular_diff > 0.05, torch.square(angular_diff), 0.0)
    linear_penalty = torch.where(linear_diff > 0.05, torch.square(linear_diff), 0.0)

    angular_penalty = torch.pow(angular_penalty, 2)
    linear_penalty = torch.pow(linear_penalty, 2)

    return (angular_penalty + linear_penalty) / env.max_episode_length

def angle_to_target_penalty(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate the penalty for the angle between the rover and the target.

    This function computes the angle between the rover's heading direction and the direction
    towards the target. A penalty is applied if this angle exceeds a certain threshold.
    """
    # Accessing rover's position and rotation
    rover_asset: RigidObject = env.scene[asset_cfg.name]
    _,_, yaw = euler_xyz_from_quat(rover_asset.data.root_state_w[:, :4])
    rover_position = rover_asset.data.root_state_w[:, :2]

    # Calculating the rover's heading direction
    direction_vector = torch.zeros([env.num_envs, 2], device=env.device)
    direction_vector[:, 0] = torch.cos(yaw)
    direction_vector[:, 1] = torch.sin(yaw)

    # Accessing the target's position and calculating the direction vector
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]
    target_vector = target_position - rover_position

    # Calculating the angle and applying the penalty
    cross_product = direction_vector[:, 1] * target_vector[:, 0] - direction_vector[:, 0] * target_vector[:, 1]
    dot_product = direction_vector[:, 0] * target_vector[:, 0] + direction_vector[:, 1] * target_vector[:, 1]
    angle = torch.atan2(cross_product, dot_product)

    return torch.where(torch.abs(angle) > 2.0, torch.abs(angle) / env.max_episode_length, 0.0)

def heading_soft_contraint(env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate a penalty for driving backwards.

    This function applies a penalty when the rover's action indicates reverse movement.
    The penalty is normalized by the maximum episode length.
    """
    return torch.where(env.action_manager.action[:, 0] < 0.0, (1.0 / env.max_episode_length), 0.0)

def collision_penalty(env: RLTaskEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """
    Calculate a penalty for collisions detected by the sensor.

    This function checks for forces registered by the rover's contact sensor.
    If the total force exceeds a certain threshold, it indicates a collision,
    and a penalty is applied.
    """
    # Accessing the contact sensor and its data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    #print(contact_sensor)
    force_matrix = contact_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)
    #force_matrix = contact_sensor.data.force_matrix_w.view(-1, env.num_envs, 3)
    # print(f'force_matrix: {force_matrix.shape}')
    # print(f'force_matrix: {contact_sensor.data.force_matrix_w.shape}')
    # print(f'force_matrix: {contact_sensor.data.force_matrix_w}')
    # normalized_forces = torch.norm(self.collisions_view.get_contact_force_matrix().view(self.num_envs, -1, 3), dim=1)
    # forces_active = torch.sum(normalized_forces, dim=-1) > 1
    # self.reset_buf = torch.where(forces_active, 1, self.reset_buf)
    # Calculating the force and applying a penalty if collision forces are detected
    normalized_forces = torch.norm(force_matrix, dim=1)
    #print(f'normalized_forces: {normalized_forces}')
    forces_active = torch.sum(normalized_forces, dim=-1) > 1
    return torch.where(forces_active, True, False)
    #Return false tensor
    #return torch.tensor([False for i in range(env.num_envs)], device=env.device)
    # return torch.any(
    #     torch.max(torch.norm(contact_sensor.data.force_matrix_w[:, :,:], dim=-1), dim=1)[0] > threshold, dim=1
    # )
