from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.command_generators import UniformPoseCommandGenerator
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

def distance_to_target(env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Calculate and return the distance to the target.

    Args:
        env: RLTaskEnv object representing the environment.
        asset_cfg: SceneEntityCfg object for the target asset configuration.

    Returns:
        torch.Tensor: Tensor representing the distance to the target.
    """
    # Get the rover asset and position
    rover_asset: RigidObject = env.scene["robot"]
    rover_position = rover_asset.data.root_state_w[:, 3]

    # Get the target manager and position
    target_manager: UniformPoseCommandGenerator = env.command_manager
    target_position = target_manager.command[:, 3]

    # Compute the distance
    distance = torch.norm(target_position - rover_position, p=2, dim=-1)

    # Calculate reward
    reward = (1.0 / (1.0 + (0.11 * distance * distance))) / env.max_episode_length

    return reward

def reached_target(env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns whether the target has been reached.

    Args:
        env: The environment.

    Returns:
        Whether the target has been reached.
    """
    # Get the rover asset and position
    rover_asset: RigidObject = env.scene[asset_cfg.name]
    rover_position = rover_asset.data.root_state_w[:, 3]

    # Get the target manager and position
    target_manager: UniformPoseCommandGenerator = env.command_manager
    target_position = target_manager.command[:, 3]

    # Compute the distance
    distance = torch.norm(target_position - rover_position, p=2, dim=-1)

    # Calcuate how many time steps are left in the episode
    time_steps_to_goal = env.max_episode_length - env.episode_length_buf

    # Scale to [0, 1]
    reward_scale = time_steps_to_goal / env.max_episode_length

    reward = torch.where(distance < 0.18, 1.0 * reward_scale, 0.0)

    return reward

def oscillation_penalty(env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the oscillation penalty.

    Args:
        env: The environment.

    Returns:
        The oscillation penalty.
    """
    # Get the rover asset and position
    rover_asset: RigidObject = env.scene[asset_cfg.name]

    # Get the actions
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    # Compute the oscillation penalty
    linear_diff = action[:, 1] - prev_action[:, 1]
    angular_diff = action[:, 0] - prev_action[:, 0]

    # TODO combine these 5 lines into two lines
    angular_penalty = torch.where(angular_diff > 0.05, torch.square(angular_diff), 0.0)
    linear_penalty = torch.where(linear_diff > 0.05, torch.square(linear_diff), 0.0)

    angular_penalty = torch.pow(angular_penalty, 2)
    linear_penalty = torch.pow(linear_penalty, 2)

    return (angular_penalty + linear_penalty) / env.max_episode_length

def goal_angle_penalty(env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """ Calculates the penalty for the angle between the rover and the target."""

    # Get the rover asset
    rover_asset: RigidObject = env.scene[asset_cfg.name]
    # Get the rover rotation
    rover_rotation_euler = euler_xyz_from_quat(rover_asset.data.root_state_w[:, :4])
    rover_position = rover_asset.data.root_state_w[:, 3]

    # Calculate the direction vector
    direction_vector = torch.zeros([env.num_envs, 2])
    direction_vector[:, 0] = torch.cos(rover_rotation_euler[:, 2])
    direction_vector[:, 1] = torch.sin(rover_rotation_euler[:, 2])

    # Get the target manager and position
    target_manager: UniformPoseCommandGenerator = env.command_manager
    target_position = target_manager.command[:, 3]
    target_vector = target_position - rover_position

    # Calculate the angle between the heading vector and the target vector
    cross_product = direction_vector[:, 1] * target_vector[:, 0] - direction_vector[:, 0] * target_vector[:, 1]
    dot_product = direction_vector[:, 0] * target_vector[:, 0] + direction_vector[:, 1] * target_vector[:, 1]
    angle = torch.atan2(cross_product, dot_product)

    # Calculate the penalty
    return torch.where(torch.abs(angle) > 2.0, torch.abs(angle) / env.max_episode_length, 0.0)


def heading_soft_contraint(env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """ Gives a penalty for driving backwards.
    Args:
        env: The environment.
        asset_cfg: The target asset configuration.

    Returns:
        The heading penalty.
    """
    return torch.where(env.action_manager.action[:, 0] < 0.0, (1.0 / env.max_episode_length), 0.0)

def collision_penalty(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:


    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix = contact_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)


    normalized_forces = torch.norm(force_matrix, dim=1)
    forces_active = torch.sum(normalized_forces, dim=1) > 1.0
    return torch.where(forces_active, 1.0, 0.0)
