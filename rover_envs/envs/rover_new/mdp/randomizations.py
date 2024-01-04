
from typing import TYPE_CHECKING

import torch
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.envs import BaseEnv
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

    from ..rover_env import RoverEnv



def reset_root_state_rover(
        env: BaseEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg):

    # Get the rover asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the default root state
    reset_state = asset.data.default_root_state[env_ids].clone()


    spawn_locations = env._terrainManager.spawn_locations
    spawn_index = torch.randperm(len(spawn_locations), device=env.device)[:len(env_ids)]
    spawn_locations = spawn_locations[spawn_index]

    positions = spawn_locations[:, :3]
    positions[:, 2] += 1.0

    # Random angle
    angle = torch.rand(len(env_ids), device=env.device) * 2 * torch.pi
    quat = torch.zeros(len(env_ids), 4, device=env.device)
    quat[:, 0] = torch.cos(angle / 2)
    quat[:, 3] = torch.sin(angle / 2)
    orientations = quat

    # Set the root state
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
