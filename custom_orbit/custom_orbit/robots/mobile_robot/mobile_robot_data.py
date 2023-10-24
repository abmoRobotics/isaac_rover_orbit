import torch
from dataclasses import dataclass

from omni.isaac.orbit.robots.robot_base_data import RobotBaseData


@dataclass
class MobileRobotData(RobotBaseData):
    
    """States"""

    root_state_w: torch.Tensor = None


    @property
    def root_pos_w(self) -> torch.Tensor:
        """ Root position in simulation world frame. Shape: (num_envs, 3)"""
        return self.root_state_w[:, :3]
    
    @property
    def root_quat_w(self) -> torch.Tensor:
        """ Root quaternion in simulation world frame. Shape: (num_envs, 4)"""
        return self.root_state_w[:, 3:7]
    
    @property
    def root_vel_w(self) -> torch.Tensor:
        """ Root velocity in simulation world frame. Shape: (num_envs, 3)"""
        return self.root_state_w[:, 7:10]
    
    @property
    def root_angular_vel_w(self) -> torch.Tensor:
        """ Root angular velocity in simulation world frame. Shape: (num_envs, 3)"""
        return self.root_state_w[:, 10:13]
    
