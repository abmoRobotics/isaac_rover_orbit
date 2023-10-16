# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import torch
from typing import Dict, Optional, Sequence

from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrimView
from pxr import PhysxSchema

#import omni.isaac.orbit.utils.kit as kit_utils
#from omni.isaac.orbit.utils.math import combine_frame_transforms, quat_rotate_inverse, subtract_frame_transforms

from omni.isaac.orbit.robots.robot_base import RobotBase
from .mobile_robot_cfg import MobileRobotCfg
from .mobile_robot_data import MobileRobotData

from omni.isaac.orbit.utils.math import quat_rotate_inverse

class MobileRobot(RobotBase):

    cfg: MobileRobotCfg


    def __init__(self, cfg: MobileRobotCfg):
        # Initialize base class
        super().__init__(cfg)
        # Container for data access
        self._data = MobileRobotData()

    """
    Properties
    """

    @property
    def mobile_robot_num_dof(self) -> int:
        """Number of degrees of freedom of the mobile robot."""
        return self.cfg.meta_info.mobile_robot_num_dof
    
    @property
    def data(self) -> MobileRobotData:
        """Data access object."""
        return self._data
    
    """
    Operations
    """

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        # Spawn the robot and set its location
        super().spawn(prim_path, translation, orientation)
        
    def initialize(self, prim_paths_expr: Optional[str] = None):
        # Initialize base class
        super().initialize(prim_paths_expr)
        # Other stuff

    def update_buffers(self, dt: float):
        # Update base class
        super().update_buffers(dt)
        # Other stuff

        self._data.root_vel[:, 0:3] = quat_rotate_inverse(self._data.root_quat_w, self._data.root_lin_vel_w)
        self._data.root_vel[:, 3:6] = quat_rotate_inverse(self._data.root_quat_w, self._data.root_ang_vel_w)
        self._data.projected_gravity[:] = quat_rotate_inverse(self._data.root_quat_w, self._GRAVITY_VEC_W)


    def _create_buffers(self):

        # Create base class buffers
        super()._create_buffers()

        # Constants
        self._GRAVITY_VEC_W = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.count, 1)

        # Mobile robot frame states -- base
        self._data.root_vel = torch.zeros((self.count, 6), device=self.device).repeat(self.count, 1)
        self._data.projected_gravity = torch.zeros(self.count, 3, dtype=torch.float, device=self.device)

        self._data.base_dof_pos = self._data.dof_pos[:, : self.mobile_robot_num_dof]
        self._data.base_dof_vel = self._data.dof_vel[:, : self.mobile_robot_num_dof]
        self._data.base_dof_acc = self._data.dof_acc[:, : self.mobile_robot_num_dof]

        