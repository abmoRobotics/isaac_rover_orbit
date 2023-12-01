# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
from typing import Dict, Optional, Sequence

import torch
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrimView
# Import stage
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.orbit.robots.robot_base import RobotBase
from omni.isaac.orbit.utils.math import quat_rotate_inverse
from pxr import PhysxSchema

from .mobile_robot_cfg import MobileRobotCfg
from .mobile_robot_data import MobileRobotData

#import omni.isaac.orbit.utils.kit as kit_utils
#from omni.isaac.orbit.utils.math import combine_frame_transforms, quat_rotate_inverse, subtract_frame_transforms




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
        # Other stuff
        #self.prepare_contact_reporter("/World/defaultGroundPlane/GroundPlane/CollisionPlane")
        # stage = get_current_stage()
        # prim = stage.GetPrimAtPath(prim_path)
        # ground_plane_prim = stage.GetPrimAtPath("/World/defaultGroundPlane/GroundPlane/CollisionPlane")
        # for link_prim in prim.GetChildren():
        #     if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        #      if "Looks" not in str(link_prim.GetPrimPath()):
        #          rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
        #          rb.CreateSleepThresholdAttr().Set(0)
        #          cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
        #          cr_api.CreateThresholdAttr().Set(0)
        #          cr_api.CreateReportPairsRel().AddTarget("/World/defaultGroundPlane/GroundPlane/CollisionPlane")#.Create(ground_plane_prim)

    def prepare_contact_reporter(self, target_prim_path: str):
        """ Prepare the contact reporter for the robot.

        Attributes
        ----------
        target_prim_path : str
            The prim path of the target to report contacts to.

        """
        prim_path = self._spawn_prim_path
        prim = get_current_stage().GetPrimAtPath(prim_path)
        for link in prim.GetChildren():
            if link.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                contact_reporter_keywords = ["Drive", "Steer", "Boogie", "Body"]
                if any(keyword in str(link.GetPrimPath()) for keyword in contact_reporter_keywords):
                    contact_report_api: PhysxSchema._physxSchema.PhysxContactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(link)
                    contact_report_api.CreateThresholdAttr().Set(0)
                    contact_report_api.CreateReportPairsRel().AddTarget(target_prim_path)
            #     and "Drive" in str(link.GetPrimPath()):
            #     contact_report_api: PhysxSchema._physxSchema.PhysxContactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(link)
            #     contact_report_api.CreateThresholdAttr().Set(0)
            #     contact_report_api.CreateReportPairsRel().AddTarget(target_prim_path)
            # elif link.HasAPI(PhysxSchema.PhysxRigidBodyAPI) and "Steer" in str(link.GetPrimPath()):
            #     contact_report_api: PhysxSchema._physxSchema.PhysxContactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(link)
            #     contact_report_api.CreateThresholdAttr().Set(0)
            #     contact_report_api.CreateReportPairsRel().AddTarget(target_prim_path)
            # elif link.HasAPI(PhysxSchema.PhysxRigidBodyAPI) and "Boogie" in str(link.GetPrimPath()):
            #     contact_report_api: PhysxSchema._physxSchema.PhysxContactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(link)
            #     contact_report_api.CreateThresholdAttr().Set(0)
            #     contact_report_api.CreateReportPairsRel().AddTarget(target_prim_path)
            # elif link.HasAPI(PhysxSchema.PhysxRigidBodyAPI) and "Body" in str(link.GetPrimPath()):
            #     contact_report_api: PhysxSchema._physxSchema.PhysxContactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(link)
            #     contact_report_api.CreateThresholdAttr().Set(0)
            #     contact_report_api.CreateReportPairsRel().AddTarget(target_prim_path)


    def initialize(self, prim_paths_expr: Optional[str] = None):
        # Initialize base class
        super().initialize(prim_paths_expr)
        # Other stuff

    def update_buffers(self, dt: float):
        # Update base class
        super().update_buffers(dt)
        # Other stuff

        # self._data.root_pos[:] = self._data.root_pos_w

        # self._data.root_vel[:, 0:3] = quat_rotate_inverse(self._data.root_quat_w, self._data.root_lin_vel_w)
        # self._data.root_vel[:, 3:6] = quat_rotate_inverse(self._data.root_quat_w, self._data.root_ang_vel_w)
        # self._data.projected_gravity[:] = quat_rotate_inverse(self._data.root_quat_w, self._GRAVITY_VEC_W)


    def _create_buffers(self):

        # Create base class buffers
        super()._create_buffers()

        # Constants
        self._GRAVITY_VEC_W = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.count, 1)

        self._data.root_pos = torch.zeros(self.count, 3, device=self.device)
        # Mobile robot frame states -- base
        self._data.root_vel = torch.zeros(self.count, 6, device=self.device)
        self._data.projected_gravity = torch.zeros(self.count, 3, dtype=torch.float, device=self.device)

        self._data.base_dof_pos = self._data.dof_pos[:, : self.mobile_robot_num_dof]
        self._data.base_dof_vel = self._data.dof_vel[:, : self.mobile_robot_num_dof]
        self._data.base_dof_acc = self._data.dof_acc[:, : self.mobile_robot_num_dof]
