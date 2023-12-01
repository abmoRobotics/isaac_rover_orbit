from dataclasses import MISSING
from typing import Optional, Sequence, Tuple

from omni.isaac.orbit.robots.robot_base_cfg import RobotBaseCfg
from omni.isaac.orbit.utils import configclass


@configclass
class MobileRobotCfg(RobotBaseCfg):

    @configclass
    class MetaInfoCfg(RobotBaseCfg.MetaInfoCfg):
        """ Metadata for the mobile robot. """

        mobile_robot_num_dof: int = MISSING
        """ Number of degrees of freedom of the mobile robot. """

        mobile_robot_steering_dof: int = MISSING
        """ Number of degrees of freemdom for steering the mobile robot. """

        mobile_robot_drive_dof: int = MISSING
        """ Number of degrees of freedom for driving the mobile robot. """

        mobile_robot_passive_dof: int = MISSING
        """ Number of degrees of freedom for passive joints. """

    meta_info: MetaInfoCfg = MetaInfoCfg()
