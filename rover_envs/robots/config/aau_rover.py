import os

from omni.isaac.orbit.actuators.group import (ActuatorControlCfg,
                                              ActuatorGroupCfg)
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg

from ..mobile_robot import MobileRobotCfg

#_AAU_ROVER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'assets', 'rover' ,'rover_instance.usd')
#_AAU_ROVER_PATH = "http://localhost:8080/omniverse://127.0.0.1/Projects/simple_instanceable_new/rover_instance.usd"
#_AAU_ROVER_PATH = "http://localhost:8080/omniverse://127.0.0.1/Projects/simplified9.usd"
_AAU_ROVER_PATH = "http://localhost:8080/omniverse://127.0.0.1/Projects/simple_instanceable_new2/rover_instance.usd"

AAU_ROVER_CFG = MobileRobotCfg(
    meta_info=MobileRobotCfg.MetaInfoCfg(
        usd_path=_AAU_ROVER_PATH,
        mobile_robot_num_dof=13,
        mobile_robot_steering_dof=4,
        mobile_robot_drive_dof=6,
        mobile_robot_passive_dof=3,
    ),
    init_state=MobileRobotCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
        dof_pos={".*Steer_Revolute": 0.0},
        dof_vel={".*Steer_Revolute": 0.0,
                 ".*Drive_Continuous": 0.0},
    ),
    rigid_props=MobileRobotCfg.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=2.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    collision_props=MobileRobotCfg.CollisionPropertiesCfg(
        contact_offset=0.04,
        rest_offset=0.01,
    ),
    articulation_props=MobileRobotCfg.ArticulationRootPropertiesCfg(
        enable_self_collisions=False,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=4,
    ),

    actuator_groups={
        "base_steering": ActuatorGroupCfg(
            dof_names=[".*Steer_Revolute"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=6, torque_limit=12),
            control_cfg=ActuatorControlCfg(command_types=["p_abs"],stiffness={".*": 8000.0},damping={".*": 1000.0}),
        ),
        "base_drive": ActuatorGroupCfg(
            dof_names=[".*Drive_Continuous"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=6, torque_limit=12),
            control_cfg=ActuatorControlCfg(command_types=["v_abs"],stiffness={".*": 100.0},damping={".*": 4000.0}),
        ),
        "passive_joints": ActuatorGroupCfg(
            dof_names=[".*Boogie_Revolute"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=15, torque_limit=0),
            control_cfg=ActuatorControlCfg(command_types=["p_abs"],stiffness={".*": 0.0},damping={".*": 0.0}),
        ),
    },


)
