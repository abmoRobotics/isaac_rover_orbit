from __future__ import annotations

import math
from typing import TYPE_CHECKING
import carb

import torch
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.envs import BaseEnv
from omni.isaac.orbit.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv  # noqa: F811

    from . import actions_cfg


class AckermannAction(ActionTerm):

    cfg: actions_cfg.AckermannActionCfg

    _asset: Articulation

    _wheelbase_length: float

    _middle_wheel_distance: float

    _rear_and_front_wheel_distance: float

    _min_steering_radius: float

    _steering_joint_names: list[str]

    _drive_joint_names: list[str]

    _scale: torch.Tensor

    _offset: torch.Tensor

    def __init__(self, cfg: actions_cfg.AckermannActionCfg, env: BaseEnv):
        super().__init__(cfg, env)

        self._drive_joint_ids, self._drive_joint_names = self._asset.find_joints(self.cfg.drive_joint_names)
        self._steering_joint_ids, self._steering_joint_names = self._asset.find_joints(self.cfg.steering_joint_names)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel = torch.zeros(self.num_envs, len(self._drive_joint_ids), device=self.device)
        self._joint_pos = torch.zeros(self.num_envs, len(self._steering_joint_ids), device=self.device)

        # cfgs
        self._wheelbase_length = cfg.wheelbase_length
        self._middle_wheel_distance = cfg.middle_wheel_distance
        self._rear_and_front_wheel_distance = cfg.rear_and_front_wheel_distance
        self._min_steering_radius = cfg.min_steering_radius

        # Save the scale and offset for the actions
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 2  # Assuming a 2D action vector (linear velocity, angular velocity)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset

    def apply_actions(self):

        self._joint_pos, self._joint_vel = self.ackermann(self._processed_actions[:, 0], self._processed_actions[:, 1])

        self._asset.set_joint_velocity_target(self._joint_vel, joint_ids=self._drive_joint_ids)
        self._asset.set_joint_position_target(self._joint_pos, joint_ids=self._steering_joint_ids)

    def ackermann(self, lin_vel, ang_vel):
        device = self.device
        # All measurements in Meters!
        num_robots = lin_vel.shape[0]
        wheel_diameter = 0.2
        # Locations of the wheels, with respect to center(between middle wheels) (X is right, Y is forward)
        wheel_FL = torch.unsqueeze(torch.transpose(torch.tensor(
            [[-0.385], [0.438]],  device=device).repeat(1, num_robots), 0, 1), 0)
        wheel_FR = torch.unsqueeze(torch.transpose(torch.tensor(
            [[0.385], [0.438]],   device=device).repeat(1, num_robots), 0, 1), 0)
        wheel_ML = torch.unsqueeze(torch.transpose(torch.tensor(
            [[-0.447], [0.0]],    device=device).repeat(1, num_robots), 0, 1), 0)
        wheel_MR = torch.unsqueeze(torch.transpose(torch.tensor(
            [[0.447], [0.0]],     device=device).repeat(1, num_robots), 0, 1), 0)
        wheel_RL = torch.unsqueeze(torch.transpose(torch.tensor(
            [[-0.385], [-0.411]], device=device).repeat(1, num_robots), 0, 1), 0)
        wheel_RR = torch.unsqueeze(torch.transpose(torch.tensor(
            [[0.385], [-0.411]],  device=device).repeat(1, num_robots), 0, 1), 0)

        # Wheel locations, collected in a single variable
        wheel_locations = torch.cat((wheel_FL, wheel_FR, wheel_ML, wheel_MR, wheel_RL, wheel_RR), 0)

        # The distance at which the rover should switch to turn on the spot mode.
        bound = 0.45

        # Turning point
        P = torch.unsqueeze(lin_vel/ang_vel, 0)
        P = torch.copysign(P, -ang_vel)
        zeros = torch.zeros_like(P)
        P = torch.transpose(torch.cat((P, zeros), 0), 0, 1)  # Add a zero component in the y-direction.
        # If turning point is between wheels, turn on the spot.
        P[:, 0] = torch.squeeze(torch.where(torch.abs(P[:, 0]) > bound, P[:, 0], zeros))
        lin_vel = torch.where(P[:, 0] != 0, lin_vel, zeros)  # If turning on the spot, set lin_vel = 0.

        # Calculate distance to turning point
        P = P.repeat((6, 1, 1))
        dist = torch.transpose((P - wheel_locations).pow(2).sum(2).sqrt(), 0, 1)

        # Motors on the left should turn opposite direction
        motor_side = torch.transpose(torch.tensor(
            [[-1.0], [1.0], [-1.0], [1.0], [-1.0], [1.0]], device=device).repeat((1, num_robots)), 0, 1)

        # When not turning on the spot, wheel velocity is actually determined by the linear direction
        wheel_linear = torch.transpose(torch.copysign(ang_vel, lin_vel).repeat((6, 1)), 0, 1)
        # When turning on the spot, wheel velocity is determined by motor side.
        wheel_turning = torch.transpose(ang_vel.repeat((6, 1)), 0, 1) * motor_side
        ang_velocities = torch.where(torch.transpose(lin_vel.repeat((6, 1)), 0, 1) != 0, wheel_linear, wheel_turning)

        # The velocity is determined by the disance from the wheel to the turning point, and the angular velocity the
        # wheel should travel with
        motor_velocities = dist * ang_velocities

        # If the turning point is more than 1000 meters away, just go straight.
        motor_velocities = torch.where(dist > 1000, torch.transpose(lin_vel.repeat((6, 1)), 0, 1), motor_velocities)

        # Convert linear velocity above ground to rad/s
        motor_velocities = (motor_velocities/wheel_diameter)

        steering_angles = torch.transpose(
            torch.where(
                torch.abs(wheel_locations[:, :, 0]) > torch.abs(P[:, :, 0]),
                torch.atan2(wheel_locations[:, :, 1], wheel_locations[:, :, 0] - P[:, :, 0]),
                torch.atan2(wheel_locations[:, :, 1], wheel_locations[:, :, 0] - P[:, :, 0])
            ), 0, 1)
        steering_angles = torch.where(steering_angles < -3.14/2, steering_angles + math.pi, steering_angles)
        steering_angles = torch.where(steering_angles > 3.14/2, steering_angles - math.pi, steering_angles)
        # print(torch.stack([steering_angles, motor_velocities], dim=1).shape)
        return torch.cat([steering_angles[:, 0:2], steering_angles[:, 4:6]], dim=1), motor_velocities


class AckermannAction2(ActionTerm):

    cfg: actions_cfg.AckermannActionCfg

    _asset: Articulation

    _wheelbase_length: float

    _middle_wheel_distance: float

    _rear_and_front_wheel_distance: float

    _wheel_radius: float

    _min_steering_radius: float

    _steering_joint_names: list[str]

    _drive_joint_names: list[str]

    _scale: torch.Tensor

    _offset: torch.Tensor

    def __init__(self, cfg: actions_cfg.AckermannActionCfg, env: BaseEnv):
        super().__init__(cfg, env)

        self._drive_joint_ids, self._drive_joint_names = self._asset.find_joints(self.cfg.drive_joint_names)
        self._steering_joint_ids, self._steering_joint_names = self._asset.find_joints(self.cfg.steering_joint_names)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel = torch.zeros(self.num_envs, len(self._drive_joint_ids), device=self.device)
        self._joint_pos = torch.zeros(self.num_envs, len(self._steering_joint_ids), device=self.device)

        # cfgs
        self._wheelbase_length = cfg.wheelbase_length
        self._middle_wheel_distance = cfg.middle_wheel_distance
        self._rear_and_front_wheel_distance = cfg.rear_and_front_wheel_distance
        self._wheel_radius = cfg.wheel_radius
        self._min_steering_radius = cfg.min_steering_radius

        # Save the scale and offset for the actions
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 2  # Assuming a 2D action vector (linear velocity, angular velocity)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset

    def apply_actions(self):

        self._joint_pos, self._joint_vel = self.ackermann(self._processed_actions[:, 0], self._processed_actions[:, 1])

        self._asset.set_joint_velocity_target(self._joint_vel, joint_ids=self._drive_joint_ids)
        self._asset.set_joint_position_target(self._joint_pos, joint_ids=self._steering_joint_ids)

    def ackermann(self, lin_vel, ang_vel):
        """
        Ackermann steering model for the rover
        wl = wheelbase length
        d_fr = distance between front and rear wheels
        d_mw = distance between middle wheels
        """
        wheel_radius = self._wheel_radius  # wheel radius
        d_fr = self._rear_and_front_wheel_distance  # distance between front and rear wheels
        d_mw = self._middle_wheel_distance  # distance between middle wheels
        wl = self._wheelbase_length  # wheelbase length
        device = self.device  # device

        # Checking the direction of the linear and angular velocities
        direction: torch.Tensor = torch.sign(lin_vel)
        turn_direction: torch.Tensor = torch.sign(ang_vel)

        direction = torch.where(direction == 0, direction+1,direction)

        # Taking the absolute values of the velocities
        lin_vel = torch.abs(lin_vel)
        ang_vel = torch.abs(ang_vel)

        # Calculates the turning radius of the rover, returns inf if ang_vel is 0
        not_zero_condition = torch.logical_not(ang_vel == 0) | torch.logical_not(lin_vel == 0)

        minimum_radius = (d_mw * 0.8)  # should be 0.5 but 0.8 makes operation more smooth
        turning_radius: torch.Tensor = torch.where(
            not_zero_condition, lin_vel / ang_vel, torch.tensor(float('inf'), device=device))
        turning_radius = torch.where(turning_radius < minimum_radius, minimum_radius, turning_radius)
        
        # if turn_radius is shorter than half of wheelbase: point turn else ackermann
        # Calculating the turning radius of the front wheels
        r_ML = turning_radius - (d_mw / 2)
        r_MR = turning_radius + (d_mw / 2)
        r_FL = turning_radius - (d_fr / 2)
        r_FR = turning_radius + (d_fr / 2)
        r_RL = turning_radius - (d_fr / 2)
        r_RR = turning_radius + (d_fr / 2)
        
        # Point turn or ackermann
        # Wheel velocities (m/s)
        # If turning radius is less than distance between middle wheels
        # Set velocities for point turn, else
        # if ang_vel is 0, wheel velocity is equal to linear velocity
        vel_FL = torch.where(turning_radius < d_mw , 
                             -(lin_vel+1)*turn_direction, 
                             torch.where(ang_vel == 0, lin_vel, (r_FL * ang_vel)) * direction)
        vel_FR = torch.where(turning_radius < d_mw , 
                             (lin_vel+1)*turn_direction, 
                             torch.where(ang_vel == 0, lin_vel, (r_FR * ang_vel)) * direction)
        vel_RL = torch.where(turning_radius < d_mw , 
                             -(lin_vel+1)*turn_direction, 
                             torch.where(ang_vel == 0, lin_vel, (r_RL * ang_vel)) * direction)
        vel_RR = torch.where(turning_radius < d_mw , 
                             (lin_vel+1)*turn_direction, 
                             torch.where(ang_vel == 0, lin_vel, (r_RR * ang_vel)) * direction)
        vel_ML = torch.where(turning_radius < d_mw , 
                             -(lin_vel+1)*turn_direction, 
                             torch.where(ang_vel == 0, lin_vel, (r_ML * ang_vel)) * direction)
        vel_MR =  torch.where(turning_radius < d_mw , 
                             (lin_vel+1)*turn_direction, 
                             torch.where(ang_vel == 0, lin_vel, (r_MR * ang_vel)) * direction)

        wl = torch.ones_like(r_FL) * wl  # Repeat wl as tensor
        
        # Steering angles for specifically point turns
        # If turning radius is less than the distance between middle wheels
        # set steering angles for point turn, else
        # ackermann       
        theta_FL = torch.where(turning_radius < d_mw,
                            torch.tensor(-(torch.pi/4), device=device),
                            torch.atan2(wl, r_FL) * turn_direction)
        theta_RR = torch.where(turning_radius < d_mw,
                            torch.tensor(-(torch.pi/4), device=device),
                            torch.atan2(wl, r_FL) * turn_direction)
        theta_FR = torch.where(turning_radius < d_mw,
                            torch.tensor((torch.pi/4), device=device),
                            torch.atan2(wl, r_FL) * turn_direction)
        theta_RL = torch.where(turning_radius < d_mw,
                            torch.tensor((torch.pi/4), device=device),
                            torch.atan2(wl, r_FL) * turn_direction)
        
        wheel_velocities = torch.stack([vel_ML, vel_FL, vel_RL, vel_RR, vel_MR, vel_FR], dim=1)
        steering_angles = torch.stack([theta_FL, theta_RL, theta_RR, theta_FR], dim=1)

            # Convert wheel velocities from m/s to rad/s
        wheel_velocities = wheel_velocities / wheel_radius

        return steering_angles, wheel_velocities  # torch.cat([steering_angles, wheel_velocities], dim=1)
    