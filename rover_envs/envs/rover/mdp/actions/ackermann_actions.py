from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.envs import BaseEnv
from omni.isaac.orbit.managers.action_manager import ActionTerm
from omni.isaac.orbit.managers.manager_term_cfg import ActionTermCfg

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

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

        self._joint_pos, self._joint_vel = self.ackermann(self._processed_actions[:,0], self._processed_actions[:,1])

        self._asset.set_joint_velocity_target(self._joint_vel, joint_ids=self._drive_joint_ids)
        self._asset.set_joint_position_target(self._joint_pos, joint_ids=self._steering_joint_ids)

    def ackermann(self, lin_vel, ang_vel):
        device = self.device
        # All measurements in Meters!
        num_robots = lin_vel.shape[0]
        wheel_diameter = 0.2
        # Locations of the wheels, with respect to center(between middle wheels) (X is right, Y is forward)
        wheel_FL = torch.unsqueeze(torch.transpose(torch.tensor(  [[-0.385],[0.438]],  device=device).repeat(1,num_robots), 0, 1),0)
        wheel_FR = torch.unsqueeze(torch.transpose(torch.tensor(  [[0.385],[0.438]],   device=device).repeat(1,num_robots), 0, 1),0)
        wheel_ML = torch.unsqueeze(torch.transpose(torch.tensor(  [[-0.447],[0.0]],    device=device).repeat(1,num_robots), 0, 1),0)
        wheel_MR = torch.unsqueeze(torch.transpose(torch.tensor(  [[0.447],[0.0]],     device=device).repeat(1,num_robots), 0, 1),0)
        wheel_RL = torch.unsqueeze(torch.transpose(torch.tensor(  [[-0.385],[-0.411]], device=device).repeat(1,num_robots), 0, 1),0)
        wheel_RR = torch.unsqueeze(torch.transpose(torch.tensor(  [[0.385],[-0.411]],  device=device).repeat(1,num_robots), 0, 1),0)

        # Wheel locations, collected in a single variable
        wheel_locations = torch.cat((wheel_FL, wheel_FR, wheel_ML, wheel_MR, wheel_RL, wheel_RR), 0)

        # The distance at which the rover should switch to turn on the spot mode.
        bound = 0.45

        # Turning point
        P = torch.unsqueeze(lin_vel/ang_vel, 0)
        P = torch.copysign(P, -ang_vel)
        zeros = torch.zeros_like(P)
        P = torch.transpose(torch.cat((P,zeros), 0), 0, 1) # Add a zero component in the y-direction.
        P[:,0] = torch.squeeze(torch.where(torch.abs(P[:,0]) > bound, P[:,0], zeros)) # If turning point is between wheels, turn on the spot.
        lin_vel = torch.where(P[:,0] != 0, lin_vel, zeros) # If turning on the spot, set lin_vel = 0.

        # Calculate distance to turning point
        P = P.repeat((6,1,1))
        dist = torch.transpose((P - wheel_locations).pow(2).sum(2).sqrt(), 0, 1)

        # Motors on the left should turn opposite direction
        motor_side = torch.transpose(torch.tensor([[-1.0],[1.0],[-1.0],[1.0],[-1.0],[1.0]], device=device).repeat((1, num_robots)), 0, 1)

        # When not turning on the spot, wheel velocity is actually determined by the linear direction
        wheel_linear = torch.transpose(torch.copysign(ang_vel, lin_vel).repeat((6,1)), 0, 1)
        # When turning on the spot, wheel velocity is determined by motor side.
        wheel_turning = torch.transpose(ang_vel.repeat((6,1)), 0, 1) * motor_side
        ang_velocities = torch.where(torch.transpose(lin_vel.repeat((6,1)), 0, 1) != 0, wheel_linear, wheel_turning)

        # The velocity is determined by the disance from the wheel to the turning point, and the angular velocity the wheel should travel with
        motor_velocities = dist * ang_velocities

        # If the turning point is more than 1000 meters away, just go straight.
        motor_velocities = torch.where(dist > 1000, torch.transpose(lin_vel.repeat((6,1)), 0, 1), motor_velocities)

        # Convert linear velocity above ground to rad/s
        motor_velocities = (motor_velocities/wheel_diameter)

        steering_angles = torch.transpose(torch.where(torch.abs(wheel_locations[:,:,0]) > torch.abs(P[:,:,0]), torch.atan2(wheel_locations[:,:,1], wheel_locations[:,:,0] - P[:,:,0]), torch.atan2(wheel_locations[:,:,1], wheel_locations[:,:,0] - P[:,:,0])), 0, 1)
        steering_angles = torch.where(steering_angles < -3.14/2, steering_angles + math.pi, steering_angles)
        steering_angles = torch.where(steering_angles > 3.14/2, steering_angles - math.pi, steering_angles)
        #print(torch.stack([steering_angles, motor_velocities], dim=1).shape)
        return torch.cat([steering_angles[:,0:2], steering_angles[:,4:6]], dim=1), motor_velocities