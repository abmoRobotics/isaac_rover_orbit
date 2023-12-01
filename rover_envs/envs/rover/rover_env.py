
import math
from typing import List

import gym.spaces
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.utils.kit as kit_utils
import torch
# omni.isaac.core imports
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.orbit.markers import PointMarker
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager
# omni.isaac.orbit imports
from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs

from rover_envs.robots.mobile_robot import MobileRobot

# ENV
from .rover_cfg import RandomizationCfg, RoverEnvCfg
from .utils.camera.camera import Camera
from .utils.terrain_utils.terrain_utils import TerrainManager


class RoverEnv(IsaacEnv):

    def __init__(self, cfg: RoverEnvCfg, **kwargs):

        self.cfg = cfg
        # parse the pre_process_cfg
        self._pre_process_cfg()

        # create classes
        self.robot = MobileRobot(cfg=self.cfg.robot)
        super().__init__(cfg, **kwargs)
        # Get initial rover state


        # Initialize the base class to setup the scene
        self._process_cfg()
        self._initialize_views()

        # Observation and reward manager
        self._observation_manager = RoverObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        self._reward_manager = RoverRewardManager(class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device)

        # Print information about observation and reward manager
        print("[INFO] Observation manager: ", self._observation_manager)
        print("[INFO] Reward manager: ", self._reward_manager)

        num_obs = self._observation_manager.group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        print(f'[INFO] Observation space: {self.observation_space}')
        print(f'[INFO] Action space: {self.action_space}')
        # take the intial step
        # self.camera = Camera(self.device, torch.tensor([0.0, 0.0, 0.0],device=self.device), False)

        self.sim.step()

        # -- fill up buffers
        self.robot.update_buffers(self.dt)

        self._post_process_cfg()
        print(f'[INFO] Max episode length: {self.max_episode_length}')
        print(f'[INFO] Rover dt: {self.dt}')
        print(f'[INFO] Rover decimation: {self.cfg.control.decimation}')
        print(f'[INFO] Rover substeps: {self.cfg.sim.substeps}')
        print(f'[INFO] Rover episode length: {self.cfg.env.episode_length_s}')


    def _design_scene(self) -> List[str]:
        self._terrainManager = TerrainManager(device=self.device)

        # If raycasting is enabled, create a mesh for the terrain.
        if self.cfg.observations.policy.enable_ray_height:
            self._terrainManager.mesh_to_omni_stage()
        else:
            # If raycasting is not enabled, create a heightmap for the terrain.
            kit_utils.create_ground_plane("/World/defaultGroundPLane",z_position=-.4)


        # Spawn a singular robot in the scene at "/World/envs/env_0/Robot"
        self.robot.spawn(self.template_env_ns + "/Robot")

        # If we want to track collisions with rocks we need to prepare the contact reporter
        if self.cfg.terminations.collision_with_rock:
            self.robot.prepare_contact_reporter("/World/rock")

        if self.cfg.viewer.debug_vis and self.enable_render:

            self._goal_markers = PointMarker(
                "/Visuals/goal_marker",
                self.num_envs,
                radius=0.1,
            )

        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: VecEnvIndices):
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)

        # Reset the rover
        self._reset_rover_state(env_ids)

        # Generate a random target position
        self._randomize_target(env_ids, self.cfg.randomization.object_target_position)

        self.extras["episode"] = dict()

        # Reset Reward and Observation Manager
        self._reward_manager.reset_idx(env_ids, self.extras["episode"])
        self._observation_manager.reset_idx(env_ids)

        # Reset History Buffers
        self.prev_actions[env_ids] = 0

        # Reset MDP Buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _step_impl(self, actions: torch.Tensor):

        # Pre-step actions
        self.actions = actions.clone().to(self.device)
        #print(f'[INFO] Rover actions: {self.actions[0]}')
        # TODO Remove these comments
        # self.actions[:, 0] = 1.0 * self.cfg.control.action_scaling
        # self.actions[:, 1] = 0.0 * self.cfg.control.action_scaling
        self.actions_scaled = self.actions * self.cfg.control.action_scaling

        # TODO: Implement controller here
        self.robot_actions[:,:10] = Ackermann2(lin_vel = self.actions_scaled[:, 0], ang_vel=self.actions_scaled[:, 1], device=self.device)

        # Perform physics steps
        for _ in range(self.cfg.control.decimation):
            # set actions into buffer
            self.robot.apply_action(self.robot_actions)
            # Simulate a step
            self.sim.step(render=self.enable_render)

            if self.sim.is_stopped():
                return

        # Post step
        self.robot.update_buffers(self.dt)

        self.reward_buf = self._reward_manager.compute()

        # Terminations
        self._check_termination()

        self.prev_actions = self.actions.clone()

        #rotation = tensor_quat_to_eul(self.robot.data.root_quat_w, device=self.device)
        # heightmap, output_pt, sources = self.camera.get_depths(positions=self.robot.data.root_pos_w, rotations=rotation)
        # sparse = self.camera.heightmap.get_sparse_vector(heightmap)
        # dense = self.camera.heightmap.get_dense_vector(heightmap)
        #print(sparse.shape)

        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        distance = torch.norm(self.target_pose[:, 0:2] - self.robot.data.root_pos_w[:, 0:2], dim=1)
        self.extras["is_success"] = torch.where(distance < 0.18, 1, self.reset_buf)

        # Update USD
        if self.cfg.viewer.debug_vis and self.enable_render:
            self._debug_vis()


            if self.cfg.observations.policy.enable_ray_height and False:
                self.camera.draw_depths()


    def _get_observations(self) -> VecEnvObs:
        return self._observation_manager.compute()

    def _setup_randomization(self):
        pass

    def _initialize_views(self):
        self.sim.reset()


        self.robot.initialize(self.env_ns + "/.*/Robot")

        # Create controller here later
        self.num_actions = 2
        self.num_joints = self.robot.cfg.meta_info.mobile_robot_num_dof
        # history buffer
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        self.robot_actions = torch.zeros((self.num_envs, self.num_joints), device=self.device)

        # Target goal pose
        self.target_pose = torch.zeros((self.num_envs, 7), device=self.device)

    def _pre_process_cfg(self):
        pass

    def _process_cfg(self):
        self.dt = self.cfg.control.decimation * self.physics_dt
        #self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / (self.dt*self.cfg.control.decimation*self.cfg.sim.substeps))
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)


        # TODO FIX THIS AND MAKE IT PART OF CFG -> shift 30, 30
        self.envs_positions += torch.tensor([30.0, 30.0, 5.0], device=self.device)

        if self.cfg.observations.policy.enable_ray_height:
            self.camera = Camera(self.device, torch.tensor([0.0, 0.0, 0.0],device=self.device), False)

    def _post_process_cfg(self):
        """ Contains cfg settings that must be set after the scene is setup and the physics is initialized """

        if self.cfg.terminations.collision_with_rock:
            # Define collision view and prepare contact_filter for force_matrix, since report_pairs is not working (isaac sim 2022.2.1)
            self.collisions_view = RigidPrimView("/World/envs/env.*/Robot/.*", name="collision_view",
                                                 reset_xform_properties=False,
                                                 track_contact_forces=True,
                                                 prepare_contact_sensors=False,
                                                 contact_filter_prim_paths_expr=["/World/rock"])
            # Initialize the view
            self.collisions_view.initialize()

    def _check_termination(self):
        self.reset_buf[:] = 0

        # Reset if the time is up
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)

        if self.cfg.terminations.is_success:
            distance = torch.norm(self.target_pose[:, 0:2] - self.robot.data.root_pos_w[:, 0:2], dim=1)
            self.reset_buf = torch.where(distance < 0.18, 1, self.reset_buf)

        if self.cfg.terminations.robot_distance_to_target:
            distance = torch.norm(self.target_pose[:, 0:2] - self.robot.data.root_pos_w[:, 0:2], dim=1)
            self.reset_buf = torch.where(distance > 15, 1, self.reset_buf)

        if self.cfg.terminations.collision_with_rock:
            # TODO: The number 6 should not be hardcoded
            normalized_forces = torch.norm(self.collisions_view.get_contact_force_matrix().view(self.num_envs, -1, 3), dim=1)
            forces_active = torch.sum(normalized_forces, dim=-1) > 1
            self.reset_buf = torch.where(forces_active, 1, self.reset_buf)

    def _debug_vis(self):
        """ Visualize the target position """

        # print(self.target_pose[:, 0:3])
        # indices = torch.tensor([0],device=self.device)
        # pos = torch.index_select(self.target_pose[:, 0:3], 0, indices)
        # ori = torch.index_select(self.target_pose[:, 3:7], 0, indices)
        # self._goal_markers.set_world_poses(pos, ori, indices)
        self._goal_markers.set_world_poses(self.target_pose[:, 0:3], self.target_pose[:, 3:7])


    def _randomize_target(self, env_ids: torch.Tensor, cfg: RandomizationCfg.TargetPositionCfg):
        """ Randomize the target position """
        radius = cfg.radius_default

        # Generate a random angle
        theta = torch.rand((len(env_ids),), device=self.device) * 2 * torch.pi

        # set the target position x and y
        self.target_pose[env_ids, 0] = radius * torch.cos(theta) + self.envs_positions[env_ids, 0]
        self.target_pose[env_ids, 1] = radius * torch.sin(theta) + self.envs_positions[env_ids, 1]

        # Ensures that the target is not placed on a rock
        self.target_pose[env_ids, 0:2] = self._terrainManager.get_valid_targets(self.target_pose[env_ids, 0:2], device=self.device)

        # Set the target height (for visualization purposes)
        self.target_pose[env_ids, 2] = self._terrainManager._heightmap_manager.get_height_at(self.target_pose[env_ids, 0:2])


    def _randomize_rover_orientation(self, env_ids):
        """ Randomize the rover orientation """
        angle = torch.rand((len(env_ids),), device=self.device) * 2 * torch.pi
        quat = torch.zeros((len(env_ids), 4), device=self.device)
        quat[:, 0] = torch.cos(angle / 2)
        quat[:, 3] = torch.sin(angle / 2)
        self.robot.set_root_state(quat, env_ids=env_ids)

    def _reset_rover_state(self, env_ids):
        """ Reset the rover state with random orientation"""
        # Get default state
        reset_state = self.robot.get_default_root_state(env_ids=env_ids)

        # Get the available spawn locations from the terrain manager
        spawn_locations = self._terrainManager.spawn_locations

        # Randomly select a spawn location
        #spawn_index = torch.randint(0, len(spawn_locations), (len(env_ids),), device=self.device)
        # Use random permutation instead of randint to avoid rover spawning in the same location (Will produce CUDA error)
        spawn_index = torch.randperm(len(spawn_locations), device=self.device)[0:len(env_ids)]

        # Set position
        self.envs_positions[env_ids] = self._terrainManager.spawn_locations[spawn_index]
        self.envs_positions[env_ids, 2] += 1

        reset_state[:, 0:3] = self.envs_positions[env_ids]

        # Orientation
        angle = torch.rand((len(env_ids),), device=self.device) * 2 * torch.pi
        quat = torch.zeros((len(env_ids), 4), device=self.device)
        quat[:, 0] = torch.cos(angle / 2)
        quat[:, 3] = torch.sin(angle / 2)
        reset_state[:, 3:7] = quat

        # Set State
        self.robot.set_root_state(reset_state, env_ids=env_ids)

class RoverObservationManager(ObservationManager):

    def distance_to_goal(self, env: RoverEnv):
        """Distance to the goal x, y """
        d = torch.norm(env.target_pose[:, 0:2] - env.robot.data.root_pos_w[:, 0:2], dim=1)
        #(f'[INFO] Distance to goal: {d[0]}')
        return torch.norm(env.target_pose[:, 0:2] - env.robot.data.root_pos_w[:, 0:2], dim=1).unsqueeze(1)

    def angle_to_goal(self, env: RoverEnv):
        """Angle to the goal"""

        # 1. convert the robot quaternion to a euler angle
        rover_rotation_euler = tensor_quat_to_eul(env.robot.data.root_quat_w[:, 0:4], device=env.device)

        # Two calculate direction of rover, and direction to target from rover
        direction_vector = torch.zeros([env.num_envs, 2], device=self._device)
        direction_vector[:,0] = torch.cos(rover_rotation_euler[:, 2])
        direction_vector[:,1] = torch.sin(rover_rotation_euler[:, 2])
        target_vector = env.target_pose[:, 0:2] - env.robot.data.root_pos_w[:, 0:2]

        cross_product = direction_vector[:,1] * target_vector[:,0] - direction_vector[:,0] * target_vector[:,1]
        dot_product = direction_vector[:,0] * target_vector[:,0] + direction_vector[:,1] * target_vector[:,1]

        d = torch.atan2(cross_product, dot_product)
        #print(f'[INFO] Angle to goal: {d[0]}')
        return torch.atan2(cross_product, dot_product).unsqueeze(1)

    def rover_actions(self, env: RoverEnv):
        """Last actions provided to the environment."""
        #print(f'[INFO] Rover old actions: {env.prev_actions[0]}')
        return env.actions

    def rover_ray_depth_map(self, env: RoverEnv):
        """Depth map from the rover camera."""

        # Get the rotation of the rover in euler angles
        rotation = tensor_quat_to_eul(env.robot.data.root_quat_w, device=env.device)
        # Get the depth map from the camera
        heightmap = env.camera.get_depths(positions=env.robot.data.root_pos_w, rotations=rotation)
        # Extract the sparse and dense vector from the heightmap
        # (some rays are used both in the sparse and dense vector that is why we extract them separately)
        sparse = env.camera.heightmap.get_sparse_vector(heightmap)
        dense = env.camera.heightmap.get_dense_vector(heightmap)

        # add them to one tensor
        return torch.cat([sparse, dense], dim=1)

class RoverRewardManager(RewardManager):

    def distance_to_target(self, env: RoverEnv):
        """Reward depending on the distance to the target"""
        distance = torch.norm(env.target_pose[:, 0:2] - env.robot.data.root_pos_w[:, 0:2], dim=1)
        return (1.0 / (1.0 + (0.33*0.33*distance*distance))) / env.max_episode_length

    def reached_goal_reward(self, env: RoverEnv):
        """Reward for reaching the goal"""
        distance = torch.norm(env.target_pose[:, 0:2] - env.robot.data.root_pos_w[:, 0:2], dim=1)
        return torch.where(distance < 0.18, 1.0*(env.max_episode_length-env.episode_length_buf) / env.max_episode_length , 0)

    def oscillation_penalty(self, env: RoverEnv):
        """Penalty for oscillating"""
        linear_difference = torch.abs(env.actions[:, 1] - env.prev_actions[:, 1])
        angular_difference = torch.abs(env.actions[:, 0] - env.prev_actions[:, 0])

        # TODO combine these 5 lines into two lines
        angular_penalty = torch.where(angular_difference > 0.05, torch.square(angular_difference), 0.0)
        linear_penalty = torch.where(linear_difference > 0.05, torch.square(linear_difference), 0.0)

        angular_penalty = torch.pow(angular_penalty, 2)
        linear_penalty = torch.pow(linear_penalty, 2)

        return (angular_penalty + linear_penalty) / env.max_episode_length

    def goal_angle_penalty(self, env: RoverEnv):

        rover_rotation_euler = tensor_quat_to_eul(env.robot.data.root_quat_w[:, 0:4], device=env.device)
        direction_vector = torch.zeros([env.num_envs, 2], device=self._device)
        direction_vector[:,0] = torch.cos(rover_rotation_euler[:, 2])
        direction_vector[:,1] = torch.sin(rover_rotation_euler[:, 2])
        target_vector = env.target_pose[:, 0:2] - env.robot.data.root_pos_w[:, 0:2]

        cross_product = direction_vector[:,1] * target_vector[:,0] - direction_vector[:,0] * target_vector[:,1]
        dot_product = direction_vector[:,0] * target_vector[:,0] + direction_vector[:,1] * target_vector[:,1]

        heading_difference = torch.atan2(cross_product, dot_product)

        return torch.where(torch.abs(heading_difference) > 2.0, torch.abs(heading_difference) / env.max_episode_length, 0.0)

    def heading_contraint_penalty(self, env: RoverEnv):
        """ Gives a penalty if the rover is driving backwards (since it has no rear camera) """
        return torch.where(env.actions[:, 0] < 0.0, (1.0 / env.max_episode_length), 0)

    def collision_penalty(self, env: RoverEnv):
        if env.cfg.terminations.collision_with_rock:
            normalized_forces = torch.norm(env.collisions_view.get_contact_force_matrix().view(env.num_envs, -1, 3), dim=1)
            forces_active = torch.sum(normalized_forces, dim=-1) > 1
            return torch.where(forces_active, 1.0, 0.0)
        else:
            return torch.zeros((env.num_envs), device=self._device)



def Ackermann(
        lin_vel: torch.Tensor,
        ang_vel: torch.Tensor,
        device: torch.device,
        wl = 0.849,
        d_fr = 0.77,
        d_mw = 0.894,
        wheel_radius = 0.1,
):
    """
    Ackermann steering model for the rover
    wl = wheelbase length
    d_fr = distance between front and rear wheels
    d_mw = distance between middle wheels
    """

    # Checking the direction of the linear and angular velocities
    direction: torch.Tensor = torch.sign(lin_vel)
    turn_direction: torch.Tensor = torch.sign(ang_vel)

    # Taking the absolute values of the velocities
    lin_vel = torch.abs(lin_vel)
    ang_vel = torch.abs(ang_vel)

    # Calculates the turning radius of the rover, returns inf if ang_vel is 0
    not_zero_condition = torch.logical_not(ang_vel == 0) & torch.logical_not(lin_vel == 0)

    minimum_radius = (d_mw * 0.8) # should be 0.5 but 0.8 makes operation more smooth
    turning_radius: torch.Tensor = torch.where(not_zero_condition, lin_vel / ang_vel, torch.tensor(float('inf'), device=device))
    turning_radius = torch.where(turning_radius < minimum_radius, minimum_radius, turning_radius)

    # Calculating the turning radius of the front wheels
    r_ML = turning_radius - (d_mw / 2)
    r_MR = turning_radius + (d_mw / 2)
    r_FL = turning_radius - (d_fr / 2)
    r_FR = turning_radius + (d_fr / 2)
    r_RL = turning_radius - (d_fr / 2)
    r_RR = turning_radius + (d_fr / 2)

    # Steering angles

    wl = torch.ones_like(r_FL) * wl # Repeat wl as tensor
    #print(turning_radius)
    theta_FL = torch.atan2(wl, r_FL) * turn_direction
    theta_FR = torch.atan2(wl, r_FR) * turn_direction
    theta_RL = -torch.atan2(wl, r_RL) * turn_direction
    theta_RR = -torch.atan2(wl, r_RR) * turn_direction

    # Wheel velocities (m/s)
    # if ang_vel is 0, wheel velocity is equal to linear velocity
    vel_FL = torch.where(ang_vel == 0, lin_vel, (r_FL * ang_vel)) * direction
    vel_FR = torch.where(ang_vel == 0, lin_vel, (r_FR * ang_vel)) * direction
    vel_RL = torch.where(ang_vel == 0, lin_vel, (r_RL * ang_vel)) * direction
    vel_RR = torch.where(ang_vel == 0, lin_vel, (r_RR * ang_vel)) * direction
    vel_ML = torch.where(ang_vel == 0, lin_vel, (r_ML * ang_vel)) * direction
    vel_MR = torch.where(ang_vel == 0, lin_vel, (r_MR * ang_vel)) * direction

    # Stack the wheel velocities and steering angles
    wheel_velocities = torch.stack([vel_FL, vel_FR, vel_RL, vel_RR, vel_ML, vel_MR], dim=1)
    steering_angles = torch.stack([theta_FL, theta_FR, theta_RL, theta_RR], dim=1)

    # Convert wheel velocities from m/s to rad/s
    wheel_velocities = wheel_velocities / wheel_radius

    return torch.cat([steering_angles, wheel_velocities], dim=1)

def tensor_quat_to_eul(quats, device):
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    # Quaternions format: W, X, Y, Z
    # Quat index:         0, 1, 2, 3
    # Euler angles:       ZYX

    euler_angles = torch.zeros([len(quats), 3], device=device)
    ones = torch.ones([len(quats)], device=device)
    zeros = torch.zeros([len(quats)], device=device)

    #Roll
    sinr_cosp = 2 * (quats[:,0] * quats[:,1] + quats[:,2] * quats[:,3])
    cosr_cosp = ones - (2 * (quats[:,1] * quats[:,1] + quats[:,2] * quats[:,2]))
    euler_angles[:,0] = torch.atan2(sinr_cosp, cosr_cosp)

    #Pitch
    sinp = 2 * (quats[:,0]*quats[:,2] - quats[:,3] * quats[:,1])
    condition = (torch.sign(sinp - ones) >= zeros)
    euler_angles[:,1] = torch.where(condition, torch.copysign((ones*torch.pi)/2, sinp), torch.asin(sinp))

    #Yaw
    siny_cosp = 2 * (quats[:,0] * quats[:,3] + quats[:,1] * quats[:,2])
    cosy_cosp = ones - (2 * (quats[:,2] * quats[:,2] + quats[:,3] * quats[:,3]))
    euler_angles[:,2] = torch.atan2(siny_cosp, cosy_cosp)

    return euler_angles

#@torch.jit.script
def Ackermann2(lin_vel, ang_vel, device='cuda:0'):

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

    return torch.cat([steering_angles[:,0:2], steering_angles[:,4:6], motor_velocities], dim=1)
