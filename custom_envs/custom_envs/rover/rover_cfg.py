import os 
from typing import Tuple
import custom_orbit
from custom_orbit.robots.mobile_robot.mobile_robot import MobileRobotCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, SimCfg, ViewerCfg, PhysxCfg
from custom_orbit.robots.config.aau_rover import AAU_ROVER_CFG
CUSTOM_ORBIT_DIR = os.path.dirname(os.path.abspath(custom_orbit.__file__))

################################
# Scene Setup Configurations
################################

@configclass
class TerrainCfg:
    """ Configuration for the terrain. """

    # Use default ground plane
    use_default_ground_plane: bool = True

    # Use custom USD ground plane
    # usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Environment/Terrains/flat_plane.usd"

@configclass
class TargetCfg:
    """ Properties for target marker"""

    usd_path = f"{CUSTOM_ORBIT_DIR}/assets/ball/ball.usd"
    # Scale the asset by this array of 3 floats
    scale = [1.0, 1.0, 1.0]


################################
# Markov Decision Process Configurations
################################


@configclass
class RandomizationCfg:
    """ Randomization of scene at each episode (when reset) """

    @configclass
    class TargetPositionCfg:
        """ Randomization of target position """

        # Category
        position_category: str = "default"

        # Randomize target position
        radius_default = 9
        radius_uniform_min = 8
        radius_uniform_max = 10
    
    object_target_position: TargetPositionCfg = TargetPositionCfg()


@configclass
class ObservationsCfg:
    """ Observation scaling"""

    @configclass
    class PolicyCfg:
        """ Observation scaling for policy """
        # global group settings
        enable_corruption: bool = True
        
        rover_actions = {"scale": 1.0}
        angle_to_goal = {"scale": 0.33}
        distance_to_goal = {"scale": 0.11}
    
    return_dict_obs_in_group = False

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """ Reward terms and weights """

    distance_to_target = {"weight": 10.0}
    reached_goal_reward = {"weight": 10.0}
    oscillation_penalty = {"weight": -0.05}
    goal_angle_penalty = {"weight": -1.0}

@configclass
class TerminationsCfg:
    """ Termination terms for the environment """

    # Reset when episode length is reached
    episode_timeout = True
    # Reset when robot distance to target is too large
    robot_distance_to_target = True
    # Reset when target is reached
    is_success = True
    


@configclass
class ControlCfg:
    """ Processing of MDP actions """
    # Decimation: Number of physics steps per action 
    decimation: int = 4
    # scaling of actions
    action_scaling: float = 0.586 # Max speed = 2.11 km/h = 0.586 m/s
    # Clipping of actions
    action_clipping: float = 1.0

@configclass
class RoverEnvCfg(IsaacEnvCfg):
    """ Configuration for the Rover environment """

    env: EnvCfg = EnvCfg(num_envs=256, env_spacing=2.0, episode_length_s=300.0)
    viewer: ViewerCfg = ViewerCfg(debug_vis=True)
    # gpu_max_rigid_contact_count: 524288
    # gpu_max_rigid_patch_count: 81920
    # gpu_found_lost_pairs_capacity: 1024
    # gpu_found_lost_aggregate_pairs_capacity: 262144
    # gpu_total_aggregate_pairs_capacity: 2048
    # gpu_max_soft_body_contacts: 1048576
    # gpu_max_particle_contacts: 1048576
    # gpu_heap_capacity: 67108864
    # gpu_temp_buffer_capacity: 16777216
    # gpu_max_num_partitions: 8
    sim: SimCfg = SimCfg(
        dt=1/40,
        substeps=2,
        physx=PhysxCfg(
            solver_type=1,
            use_gpu=True,
            enable_stabilization=True,
            bounce_threshold_velocity = 0.2,
            friction_offset_threshold =  0.04,
            friction_correlation_distance = 0.025,

            gpu_max_rigid_contact_count=524288,
            gpu_max_rigid_patch_count=81920,
            gpu_found_lost_pairs_capacity=1024,
            gpu_found_lost_aggregate_pairs_capacity=262144,
            gpu_total_aggregate_pairs_capacity=2048,
            gpu_max_soft_body_contacts=1048576,
            gpu_max_particle_contacts=1048576,
            gpu_heap_capacity=67108864,
            gpu_temp_buffer_capacity=16777216,
            gpu_max_num_partitions=8,
        ),
        replicate_physics=False,
        device="cuda:0",
        use_flatcache=True,
        use_gpu_pipeline=True,
        enable_scene_query_support=False
    )

    # Scene settings
    # -- Terrain
    terrain: TerrainCfg = TerrainCfg()
    # -- Robot
    robot: MobileRobotCfg = AAU_ROVER_CFG
    # -- Target marker visualization
    target_marker: TargetCfg = TargetCfg()

    # MDP settings
    randomization: RandomizationCfg = RandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # Controller settings
    control: ControlCfg = ControlCfg()