from __future__ import annotations

import math
from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
#from omni.isaac.orbit.command_generators import UniformVelocityCommandGeneratorCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import ActionTermCfg as ActionTerm
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.sim import PhysxCfg
from omni.isaac.orbit.sim import SimulationCfg as SimCfg
from omni.isaac.orbit.terrains import TerrainImporter, TerrainImporterCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

import rover_envs.envs.rover.mdp as mdp
from rover_envs.envs.rover.utils.terrains.rover_terrain_importer import \
    RoverTerrainImporter
from rover_envs.robots.aau_rover import AAU_ROVER_CFG

##
# Scene Description
##

@configclass
class RoverSceneCfg(InteractiveSceneCfg):



    # Hidden Terrain (merged terrain of ground and obstacles) for raycaster. This is done because the raycaster doesn't work with multiple meshes
    hidden_terrain = AssetBaseCfg(
        prim_path="/World/terrain/hidden_terrain",
        spawn=sim_utils.UsdFileCfg(
            visible=False,
            usd_path="/home/anton/1._University/0._Master_Project/Workspace/terrain_generation/terrains/mars1/terrain_merged3.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Ground Terrain
    ground_terrain = TerrainImporterCfg(
        class_type=RoverTerrainImporter,
        prim_path="/World/terrain",
        terrain_type="usd",
        collision_group=-1,
        usd_path="/home/anton/1._University/0._Master_Project/Workspace/terrain_generation/terrains/mars1/terrain_only.usd",
    )

    #Obstacles
    obstacles = AssetBaseCfg(
        prim_path="/World/terrain/obstacles",
        spawn=sim_utils.UsdFileCfg(
            #usd_path="omniverse://127.0.0.1/Projects/P7 - Exam/Big rocks only.usd",
            usd_path="/home/anton/1._University/0._Master_Project/Workspace/terrain_generation/terrains/mars1/rocks_merged2.usd"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color_temperature=4500.0,
            intensity=100,
            enable_color_temperature=True,
            texture_file="/home/anton/Downloads/image(12).png",
            texture_format="latlong"),
    )

    sphere_light = AssetBaseCfg(
        prim_path="/World/SphereLight",
        spawn=sim_utils.SphereLightCfg(intensity=30000.0, radius=50, color_temperature=5500, enable_color_temperature=True),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -180.0, 80.0)),
    )

    robot: ArticulationCfg = AAU_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_(Drive|Steer|Boogie|Body)",
        filter_prim_paths_expr=["/World/terrain/obstacles/obstacles"]
        )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body",
        offset=RayCasterCfg.OffsetCfg(pos=[0.0, 0.0, 10.0]),
        attach_yaw_only=False,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.0,3.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/terrain/hidden_terrain"],
        max_distance=100.0,
        )


@configclass
class ActionsCfg:
    """ Action """
    actions: ActionTerm = mdp.AckermannActionCfg(
        asset_name="robot",
        wheelbase_length=0.849,
        middle_wheel_distance=0.894,
        rear_and_front_wheel_distance=0.77,
        wheel_radius=0.1,
        min_steering_radius=0.8,
        steering_joint_names=[".*Steer_Revolute"],
        drive_joint_names=[".*Drive_Continuous"],
    )



@configclass
class ObservationCfg:
    """ Observation configuration for the task.  """

    @configclass
    class PolicyCfg(ObsGroup):

        actions = ObsTerm(func=mdp.last_action)
        distance = ObsTerm(func=mdp.distance_to_target_euclidean,
                           params={
                               "command_name": "target_pose"
                               },
                               scale=0.11)
        heading = ObsTerm(func=mdp.angle_to_target_observation,
                          params={
                                "asset_cfg": SceneEntityCfg(name="robot"),
                                "command_name": "target_pose",
                          },
                          scale=1/math.pi)
        height_scan = ObsTerm(
            func=mdp.height_scan_rover,
            scale=1,
            params={"sensor_cfg": SceneEntityCfg(name="height_scanner")},)



        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    distance_to_target = RewTerm(
        func=mdp.distance_to_target_reward,
        weight=5.0,
        params={"command_name": "target_pose"},
    )
    reached_target = RewTerm(
        func=mdp.reached_target,
        weight=5.0,
        params={"command_name": "target_pose", "threshold": 0.18},
    )
    oscillation = RewTerm(
        func=mdp.oscillation_penalty,
        weight=-0.1,
        params={},
    )
    angle_to_target = RewTerm(
        func=mdp.angle_to_target_penalty,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg(name="robot"), "command_name": "target_pose"},
    )
    # heading_soft_contraint = RewTerm(
    #     func=mdp.heading_soft_contraint,
    #     weight=-0.5,
    #     params={"asset_cfg": SceneEntityCfg(name="robot")},
    # )
    # collision = RewTerm(
    #     func=mdp.collision_penalty,
    #     weight=-1.5,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
    # )

@configclass
class TerminationsCfg:
    """ Termination conditions for the task. """
    time_limit = DoneTerm(func=mdp.time_out, time_out=True)
    is_success = DoneTerm(
        func=mdp.is_success,
        params={"command_name": "target_pose", "threshold": 0.18},
        )
    far_from_target = DoneTerm(
        func=mdp.far_from_target,
          params={"command_name": "target_pose", "threshold": 20.0},
          )
    collision = DoneTerm(
        func=mdp.collision_with_obstacles,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
        )
# "mdp.illegal_contact
@configclass
class CommandsCfg:
    """ Command terms for the MDP. """
    target_pose = mdp.TerrainBasedPositionCommandCfg(
        asset_name="robot",
        rel_standing_envs=0.0,
        simple_heading=False,
        resampling_time_range=(150.0, 150.0),
        ranges=mdp.TerrainBasedPositionCommandCfg.Ranges(heading=(-math.pi, math.pi)),
        debug_vis=True,
    )
@configclass
class RandomizationCfg:
    """ Randomization configuration for the task. """
    reset_state = RandTerm(
        func=mdp.reset_root_state_rover,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            },
    )


@configclass
class RoverEnvCfg(RLTaskEnvCfg):
    """ Configuration for the rover environment."""

    # Create scene
    scene: RoverSceneCfg = RoverSceneCfg(num_envs=256, env_spacing=5, replicate_physics=False)

    # Setup PhysX Settings
    sim: SimCfg = SimCfg(physx=PhysxCfg(
        gpu_max_rigid_contact_count=8388608,
        gpu_max_rigid_patch_count=262144,
        gpu_found_lost_pairs_capacity=4096,
        gpu_found_lost_aggregate_pairs_capacity=1048576,
        gpu_total_aggregate_pairs_capacity=4096,
        gpu_max_soft_body_contacts=1048576,
        gpu_max_particle_contacts=1048576,
        gpu_heap_capacity=67108864,
        gpu_temp_buffer_capacity=16777216,
        gpu_max_num_partitions=8,
        gpu_collision_stack_size=2**28,
        friction_correlation_distance=0.025,
        friction_offset_threshold=0.04,
        bounce_threshold_velocity=2.0,
        )
    )


    # Basic Settings
    observations: ObservationCfg = ObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    randomization: RandomizationCfg = RandomizationCfg()


    # MDP Settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 200
        self.viewer.eye = (3.5, 3.5, 3.5)

        # update sensor periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt * self.decimation
