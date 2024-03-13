from __future__ import annotations

from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import ActionTermCfg as ActionTerm  # noqa: F401
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm  # noqa: F401
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm  # noqa: F401
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm  # noqa: F401
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm  # noqa: F401
from omni.isaac.orbit.managers import SceneEntityCfg  # noqa: F401
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm  # noqa: F401
from omni.isaac.orbit.scene import InteractiveSceneCfg  # noqa: F401
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns  # noqa: F401
from omni.isaac.orbit.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.orbit.sim import PhysxCfg
from omni.isaac.orbit.sim import SimulationCfg as SimCfg
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.orbit.terrains import TerrainImporter, TerrainImporterCfg  # noqa: F401
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: F401

import rover_envs.envs.manipulation.mdp as mdp  # noqa: F401
# from rover_envs.assets.terrains.debug import DebugTerrainSceneCfg  # noqa: F401
# from rover_envs.assets.terrains.mars import MarsTerrainSceneCfg  # noqa: F401
from rover_envs.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter  # noqa: F401

# from rover_envs.envs.navigation.utils.terrains.terrain_importer import TerrainBasedPositionCommandCustom

##
# Scene Description
##


@configclass
class ManipulatorSceneCfg(InteractiveSceneCfg):
    """
    Rover Scene Configuration

    Note:
        Terrains can be changed by changing the parent class e.g.
        RoverSceneCfg(MarsTerrainSceneCfg) -> RoverSceneCfg(DebugTerrainSceneCfg)

    """

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color_temperature=4500.0,
            intensity=100,
            enable_color_temperature=True,
            texture_file="/home/anton/Downloads/image(12).png",
            texture_format="latlong",
        ),
    )

    sphere_light = AssetBaseCfg(
        prim_path="/World/SphereLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=30000.0, radius=50, color_temperature=5500, enable_color_temperature=True
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -180.0, 80.0)),
    )

    # Robot
    robot: ArticulationCfg = MISSING

    # End effector frame
    ee_frame: FrameTransformerCfg = MISSING

    # Target object
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )


@configclass
class ActionsCfg:
    """Action"""
    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_joint_pos: mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationCfg:
    """Observation configuration for the task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_pose = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.06}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.06, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.06, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=1e-3)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination conditions for the task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.base_height, params={
            "minimum_height": -0.05,
            "asset_cfg": SceneEntityCfg("object")
        }
    )


# "mdp.illegal_contact
@configclass
class CommandsCfg:
    """Command configuration for the task."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # Will be filled in the environment cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.7), pos_y=(0.3, 0.7), pos_z=(0.0, 0.0), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class RandomizationCfg:
    """Configuration for randomization of the task."""

    reset_all = RandTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = RandTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class ManipulatorEnvCfg(RLTaskEnvCfg):
    """Configuration for the rover environment."""

    # Create scene
    scene: ManipulatorSceneCfg = ManipulatorSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Setup PhysX Settings
    sim: SimCfg = SimCfg(
        physx=PhysxCfg(
            enable_stabilization=True,
            gpu_max_rigid_contact_count=8388608,
            gpu_max_rigid_patch_count=262144,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_found_lost_aggregate_pairs_capacity=2**25,  # 2**21,
            gpu_total_aggregate_pairs_capacity=2**21,   # 2**13,
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
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.sim.dt = 1 / 100.0  # 100 Hz
        self.decimation = 2
        self.episode_length_s = 5
        self.viewer.eye = (-6.0, -6.0, 3.5)
