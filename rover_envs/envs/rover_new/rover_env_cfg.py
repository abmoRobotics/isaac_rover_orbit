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
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

import rover_envs.envs.rover_new.mdp as mdp
from rover_envs.robots.config.aau_rover import AAU_ROVER_CFG

##
# Scene Description
##

@configclass
class RoverSceneCfg(InteractiveSceneCfg):


    # Ground Terrain

    # ground_terrain = AssetBaseCfg(
    #     prim_path="/World/terrain",
    #     spawn=sim_utils.UsdFileCfg(
    #         #usd_path="omniverse://127.0.0.1/Projects/P7 - Exam/Big rocks.usd",
    #         usd_path="/home/anton/1._University/0._Master_Project/Workspace/terrain_generation/terrains/mars1/terrain_only.usd",
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(10.0, 10.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
    # )

    # #Obstacles
    # obstacles = AssetBaseCfg(
    #     prim_path="/World/rock",
    #     spawn=sim_utils.UsdFileCfg(
    #         #usd_path="omniverse://127.0.0.1/Projects/P7 - Exam/Big rocks only.usd",
    #         usd_path="/home/anton/1._University/0._Master_Project/Workspace/terrain_generation/terrains/mars1/rocks_merged.usd"
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(10.0, 10.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
    # )

    terrain = AssetBaseCfg(
        prim_path="/World/terrain",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://127.0.0.1/Projects/terrain_3.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(10.0, 10.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=2000.0),
    )
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/terrain",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    #     debug_vis=False,
    # )

    robot: ArticulationCfg = AAU_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_(Drive|Steer|Boogie|Body)", filter_prim_paths_expr=["/World/terrain/combined/rocks_merged/Mesh_766"])
    #contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_(Drive|Steer|Boogie|Body)")


    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Body",
    #     offset=[0.0, 0.0, 0.0],
    #     attach_yaw_only=False,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.0,3.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/terrain"],
    #     )



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


mdp.illegal_contact
@configclass
class ObservationCfg:
    """ Observation configuration for the task.  """

    @configclass
    class PolicyCfg(ObsGroup):

        actions = ObsTerm(func=mdp.last_action)
        distance = ObsTerm(func=mdp.distance_to_target_euclidean,
                           params={
                               "asset_cfg": SceneEntityCfg(name="robot"),
                               "command_name": "target_pose"
                               },
                               scale=0.11)
        heading = ObsTerm(func=mdp.angle_to_target_observation,
                          params={
                                "asset_cfg": SceneEntityCfg(name="robot"),
                                "command_name": "target_pose",
                          },
                          scale=1/math.pi)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     scale=0.33,
        #     params={"sensor_cfg": SceneEntityCfg(name="height_scanner")},)



        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:

    distance_to_target = RewTerm(
        func=mdp.distance_to_target,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg(name="robot"), "command_name": "target_pose"},
    )
    reached_target = RewTerm(
        func=mdp.reached_target,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg(name="robot"), "command_name": "target_pose", "threshold": 0.18},
    )
    oscillation = RewTerm(
        func=mdp.oscillation_penalty,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg(name="robot")},
    )
    angle_to_target = RewTerm(
        func=mdp.angle_to_target_penalty,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg(name="robot"), "command_name": "target_pose"},
    )
    heading_soft_contraint = RewTerm(
        func=mdp.heading_soft_contraint,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg(name="robot")},
    )
    # collision = RewTerm(
    #     func=mdp.collision_penalty,
    #     weight=-1.5,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
    # )

@configclass
class TerminationsCfg:
    """ Termination conditions for the task. """
    time_limit = DoneTerm(func=mdp.time_out, time_out=True)
    # is_success = DoneTerm(
    #     func=mdp.reached_target,
    #     params={"asset_cfg": SceneEntityCfg(name="robot"), "command_name": "target_pose", "threshold": 0.18},
    #     )
    # far_from_target = DoneTerm(
    #     func=mdp.far_from_target,
    #       params={"asset_cfg": SceneEntityCfg(name="robot"), "command_name": "target_pose", "threshold": 15.0},
    #       )
    collision = DoneTerm(
        func=mdp.collision_penalty,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
        )
mdp.illegal_contact
@configclass
class CommandsCfg:
    """ Command terms for the MDP. """

    target_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="Body",
        resampling_time_range=(150.0, 150.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-9.0, 9.0), pos_y=(-9.0, 9.0), pos_z=(3.0, 3.0), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        )
    )

@configclass
class RandomizationCfg:
    """ Randomization configuration for the task. """
    # pose_range: dict[str, tuple[float, float]],
    # reset
    reset_orientation = RandTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 3.14)},
            "velocity_range": (0.0, 0.0),
            },
        )
#         params={"asset_cfg": SceneEntityCfg(name="robot"),
#                 "rotation_range": (0.0, 0.0)},


@configclass
class RoverEnvCfg(RLTaskEnvCfg):
    """ """

    # Create scene
    scene: RoverSceneCfg = RoverSceneCfg(num_envs=256, env_spacing=2.5, replicate_physics=False)

    # Basic Settings
    observations: ObservationCfg = ObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    # randomization: RandomizationCfg = RandomizationCfg()

    # TODO: add command generator

    # MDP Settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10
        self.viewer.eye = (3.5, 3.5, 3.5)

        # Simulation Settings
        self.sim.dt = 1.0 / 20.0 # 20 Hz
        self.sim.disable_contact_processing = True

        # update sensor periods
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt * self.decimation
