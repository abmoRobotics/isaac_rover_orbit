from __future__ import annotations

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.envs.mdp.commands.position_command import TerrainBasedPositionCommand  # noqa: F401
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
from omni.isaac.orbit.sim import PhysxCfg
from omni.isaac.orbit.sim import SimulationCfg as SimCfg
from omni.isaac.orbit.terrains import TerrainImporter, TerrainImporterCfg  # noqa: F401
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: F401

import rover_envs.envs.navigation.mdp as mdp  # noqa: F401
from rover_envs.assets.terrains.debug import DebugTerrainSceneCfg  # noqa: F401
from rover_envs.assets.terrains.mars import MarsTerrainSceneCfg  # noqa: F401
from rover_envs.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter  # noqa: F401

# from rover_envs.envs.navigation.utils.terrains.terrain_importer import TerrainBasedPositionCommandCustom

##
# Scene Description
##


@configclass
class RoverSceneCfg(DebugTerrainSceneCfg):
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

    robot: ArticulationCfg = None


@configclass
class ActionsCfg:
    """Action"""
    pass


@configclass
class ObservationCfg:
    """Observation configuration for the task."""

    @configclass
    class PolicyCfg(ObsGroup):
        pass

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    """Termination conditions for the task."""
    pass


# "mdp.illegal_contact
@configclass
class CommandsCfg:
    pass


@configclass
class RandomizationCfg:
    pass


@configclass
class RoverEnvCfg(RLTaskEnvCfg):
    """Configuration for the rover environment."""

    # Create scene
    scene: RoverSceneCfg = RoverSceneCfg(
        num_envs=256, env_spacing=4.0, replicate_physics=False)

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
        self.sim.dt = 1 / 100.0
        self.decimation = 4
        self.episode_length_s = 150
        self.viewer.eye = (-6.0, -6.0, 3.5)
