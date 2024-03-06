from __future__ import annotations

from omni.isaac.orbit.utils import configclass

from rover_envs.assets.aau_rover_simple import AAU_ROVER_SIMPLE_CFG
from rover_envs.envs.navigation.rover_env_cfg import RoverEnvCfg


@configclass
class AAURoverEnvCfg(RoverEnvCfg):
    """Configuration for the AAU rover environment."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = AAU_ROVER_SIMPLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")