import torch
from omni.isaac.orbit.envs.rl_task_env import RLTaskEnv

from .rover_env_cfg import RoverEnvCfg


class RoverEnv(RLTaskEnv):
    """ Rover environment.

    Note:
        This is a placeholder class for the rover environment. That is, this class is not yet implemented."""
    def __init__(self, cfg: RoverEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
