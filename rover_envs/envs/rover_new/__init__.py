# import gymnasium as gym

# from . import rover_env_cfg

# gym.register(
#     id='RoverNew-v0',
#     entry_point='omni.isaac.orbit.envs:RLTaskEnv',
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": rover_env_cfg.RoverEnvCfg,
#     }
# )


# from .rover_camera_env import RoverEnvCamera
# from .rover_cfg import RoverEnvCfg
# from .rover_env import RoverEnv
from .rover_env_cfg import RoverEnvCfg

__all__ = ["RoverEnvCfg"]
# __all__ = ["RoverEnvCfg", "RoverEnv", "RoverEnvCamera"]
