import os

import gymnasium as gym

from .rover import rover_env_cfg

ORBIT_CUSTOM_ENVS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

ORBIT_CUSTOM_ENVS_DATA_DIR = os.path.join(ORBIT_CUSTOM_ENVS_EXT_DIR, "learning")



gym.register(
    id='Rover-v0',
    #entry_point='omni.isaac.orbit.envs:RLTaskEnv',
    entry_point='rover_envs.envs.rover:RoverEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rover_envs.envs.rover:RoverEnvCfg",
    }
)
# gym.register(
#     id="RoverCamera-v0",
#     entry_point="envs.rover:RoverEnvCamera",
#     kwargs={"cfg_entry_point": "envs.rover:RoverEnvCfg"},
# )
