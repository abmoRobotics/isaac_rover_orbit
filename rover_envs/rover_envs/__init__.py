import os

import gym

ORBIT_CUSTOM_ENVS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

ORBIT_CUSTOM_ENVS_DATA_DIR = os.path.join(ORBIT_CUSTOM_ENVS_EXT_DIR, "data")

gym.register(
    id="Rover-v0",
    entry_point="rover_envs.rover:RoverEnv",
    kwargs={"cfg_entry_point": "rover_envs.rover:RoverEnvCfg"},
)

gym.register(
    id="RoverCamera-v0",
    entry_point="rover_envs.rover:RoverEnvCamera",
    kwargs={"cfg_entry_point": "rover_envs.rover:RoverEnvCfg"},
)
