import gym




gym.register(
    id="Rover-v0",
    entry_point="custom_envs.rover:RoverEnv",
    kwargs={"cfg_entry_point": "custom_envs.rover:RoverEnvCfg"},
)