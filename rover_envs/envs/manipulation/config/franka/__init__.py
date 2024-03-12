import gymnasium as gym

from . import agents
from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg

gym.register(
    id="FrankaCubeLift-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": FrankaCubeLiftEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
