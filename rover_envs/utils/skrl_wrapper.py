from typing import Any, Tuple

import torch
from omni.isaac.orbit.envs.rl_task_env import RLTaskEnv
from skrl.envs.torch.wrappers import IsaacOrbitWrapper


class IsaacOrbitWrapperFixed(IsaacOrbitWrapper):
    """ Wrapper for the Isaac Orbit environment.

    Note: The wrapper from SKRL broke in ORBIT version 0.2.0, due to RlTaskEnv.step() and RlTaskEnv.reset() being changed
    this is just a simple fix to make it work again.
    """
    def __init__(self, env: RLTaskEnv):
        super().__init__(env)
        self._env = env

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        self._obs_dict, reward, done, truncated, info = self._env.step(action)

        self._obs_dict["policy"] = torch.nan_to_num(self._obs_dict["policy"], nan=0.0, posinf=0.0, neginf=0.0)
        return self._obs_dict["policy"], reward.view(-1, 1), done.view(-1, 1), truncated.view(-1, 1), info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        self._obs_dict, info = self._env.reset()

        if self._reset_once:
            self._reset_once = False
            self._obs_dict, info = self._env.reset()

        self._obs_dict["policy"] = torch.nan_to_num(self._obs_dict["policy"], nan=0.0, posinf=0.0, neginf=0.0)
        return self._obs_dict["policy"].nan_to_num(nan=0.01), info
