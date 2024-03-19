import torch
from omni.isaac.orbit.envs.rl_task_env import RLTaskEnv
from omni.isaac.orbit.envs.rl_task_env_cfg import RLTaskEnvCfg


class RLTaskEnv(RLTaskEnv):
    """ Rover environment.
    """

    def __init__(self, cfg: RLTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _reset_idx(self, idx: torch.Tensor):
        """Reset the environment at the given indices.

        Note:
            This function inherits from :meth:`omni.isaac.orbit.envs.rl_task_env.RLTaskEnv._reset_idx`.
            This is done because SKRL requires the "episode" key in the extras dict to be present in order to log.
        Args:
            idx (torch.Tensor): Indices of the environments to reset.
        """
        super()._reset_idx(idx)

        # Done this way because SKRL requires the "episode" key in the extras dict to be present in order to log.
        self.extras["episode"] = self.extras["log"]
