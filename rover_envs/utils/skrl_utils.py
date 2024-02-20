import copy

import torch
import tqdm
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG


class SkrlSequentialLogTrainer(Trainer):
    """Sequential trainer with logging of episode information.

    This trainer inherits from the :class:`skrl.trainers.base_class.Trainer` class. It is used to
    train agents in a sequential manner (i.e., one after the other in each interaction with the
    environment). It is most suitable for on-policy RL agents such as PPO, A2C, etc.

    It modifies the :class:`skrl.trainers.torch.sequential.SequentialTrainer` class with the following
    differences:

    * It also log episode information to the agent's logger.
    * It does not close the environment at the end of the training.

    Reference:
        https://skrl.readthedocs.io/en/latest/modules/skrl.trainers.base_class.html
    """

    def __init__(
        self,
        env: Wrapper,
        agents: Agent | list[Agent],
        agents_scope: list[int] | None = None,
        cfg: dict | None = None,
    ):
        """Initializes the trainer.

        Args:
            env: Environment to train on.
            agents: Agents to train.
            agents_scope: Number of environments for each agent to
                train on. Defaults to None.
            cfg: Configuration dictionary. Defaults to None.
        """
        # update the config
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        # store agents scope
        agents_scope = agents_scope if agents_scope is not None else []
        # initialize the base class
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)
        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def train(self):
        """Train the agents sequentially.

        This method executes the training loop for the agents. It performs the following steps:

        * Pre-interaction: Perform any pre-interaction operations.
        * Compute actions: Compute the actions for the agents.
        * Step the environments: Step the environments with the computed actions.
        * Record the environments' transitions: Record the transitions from the environments.
        * Log custom environment data: Log custom environment data.
        * Post-interaction: Perform any post-interaction operations.
        * Reset the environments: Reset the environments if they are terminated or truncated.

        """
        # init agent
        self.agents.init(trainer_cfg=self.cfg)
        self.agents.set_running_mode("train")
        # reset env
        states, infos = self.env.reset()
        # training loop
        for timestep in tqdm.tqdm(range(self.timesteps), disable=self.disable_progressbar):
            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)
            # note: here we do not call render scene since it is done in the env.step() method
            # record the environments' transitions
            with torch.no_grad():
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )
            # log custom environment data
            if "episode" in infos:
                for k, v in infos["episode"].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        self.agents.track_data(f"EpisodeInfo / {k}", v.item())
            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)
            # reset the environments
            # note: here we do not call reset scene since it is done in the env.step() method
            # update states
            states.copy_(next_states)
