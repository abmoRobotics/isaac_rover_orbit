from gymnasium.spaces.box import Box
from omni.isaac.orbit.envs import RLTaskEnv
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

from rover_envs.learning.train.get_models import get_models
from rover_envs.utils.config import convert_skrl_cfg


def TD3_agent(experiment_cfg, observation_space: Box, action_space: Box, env: RLTaskEnv):

    # Define memory size
    memory_size = experiment_cfg["agent"]["batch_size"] * 2
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the models
    models = get_models("TD3", env, observation_space, action_space)

    # Agent cfg
    agent_cfg = TD3_DEFAULT_CONFIG.copy()
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    # Create the agent
    agent = TD3(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
    )
    return agent  # noqa R504
