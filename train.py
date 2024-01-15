import argparse
import math
import os
from datetime import datetime

import gymnasium as gym
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
# app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.kit"
# launch the simulator

simulation_app = SimulationApp(config, experience=app_experience)


from omni.isaac.orbit.envs import RLTaskEnv  # noqa: E402
from omni.isaac.orbit.utils.dict import print_dict  # noqa: E402
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml  # noqa: E402
from omni.isaac.orbit_tasks.utils import parse_env_cfg  # noqa: E402
from omni.isaac.orbit_tasks.utils.wrappers.skrl import SkrlSequentialLogTrainer  # noqa: E402
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG  # noqa: E402
from skrl.memories.torch import RandomMemory  # noqa: E402
from skrl.utils import set_seed  # noqa: E402

import rover_envs.envs  # noqa: F401, E402
from rover_envs.envs.rover.learning.models import DeterministicNeuralNetwork, GaussianNeuralNetwork  # noqa: E402
from rover_envs.utils.config import convert_skrl_cfg, parse_skrl_cfg  # noqa: E402
from rover_envs.utils.skrl_wrapper import IsaacOrbitWrapperFixed  # noqa: E402


def log_setup(experiment_cfg, env_cfg):
    """
    Setup the logging for the experiment.

    Note:
        Copied from the ORBIT framework.
    """
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs
    log_dir = datetime.now().strftime("%b%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'

    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir

    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)
    return log_dir


def get_models(env: RLTaskEnv, observation_space, action_space):
    """
    Placeholder function for getting the models.

    Note:
        This function will be further improved in the future, by reading the model config from the experiment config.

    Args:
        env (RLTaskEnv): The environment.
        observation_space (gym.spaces.Space): The observation space of the environment.
        action_space (gym.spaces.Space): The action space of the environment.

    Returns:
        dict: A dictionary containing the models.
    """

    models = {}
    encoder_input_size = env.observation_manager.group_obs_term_dim["policy"][-1][0]

    mlp_input_size = 4

    models["policy"] = GaussianNeuralNetwork(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    models["value"] = DeterministicNeuralNetwork(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    return models


def video_record(env: RLTaskEnv, log_dir: str, video: bool, video_length: int, video_interval: int) -> RLTaskEnv:
    """
    Function to check and setup video recording.

    Note:
        Copied from the ORBIT framework.

    Args:
        env (RLTaskEnv): The environment.
        log_dir (str): The log directory.
        video (bool): Whether or not to record videos.
        video_length (int): The length of the video (in steps).
        video_interval (int): The interval between video recordings (in steps).

    Returns:
        RLTaskEnv: The environment.
    """

    if video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % video_interval == 0,
            "video_length": video_length,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)

    return gym.wrappers.RecordVideo(env, **video_kwargs)


def main():
    args_cli_seed = args_cli.seed
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    experiment_cfg = parse_skrl_cfg(args_cli.task)

    log_dir = log_setup(experiment_cfg, env_cfg)

    # Create the environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless, viewport=args_cli.video)
    # Check if video recording is enabled
    env = video_record(env, log_dir, args_cli.video, args_cli.video_length, args_cli.video_interval)
    # Wrap the environment
    env: RLTaskEnv = IsaacOrbitWrapperFixed(env)
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    # Get the observation and action spaces
    num_obs = env.observation_manager.group_obs_dim["policy"][0]
    num_actions = env.action_manager.action_term_dim[0]
    observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))

    # Memory for the agent
    memory_size = experiment_cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the standard agent config and update it with the experiment config
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    # experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    # agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    # agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})

    # Get the models
    models = get_models(env, observation_space, action_space)

    # Create the agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
    )
    # agent.load("logs/skrl/rover/Nov15_13-24-49/checkpoints/best_agent.pt")
    # agent.load("best_agents/Nov26_17-18-32/checkpoints/best_agent.pt")
    trainer_cfg = experiment_cfg["trainer"]
    print(trainer_cfg)

    trainer = SkrlSequentialLogTrainer(cfg=trainer_cfg, agents=agent, env=env)
    trainer.train()
    # trainer.eval()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
