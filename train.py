import argparse
import os

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


from datetime import datetime

import gymnasium as gym
from omni.isaac.orbit.utils.dict import print_dict
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml
from omni.isaac.orbit_tasks.utils import parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.skrl import (
    SkrlSequentialLogTrainer, SkrlVecEnvWrapper)
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed
from skrl.utils.model_instantiators import (deterministic_model,
                                            gaussian_model, shared_model)

import rover_envs.envs
from config import convert_skrl_cfg, parse_skrl_cfg
from rover_envs.envs.rover.learning.models import (
    DeterministicNeuralNetwork, DeterministicNeuralNetworkSimple,
    GaussianNeuralNetwork, GaussianNeuralNetworkSimple)

#import omni.isaac.contrib_envs  # noqa: F401
#import omni.isaac.orbit_envs  # noqa: F401


def log_setup(experiment_cfg, env_cfg):
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

def main():
    args_cli_seed = args_cli.seed
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    experiment_cfg = parse_skrl_cfg(args_cli.task)

    log_dir = log_setup(experiment_cfg, env_cfg)
    #print(args_cli)
    # create isaac environment
    from gymnasium import envs
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless, viewport=args_cli.video)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env  = SkrlVecEnvWrapper(env)

    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])
    print(env.action_space)
    models = {}
    #ray = env._env.cfg.observations.policy.enable_ray_height
    print(env.observation_space)
    print(env.action_space)
    env._env
    ray = False
    # if not ray:
    #     models["policy"] = gaussian_model(
    #         observation_space=env.observation_space["policy"],
    #         action_space=env.action_space,
    #         device=env.device,
    #         **convert_skrl_cfg(experiment_cfg["models"]["policy"])
    #     )
    #     models["value"] = deterministic_model(
    #         observation_space=env.observation_space["policy"],
    #         action_space=env.action_space,
    #         device=env.device,
    #         **convert_skrl_cfg(experiment_cfg["models"]["value"])
    #     )
    # else:
    #     models["policy"] = GaussianNeuralNetwork(
    #         observation_space=env.observation_space["policy"],
    #         action_space=env.action_space,
    #         device=env.device)
    #     models["value"] = DeterministicNeuralNetwork(
    #         observation_space=env.observation_space,
    #         action_space=env.action_space,
    #         device=env.device)
    models["policy"] = GaussianNeuralNetworkSimple(
            observation_space=env.observation_space["policy"],
            action_space=env.action_space,
            device=env.device)
    models["value"] = DeterministicNeuralNetworkSimple(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)

    memory_size = experiment_cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    # experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    # agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    # agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})

    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
    #agent.load("logs/skrl/rover/Nov15_13-24-49/checkpoints/best_agent.pt")
    #agent.load("best_agents/Nov26_17-18-32/checkpoints/best_agent.pt")
    trainer_cfg = experiment_cfg["trainer"]
    print(trainer_cfg)

    trainer = SkrlSequentialLogTrainer(cfg=trainer_cfg, agents=agent, env=env)
    trainer.train()
    #trainer.eval()

    env.close()
    simulation_app.close()



if __name__ == "__main__":
    main()
