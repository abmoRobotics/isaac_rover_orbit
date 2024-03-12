# RL Agent (Mars Rover) for Isaac Gym
This repository is currently in heavy development, please refer to the [DEV BRANCH](https://github.com/abmoRobotics/isaac_rover_orbit/tree/dev), to get the latest version. 
## Introduction
This project implements a Reinforcement Learning (RL) agent for autonomous mapless navigation in complex environments. The environment is simulated using Isaac Sim and implemented using the [ORBIT](https://isaac-orbit.github.io/orbit/) framework.

## Implementation Overview
[WIP]

# Installation
In order to ease the setup of this suite, we use docker to install Isaac Sim, ORBIT, and this framework. The following documents the process and requirements of doing this.
## Requirements
### Hardware
- GPU: Any RTX GPU with at least 8 GB VRAM (Tested on NVIDIA RTX 3090 and NVIDIA RTX A6000)
- CPU: Intel i5/i7 or equivalent
- RAM: 32GB or more

### Software
- Operating System: Ubuntu 20.04 or 22.04
- Packages: Docker and Nvidia Container Toolkit

## Building the docker image
1. Clone and build docker:
```bash
# Clone Repo
git clone https://github.com/abmoRobotics/isaac_rover_orbit
cd isaac_rover_orbit

# Build and start docker
cd docker
./run.sh
docker exec -it orbit bash

```

2. Train an agent
Once inside the docker container you can train and agent by using the following command
```bash
/workspace/orbit/orbit.sh -p train.py --task="Rover-v0" --num_envs=256
```

## Installing natively
1. Install ORBIT using the following steps:
```bash
git clone https://github.com/abmoRobotics/Orbit-fork
cd Orbit

# create aliases
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# Create symbolic link
ln -s ${ISAACSIM_PATH} _isaac_sim

# Create Conda Env
./orbit.sh --conda orbit_env

# Activate Env
conda activate orbit_env

# Install dependencies
./orbit.sh --install --extra
```
2. Clone Repo

```bash
# Clone Repo
git clone https://github.com/abmoRobotics/isaac_rover_orbit
cd isaac_rover_orbit

# Run training script or evaluate pre-trained policy
python train.py
python eval.py
```

# Contact
For other questions feel free to contact:
* Anton Bjørndahl Mortensen: antonbm2008@gmail.com
