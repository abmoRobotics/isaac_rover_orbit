# RoverLab
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
1. Install Isaac Sim 2023.1.1 through the [Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/download/).
2. Install ORBIT using the following steps:
```bash
git clone https://github.com/NVIDIA-Omniverse/orbit
cd Orbit

# create aliases
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2023.1.1"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# Create symbolic link
ln -s ${ISAACSIM_PATH} _isaac_sim

# Create Conda Env
./orbit.sh --conda orbit_env

# Activate Env
conda activate orbit_env

# Install dependencies
conda --install

```
3. Clone Repo

```bash
# Clone Repo
git clone https://github.com/abmoRobotics/isaac_rover_orbit
cd isaac_rover_orbit

# Install Repo (make sure conda is activated)
python -m pip install -e .[all]

# Run training script or evaluate pre-trained policy
cd examples/02_train/train.py
python train.py

cd examples/03_inference_pretrained/eval.py
python eval.py
```

# Contact
For other questions feel free to contact:
* Anton Bjørndahl Mortensen: antonbm2008@gmail.com
