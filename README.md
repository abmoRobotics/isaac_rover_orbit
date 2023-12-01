# RL Agent (Mars Rover) for Isaac Gym

## Introduction
This project implements a Reinforcement Learning (RL) agent designed to operate within the Isaac Gym environment. Our RL agent leverages the high-performance physics simulation capabilities of Isaac Gym to train on various robotic tasks.

## Implementation Overview
[Isaac Gym](https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_gym_tutorials/index.html) is NVIDIA's advanced robotics simulation platform. This project utilizes Isaac Gym to simulate complex environments where our RL agent learns through interaction.

## Requirements
### Hardware
- GPU: NVIDIA RTX 3090 or better
- CPU: Intel i5/i7 or equivalent
- RAM: 32GB or more

### Software
- Python 3.10+
- PyTorch 2.1+
- CUDA 11.0+

# Installation
## Requirements
### Hardware
- GPU: NVIDIA RTX 3090 or better
- CPU: Intel i5/i7 or equivalent
- RAM: 32GB or more

### Software
- Operating System: Ubuntu 20.04 or 22.04
- Python Version: Python 3.10
- Isaac Sim 2023.1.0-hotfix.1
- Conda

1. Install Orbit using the following steps:


```bash
git clone --branch v0.1.0 --single-branch https://github.com/NVIDIA-Omniverse/Orbit.git
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
./orbit --install
./orbit --extra
```
2. Clone Repo


# Contact
For other questions feel free to contact:
* Anton Bjørndahl Mortensen: antonbm2008@gmail.com
