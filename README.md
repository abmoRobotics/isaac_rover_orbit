# RL Agent (Mars Rover) for Isaac Gym
[WIP] The rover assets are not available yet.
## Introduction
This project implements a Reinforcement Learning (RL) agent for autonomous mapless navigation in complex environments. The environment is simulated using Isaac Sim and implemented using the [ORBIT](https://isaac-orbit.github.io/orbit/) framework.

## Implementation Overview
[WIP]

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
./orbit --install
./orbit --extra
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
