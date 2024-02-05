ARG ISAAC_SIM_VERSION
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

# Set default RUN shell to bash
SHELL ["/bin/bash", "-c"]

ARG ISAAC_SIM_PATH

# Set environment variables
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV ORBIT_PATH=/workspace/orbit
ENV ROVER_LAB_PATH=/workspace/orbit_rover

# Install dependencies and remove cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    ncurses-term && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Clone orbit
RUN git clone https://github.com/NVIDIA-Omniverse/orbit ${ORBIT_PATH}

# Create symlink to Isaac Sim
RUN ln -sf ${ISAAC_SIM_PATH} ${ORBIT_PATH}/_isaac_sim

# Install orbit
RUN ${ORBIT_PATH}/orbit.sh --install --extra

RUN echo "alias orbit=${ORBIT_PATH}/orbit.sh" >> ${HOME}/.bashrc && \
    echo "alias python=${ISAAC_SIM_PATH}/python.sh" >> ${HOME}/.bashrc && \
    echo "alias python3=${ISAAC_SIM_PATH}/python.sh" >> ${HOME}/.bashrc && \
    echo "alias pip='${ISAAC_SIM_PATH}/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias pip3='${ISAAC_SIM_PATH}/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias tensorboard='${ISAAC_SIM_PATH}/python.sh ${ISAAC_SIM_PATH}/tensorboard'" >> ${HOME}/.bashrc && \
    echo "export TZ=$(date +%Z)" >> ${HOME}/.bashrc

# Clone rover_lab
RUN git clone https://github.com/abmoRobotics/isaac_rover_orbit ${ROVER_LAB_PATH}
# Checkout to dev branch and install dependencies
RUN /bin/bash -c "cd ${ROVER_LAB_PATH} && git checkout dev && ${ISAAC_SIM_PATH}/python.sh -m pip install -e ."
#RUN cd ${ROVER_LAB_PATH} && python -m install -e .
#RUN pip install -e ${ROVER_LAB_PATH}

# Set working directory
WORKDIR ${ROVER_LAB_PATH}
