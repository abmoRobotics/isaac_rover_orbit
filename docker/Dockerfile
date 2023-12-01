FROM nvcr.io/nvidia/isaac-sim:2023.1.0-hotfix.1

# Set the working directory to /home/ubuntu
WORKDIR /home/ubuntu

# Set the environment variables
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies git cmake
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ncurses-term && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

### Install Orbit

# Clone the repository
RUN git clone https://github.com/abmoRobotics/Orbit-fork

# Install Orbit
RUN cd Orbit-fork && ln -s "/isaac-sim" _isaac_sim

RUN cd Orbit-fork && /bin/bash -c "./orbit.sh --install"



### Clone isaac_rover_orbit
RUN git clone https://github.com/abmoRobotics/isaac_rover_orbit

# RUN cd isaac_rover_orbit && ln -s "/isaac-sim" _isaac_sim
RUN echo 'alias python="/home/ubuntu/Orbit-fork/orbit.sh -p"' >> /home/ubuntu/.bashrc

RUN echo 'alias python="/home/ubuntu/Orbit-fork/orbit.sh -p"' >> ~/.bashrc

ENTRYPOINT [ "/bin/bash" ]
