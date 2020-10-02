FROM ubuntu:18.04 AS build

# Update Python
RUN apt update && apt install -y git && \ 
      apt -y upgrade && \
      apt install -y python3-pip

# Java required by Gym-Eplus / EnergyPlus
RUN apt-get install -y vim && \
      apt-get install -y openjdk-8-jdk

# Git clone Gnu-RL in the Volume
RUN cd / && git clone https://github.com/zhangzhizza/Gym-Eplus.git
RUN cd / && git clone https://github.com/INFERLab/Gnu-RL.git

# EnergyPlus runtime is a dependancy of Gym-Eplus.
RUN apt-get install -y wget && \
       cd /Gym-Eplus/eplus_env/envs/EnergyPlus-8-6-0/ && \
       wget -q https://github.com/NREL/EnergyPlus/archive/v8.7.0.tar.gz -O - | tar -xz

# Setup the Python environment
RUN cd /Gym-Eplus && \ 
        pip3 install virtualenv && \
        virtualenv virt_env --python=python3 && \
        /bin/bash -c "source virt_env/bin/activate" && \
        pip3 install gym && \
        pip3 install -e /Gym-Eplus/ && \
        cd /Gnu-RL && \
        pip3 install -r requirements.txt && \
        cd /Gnu-RL/eplus_env && \
        cp -R * /Gym-Eplus/eplus_env/ && \
        pip3 install matplotlib


RUN pip3 install jupyterlab

# Expose Jupyter port to access it from a browser on the Docker host
EXPOSE 8888/tcp