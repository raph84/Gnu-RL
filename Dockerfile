FROM ubuntu:18.04 AS build

# Update Python
RUN apt update && apt install -y git && \ 
      apt -y upgrade && \
      apt install -y python3-pip && \
      apt-get install -y openssh-server && \
      apt-get install -y curl && \
      curl -sL https://deb.nodesource.com/setup_10.x | bash - && \
      apt-get install -y nodejs

RUN mkdir /var/run/sshd
RUN echo 'root:password' | chpasswd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

CMD ["/usr/sbin/sshd", "-D"]


# Java required by Gym-Eplus / EnergyPlus
RUN apt-get install -y vim && \
      apt-get install -y openjdk-8-jdk

# Git clone Gnu-RL in the Volume
RUN cd / && git clone https://github.com/zhangzhizza/Gym-Eplus.git
RUN cd / && git clone --single-branch --branch gnu-rl-svc https://github.com/raph84/Gnu-RL.git && \
       echo "Done.."


# EnergyPlus runtime is a dependancy of Gym-Eplus.
RUN apt-get install -y wget && \
       cd /Gym-Eplus/ && \
#       wget -q https://github.com/NREL/EnergyPlus/archive/v8.6.0.tar.gz -O - | tar -xz
       wget -q https://github.com/NREL/EnergyPlus/releases/download/v8.6.0/EnergyPlus-8.6.0-198c6a3cff-Linux-x86_64.sh && \
       chmod +x EnergyPlus-8.6.0-198c6a3cff-Linux-x86_64.sh && \
       echo "y\r" | ./EnergyPlus-8.6.0-198c6a3cff-Linux-x86_64.sh && \
       echo "Done."

# Setup the Python environment
RUN cd / && \ 
        pip3 install virtualenv && \
        virtualenv virt_env --python=python3 && \
        /bin/bash -c "source /virt_env/bin/activate && \
                        pip3 install gym && \
                        cd /Gnu-RL && \
                        pip3 install -r requirements.txt && \
                        cd /Gnu-RL/eplus_env && \
                        cp -R * /Gym-Eplus/eplus_env/ && \
                        pip3 install -e /Gym-Eplus/ && \
                        pip3 install matplotlib && \
                        pip3 install --upgrade debugpy && \
                        pip3 install jupyterlab"

# Expose Jupyter port to access it from a browser on the Docker host
EXPOSE 8888/tcp

# Expose Python debug port
EXPOSE 5678

# Expose SSH
EXPOSE 22