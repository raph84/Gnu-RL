FROM raph84/pytorch-slim-buster:latest AS base
RUN apt-get update && \
    apt-get install -y git && \
    cd / && git clone --single-branch --branch gnu-rl-svc https://github.com/raph84/Gnu-RL.git && \
    cd / && git clone https://github.com/zhangzhizza/Gym-Eplus.git


FROM raph84/pytorch-slim-buster:latest AS gnu-rl

COPY --from=base /Gnu-RL /Gnu-RL

# RUN pip3 install virtualenv && \
#    virtualenv virt_env --python=python3 

# Setup the Python environment
RUN pip3 install --default-timeout=100 -r /Gnu-RL/requirements.txt

EXPOSE 8080/tcp



FROM gnu-rl AS gnu-rl-api
RUN pip install Flask gunicorn

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
WORKDIR /Gnu-RL/agent
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 PPO_MPC_API:app


FROM gnu-rl AS dev

COPY --from=base /Gym-Eplus /Gym-Eplus

RUN   apt-get install -y openssh-server && \
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
    # mkdir needed to install openjdk on slim image
    mkdir -p /usr/share/man/man1 && \ 
    apt-get install -y openjdk-11-jdk


# EnergyPlus runtime is a dependancy of Gym-Eplus.
RUN apt-get install -y wget && \
       cd /Gym-Eplus/ && \
#       wget -q https://github.com/NREL/EnergyPlus/archive/v8.6.0.tar.gz -O - | tar -xz
       wget -q https://github.com/NREL/EnergyPlus/releases/download/v8.6.0/EnergyPlus-8.6.0-198c6a3cff-Linux-x86_64.sh && \
       chmod +x EnergyPlus-8.6.0-198c6a3cff-Linux-x86_64.sh && \
       echo "y\r" | ./EnergyPlus-8.6.0-198c6a3cff-Linux-x86_64.sh


# Setup the Python environment
RUN /bin/bash -c "source /virt_env/bin/activate && \
                    pip3 install gym && \
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