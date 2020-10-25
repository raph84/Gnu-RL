 FROM python:3.8.6-buster

 WORKDIR /root

 RUN pip install cloudml-hypertune

 # Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg



# Java required by Gym-Eplus / EnergyPlus
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-add-repository 'deb http://security.debian.org/debian-security stretch/updates main'  && \
    apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get remove -y software-properties-common




# Git clone Gnu-RL in the Volume
RUN git clone https://github.com/zhangzhizza/Gym-Eplus.git
RUN git clone --single-branch --branch gnu-rl-svc https://github.com/raph84/Gnu-RL.git

# EnergyPlus runtime is a dependancy of Gym-Eplus.
RUN apt-get install -y wget && \
       cd /Gym-Eplus/ && \
#       wget -q https://github.com/NREL/EnergyPlus/archive/v8.6.0.tar.gz -O - | tar -xz
       wget -q https://github.com/NREL/EnergyPlus/releases/download/v8.6.0/EnergyPlus-8.6.0-198c6a3cff-Linux-x86_64.sh && \
       chmod +x EnergyPlus-8.6.0-198c6a3cff-Linux-x86_64.sh

# Setup the Python environment
RUN cd / && \ 
        pip3 install virtualenv && \
        virtualenv virt_env --python=python3 && \
        /bin/bash -c "source /virt_env/bin/activate && \
                        pip3 install gym && \
                        pip3 install -e /Gym-Eplus/ && \
                        cd /Gnu-RL && \
                        pip3 install -r requirements.txt && \
                        cd /Gnu-RL/eplus_env && \
                        cp -R * /Gym-Eplus/eplus_env/ && \
                        pip3 install matplotlib && \
                        pip3 install --upgrade debugpy && \
                        pip3 install jupyterlab"

