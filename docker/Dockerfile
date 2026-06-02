FROM nvidia/cuda:12.1.0-devel-ubuntu20.04 AS dev

ARG ROS_VERSION=noetic
ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

################# ROS Install #################
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl gnupg lsb-release git build-essential nano unzip wget \
 && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
      > /etc/apt/sources.list.d/ros-latest.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_VERSION}-desktop-full \
      python3-pip python3-rosdep python3-catkin-tools \
 && rosdep init && rosdep update \
 && echo "source /opt/ros/${ROS_VERSION}/setup.bash" >> /root/.bashrc \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

#################  python depen #################
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir "typing-extensions<4.9.0" \
    && pip3 install --no-cache-dir "numpy==1.23.5" \
    && pip3 install --no-cache-dir \
         torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir \
         opencv-python scikit-image "pandas==1.5.3" open3d pcl \
    && pip3 install --no-cache-dir --force-reinstall --no-deps "numpy==1.23.5"

################# Depen #################
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpcl-dev libeigen3-dev \
      ros-noetic-grid-map ros-noetic-grid-map-core ros-noetic-tf2-msgs \
      ros-noetic-tf2-sensor-msgs \
      python3-tk \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

################# workspace #################
RUN mkdir -p /root/gsat_ws/src
WORKDIR /root/gsat_ws

RUN /bin/bash -c "source /opt/ros/${ROS_VERSION}/setup.bash && \
                  catkin config --extend /opt/ros/${ROS_VERSION}"

#--------[Add workspace src to PYTHONPATH so 'from GSAT_Traversability.gsat...' absolute imports resolve]--------#
ENV PYTHONPATH=/root/gsat_ws/src:${PYTHONPATH}
RUN echo "export PYTHONPATH=/root/gsat_ws/src:\${PYTHONPATH}" >> /root/.bashrc

WORKDIR /root/gsat_ws