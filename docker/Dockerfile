FROM ros:noetic-ros-core-focal

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 ROS_DISTRO=noetic
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN sed -i -e 's|http://|https://|g' \
      -e 's|security.ubuntu.com|archive.ubuntu.com|g' /etc/apt/sources.list \
    && rm /etc/apt/sources.list.d/* \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main' \
      > /etc/apt/sources.list.d/ros.list \
    && curl -fsSL https://nrvcorp.github.io/camera/apt/ubuntu20.04/nrv-archive-keyring.gpg \
      -o /usr/share/keyrings/nrv-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/nrv-archive-keyring.gpg] https://nrvcorp.github.io/camera/apt/ubuntu20.04/ ./' \
      > /etc/apt/sources.list.d/nrv.list

ARG NRV_SDK_VERSION=1.2.2-1~ubuntu20.04
ARG NRV_ROS_VERSION=1.2.0-0focal
ARG NRV_MSGS_VERSION=2.0.1-0focal
ARG NRV_CODECS_VERSION=1.0.0-0focal
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake python3-numpy python3-opencv \
      libgl1-mesa-dri libgl1-mesa-glx \
      ros-noetic-nodelet ros-noetic-camera-info-manager \
      ros-noetic-rqt-image-view ros-noetic-message-generation \
      libdelta-sdk="${NRV_SDK_VERSION}" \
      ros-noetic-delta-driver="${NRV_ROS_VERSION}" \
      ros-noetic-delta-tools="${NRV_ROS_VERSION}" \
      ros-noetic-event-camera-msgs="${NRV_MSGS_VERSION}" \
      ros-noetic-event-camera-codecs="${NRV_CODECS_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

COPY ros_ws/src /opt/nrv_demo_ws/src
RUN source /opt/ros/noetic/setup.bash \
    && cd /opt/nrv_demo_ws \
    && catkin_make -DCMAKE_BUILD_TYPE=Release -DCATKIN_ENABLE_TESTING=OFF -j2

RUN apt-get update && apt-get install -y --no-install-recommends python3-pyqt5 fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["roslaunch", "nrv_demo", "gui.launch"]
