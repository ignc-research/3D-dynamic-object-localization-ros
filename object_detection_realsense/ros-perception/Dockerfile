FROM ubuntu:focal

# Packages
#
# Ubuntu 20.04: https://github.com/tianon/docker-brew-ubuntu-core/blob/74249faf47098bef2cedad89696bfd1ed521e019/focal/Dockerfile
# ROS Noetic (base) Docker: https://github.com/osrf/docker_images/blob/df19ab7d5993d3b78a908362cdcd1479a8e78b35/ros/noetic/ubuntu/focal/ros-base/Dockerfile
# MMDetection: https://mmdetection.readthedocs.io/en/latest/get_started.html#installation
# PyTorch: https://pytorch.org/get-started/locally/
# CvBridge: https://github.com/ros-perception/vision_opencv.git
# ROS Vision Messages: https://github.com/ros-perception/vision_msgs.git

ARG PYTORCH="1.7.0"
ARG CUDA="11.0"
ARG CUDNN="8"
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0"
ENV NVIDIA_DRIVER_CAPABILITIES compute

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-pip libopencv-dev python3-opencv
RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV
RUN pip3 install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

# Install MMDetection
#RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip3 install -r requirements/build.txt
RUN pip3 install --no-cache-dir -e .


### ubuntu 20.04 with ros noetic

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO noetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-core=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

# setup entrypoint?
WORKDIR /
COPY ./ros_entrypoint.sh /
COPY ./start_ros_node.sh /
#RUN chmod +x ./ros_entrypoint.sh
#ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
RUN ./ros_entrypoint.sh
#RUN /bin/bash -c "bash /ros_entrypoint.sh"

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

# create and build catkin_ws
SHELL ["/bin/bash", "-c", "-l"]
WORKDIR /home/catkin_ws
RUN mkdir src
RUN source "/opt/ros/$ROS_DISTRO/setup.bash" && catkin_make
WORKDIR /home/catkin_ws/src
RUN git clone -b noetic https://github.com/ros-perception/vision_opencv.git && git clone -b noetic-devel https://github.com/ros-perception/vision_msgs.git
WORKDIR /home/catkin_ws/
RUN source /home/catkin_ws/devel/setup.bash && catkin_make

## copy our mmdetection neural network
WORKDIR /home
COPY ./mmdetection_ros_node.py .
COPY ./model/ ./model/
ENTRYPOINT ["/start_ros_node.sh"]
