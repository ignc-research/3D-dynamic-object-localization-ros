ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.5.0
#ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.6-py3
FROM ${BASE_IMAGE}

#ARG ROS_PKG=ros_base
ARG ROS_PKG=desktop-full
ENV ROS_DISTRO=melodic
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=2
ENV OPENCV_VERSION=4.5.3

ENV OPENBLAS_CORETYPE=ARMV8
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential curl wget gnupg2 \
    lsb-release ca-certificates software-properties-common \ 
    && add-apt-repository universe && apt-get update \
    && rm -rf /var/lib/apt/lists/*


# **********************
# Installing python 3.7*
# **********************
ARG PYTHON=python3.7
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y ${PYTHON} \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && ${PYTHON} get-pip.py \ 
    && rm get-pip.py 


# ************************************************
# Setting python version to 3.7 as system python *
# and upgrading pip and setuptools               *
# ************************************************
RUN ln -sf /usr/bin/${PYTHON} /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/pip /usr/local/bin/pip3 \ 
    && pip3 --no-cache-dir install --upgrade pip setuptools \
    && ln -s $(which ${PYTHON}) /usr/local/bin/python \
    && apt-get install -y nano ${PYTHON}-dev \
    && python3 -m pip install cython \ 
    && python3 -m pip install numpy  \
    && pip3 install scikit-build setuptools wheel matplotlib cython pyopengl funcy\ 
    && apt-get clean all \ 
    && rm -rf /var/lib/apt/lists/*


# ***************************************
# install zed sdk and setup the ZED SDK *
# ***************************************
COPY ./ZED_SDK_Tegra_JP45_v3.5.5.run . 
RUN apt-get update -y && apt-get install --no-install-recommends \
        lsb-release wget less udev sudo apt-transport-https \
        libqt5xml5 libxmu-dev libxi-dev build-essential cmake -y \
    && chmod +x ZED_SDK_Tegra_JP45_v3.5.5.run \ 
    && echo "# R32 (release), REVISION: 5.0, GCID: 25531747, BOARD: t186ref, EABI: aarch64, DATE: Fri Jan 15 23:21:05 UTC 2021" | tee /etc/nv_tegra_release \
    && /ZED_SDK_Tegra_JP45_v3.5.5.run -- silent 

# ************************************
# installing:
    # - cuda enabled pytorch version:  1.7
    # - mmcv version:  1.3.14
        # - mmdetection version:  2.16.0 
# ************************************

# Installing cuda enbaled pytorch  
WORKDIR /pytorch_wheel
COPY pytorch_wheel/torch-1.7.0a0-cp37-cp37m-linux_aarch64.whl .
RUN pip3 install torch-1.7.0a0-cp37-cp37m-linux_aarch64.whl \ 
    && pip3 install torchvision \ 
    && apt-get clean all \ 
    && rm -rf /var/lib/apt/lists/*

# Install cuda enabled mmcv (mmcv-full version)   
WORKDIR /mmcv_wheel
COPY mmcv_wheels/mmcv_full-1.3.14-cp37-cp37m-linux_aarch64.whl /mmcv_wheel
RUN pip3 install mmcv_full-1.3.14-cp37-cp37m-linux_aarch64.whl \ 
    && pip3 install pycocotools terminaltables mmdet \
    && pip3 install rospkg \
    && apt-get clean all \ 
    && rm -rf /var/lib/apt/lists/*


# Done: install ros melodic with python 2
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - 
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# install ROS packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
		ros-melodic-${ROS_PKG} \
		ros-melodic-image-transport \
		ros-melodic-vision-msgs \
          python-rosdep \
          python-rosinstall \
          python-rosinstall-generator \
          python-wstool \
    && cd ${ROS_ROOT} \
    && rosdep init \
    && rosdep update \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /workspace
RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> /root/.bashrc

COPY zed_catkin_ws /workspace/zed_catkin_ws
WORKDIR /workspace/zed_catkin_ws

# build catkin workspace  
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash ; catkin_make -DCMAKE_BUILD_TYPE=Release'

# source ros workspace
RUN echo 'source /workspace/zed_catkin_ws/devel/setup.bash' >> /root/.bashrc

# setup entrypoint
COPY ros_entrypoint.sh /workspace/zed_catkin_ws/ros_entrypoint.sh 
ENTRYPOINT [ "./ros_entrypoint.sh" ]  
