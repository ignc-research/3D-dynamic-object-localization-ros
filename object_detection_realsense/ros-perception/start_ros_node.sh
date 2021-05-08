#!/bin/bash

source /home/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.7:11311
export ROS_IP=192.168.0.7
export ROS_HOSTNAME=192.168.0.7
./mmdetection_ros_node.py
