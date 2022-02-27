# ROS integrated application to localize dynamic objects in 3D using deep learning approaches

The objective of this bachelor thesis was to develop a real-time object detection system integrated in ROS to detect 
and classify three different robot models, namely Pepper, TIAGo, and the KUKA youBot, and humans and 
locate them in relation to a camera mounted on a robot. All the modules, implementations and resulting products of this thesis  are included in this Repo.

This repo contains the workspace to run the docker container deployment for detecting robots in 
2D using the YOLO v3 algorithm. The 2D detection results are processed with the depth data to localize the robots in 3D.
The human detection and tracking in 3D is produced by the pre-implemented human tracking service 
of the ZED2 camera.  

To run the ROS dynamic object localization softwarre, clone the repository, download the trained model weights, the mmcv and the pytorch wheels from the following links: 

# Trained model weights: 
Download the trained model weights stored in the following tubit cloud link: 
- https://tubcloud.tu-berlin.de/s/wTFCLeg6P9dFXKq

and store it in zed_catkin_ws/src/mmdetection_ros/scripts/mmdetector.py
# MMCV wheel (built from source): 
Download the python wheel file from the following tubit cloud link:  
- https://tubcloud.tu-berlin.de/s/GAdrobXJ6zJ93Hx

# PYTORCH wheel (built from source):
Download the python wheel file from the following tubit cloud link: 
- https://tubcloud.tu-berlin.de/s/9TcyHfii3JWt3q4

# Execution and Configuration
During the first execution with the human tracking service enabled, the ZED SDK downloads an AI model 
for the ZED 2 camera and configures it at runtime. 
The ROS workspace is structured withing the following roslaunch file: 
tracking_humans_and_robots_3d/launch/zed_people_robot_tracking.launch
to enable/disable the 2D and 3D visualizations in RVIZ, start the ZED2 camera, start the 
detector node use this command: 

	nano zed_catkin_ws/src/tracking_humans_and_robots_3d/launch/zed_people_robot_tracking.launch

# Disable/Enable Human Tracking 

To disable/enable the human tracking service provided by the ZED2, you have to open the file located in 
"/zed_catkin_ws/src/zed-ros-wrapper/zed_wrapper/launch/zed2.launch" and 
change the obj_hum_det parameter to true/false. 


# Human detection and tracking performance
Run this command to manipulate the human tracking and detection performances: 

	nano ros-tracking-robot-human-3d/zed_catkin_ws/src/zed-ros-wrapper/zed_wrapper/params/zed2.yaml 
	

This list of models are available to choose from: 
- 0 : MULTI_CLASS_BOX 
- 1 : MULTI_CLASS_BOX_ACCURATE 
- 2 : HUMAN_BODY_FAST
- 3 : HUMAN_BODY_ACCURATE 


# Building your own image: 

## to build your docker image: 

	sudo docker build -t custom-image:1.0.0 .
## to run your docker container:

	sudo docker run -it --runtime nvidia --gpus all --net=host --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix:rw custom-image:1.0.0 /bin/bash

## to delete all docker related files: 

	sudo docker system prune --all --volumes --force
	
## To start tracking robots and humans run this command inside the container    
	
	roslaunch tracking_humans_and_robots_3d zed_people_robot_tracking.launch 


## Common issues: 
## X11 screen error: 
### run this command to disable the display control management:
	sudo xhost +
### In case rviz can't access the display then check the assigned display number with this command: 
	xauth info 
### and then run:
	export DISPLAY=:"display_number"
