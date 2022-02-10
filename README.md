# ros-tracking-robot-human-3d 

This repo contains the docker container deployment for detection and localizing humand and robots in 3D. 
To install the software, clone the repository and follow the below instructions: 
## Building your own image: 

## to build your docker image: 

	sudo docker build -t hichdh/ws-thesis-final:1.0.0 .
	
### to run your docker container:

	sudo docker run -it --runtime nvidia --gpus all --net=host --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix:rw hichdh/ws-thesis-final:1.0.0 /bin/bash

### to delete all docker related files: 

	sudo docker system prune --all --volumes --force

## Downloading the image from docker hub  

	sudo docker pull hichdh/ws-thesis-final:1.0.0
	
## To start tracking robots and humans run this command inside the container    
	
	roslaunch tracking_humans_and_robots_3d zed_people_robot_tracking.launch 

## First execution: 

An AI model for the ZED 2 camera needs to be downloaded first. to permit a smooth download, you have to disable the robot tracking in the zed_people_robot_tracking launch file. 

Use this command to enable/disable robot tracking: 
nano tracking_humans_and_robots_3d/launch/zed_people_robot_tracking.launch

You can also disable the ros visualisation by changing the corresponding parameter in the same launch file. 


## Second execution: 

After the model is downloaded you can stop the execution, change back the robot tracking parameter in the launch file to true and run the above roslaunch command again. 

## Disable human tracking service

To disable/enable the human tracking service provided by the ZED2, you have to open the file located in "ros-tracking-robot-human-3d/zed_catkin_ws/src/zed-ros-wrapper/zed_wrapper/launch/zed2.launch" and change the obj_hum_det parameter to true/false. 

from the ros-tracking-robot-human-3d directory you can run this command: nano ros-tracking-robot-human-3d/zed_catkin_ws/src/zed-ros-wrapper/zed_wrapper/launch/zed2.launch

## Common issues: 
## X11 screen error: run this command to disable the display control management:
	
	sudo xhost +local:root

# In case rviz can't access display then please check the assigned display number with the help of the following command: 
	
	xauth info 

# and then run:

	export DISPLAY=:"display_number"
