# ZED 2 Set-up

This is a guide of how to set up detection with the Stereolabs ZED2 camera and the Nvidia Jetson.

## Hardware

Following hardware is needed:
* Stereolabs ZED2 Camera
* Nvidia Jetson AGX Xavier
* PC running Ubuntu 18

## Packages
### darknet_ros

darknet_ros is a ROS package developed for object detection. It offers a YOLOv3 implementation that can detect and classify objects in a video stream. YOLO is pre-trained, but can also be trained on a custom data set.

### zed-ros-wrapper
zed-ros-wrapper is a ROS package from Stereolabs that allows to use the ZED2 camera in ROS. It gives access to many camera information such as the images, the calibration and the depth.

### depth_yolo
depth_yolo is a ROS package developed to compute the 3D coordinates of an object in the real world. It subscribes to the object detections of darknes_ros and the depth map of zed-wrapper, computes the coordinates and publishes them as markers that can be visualized in RViz.

## Prepare Hardware

1. Set up Nvidia Jetson as described in the Quick Start User Guide (in the Jetson packet)
2. Connect ZED2 to Jetson
3. Follow steps in https://www.stereolabs.com/docs/installation/jetson/ to set up the Camera

## Integrate Camera in ROS

1. Install ROS Melodic and build catkin workspace: http://wiki.ros.org/melodic/Installation/Source
2. Copy the subfolder of this folder to your catkin \src
3. Navigate to your catkin workspace and run:
```
$ rosdep install --from-paths src --ignore-src -r -y
$ catkin build -DCMAKE_BUILD_TYPE=Release
$ source ./devel/setup.bash
```
4.  Further information can be found on: https://www.stereolabs.com/docs/ros
  * Don't git clone, since the wrapper is already in the \src folder (zed-ros-wrapper)
  * Use catkin build instead of catkin_make

## Prepare YOLO

This project used a YOLO implementation from https://github.com/leggedrobotics/darknet_ros. You can either use the version already included in this directory (darknet_ros) or clone the repo from git. Both methods are described in the next two sections.
### Use our Implementation
* This project includes the following darknet_ros implementation: https://github.com/leggedrobotics/darknet_ros
* The repo was already cloned and is located in the \darknet_ros folder
* Follow the instructions in the main repository to copy the weight from Google drive to */zed2-perception/darknet_ros/darknet_ros/yolo_network_config/weights/*
* There are some potential problems with OpenCV, Jetson and YOLO. Most of them are discussed here: https://github.com/leggedrobotics/darknet_ros/issues
  * In this project OpenCV 3.4.8 was used. You can download it here: https://opencv.org/releases/page/2/
    * This does not mean that other versions don't work. However, there may different issues to resolve
1. In your catkin \src folder go to edit \darknet_ros\darknet_ros\CMakeLists.txt. Change
```
find_package(OpenCV REQUIRED PATHS "/home/vis2020/build")
```
to point to the folder where OpenCV is located.
2. You may also have to change this is in /opt/ros/melodic/share/cv_bridge/cmake/cv_bridgeConfig.cmake. Edit:
```
set(_include_dirs <...>)
```
  * For more information: https://github.com/leggedrobotics/darknet_ros/issues/273
3. Run catkin build -DCMAKE_BUILD_TYPE=Release

### Set up from Scratch

1. Clone project from https://github.com/leggedrobotics/darknet_ros and follow steps in the README
2. Edit \src\darknet_ros\darknes_ros\config\ros.yaml and set camera_reading to the ZED topic

## Run
Once everything was build successfully, open 3 terminals and navigate to your catkin workspace
1. Start ZED2 wrapper:
```
$ source ./devel/setup.bash
$ roslaunch zed_wrapper zed2.launch
```
2. Start YOLO:
```
$ source ./devel/setup.bash
$ roslaunch darknet_ros darknet_ros.launch
```
3. Start depth_yolo
```
$ source ./devel/setup.bash
$ rosrun depth_yolo depth_yolo_node.py
```

## Visualize in RViz

1. When everything is running, run following commands in a new terminal to start RViz:
```
$ source ./devel/setup.bash
$ rosrun rviz rviz
```
2. In the bottom right corner of RViz click on "Add".
3. Go to the "By topic" tab.
4. Click on "MarkerArray" under /visualization_marker_array and then "OK". You should see the markers now.
5. You can add more topics in the same way. For example add the PointCloud by adding PointCloud2 of /zed2/zed_node/point_cloud/cloud_registered
