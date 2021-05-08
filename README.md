# 3D-robot-localization-ros

Our project's objective was to develop a real-time object detection system integrated in ROS to detect and classify three different robot models, namely Pepper, TIAGo, and the KUKA youBot, and humans and locate them in relation to a camera mounted on a robot.

All the modules, implementations and resulting products of this project are included in this Repo.

## Weight files

Since the size of the training data as well as the weight files for the different models exceed GitHub's limits, they are stored on Google Drive and can be accessed via this link: https://drive.google.com/drive/folders/1tYig62OAzJDJrKIc3dWvHv2Ggf9e-bMg?usp=sharing
The files have to be downloaded and moved to their original folders. 
The Drive folder contains:
* **\MMDetection**
  * \models: Contains the .pth weight files for MMDetection models as well as their respective configuration files
  * \training_data: Contains training datasets to train MMDetection networks and respective annotation.json files
* **\original_videos**: Contains the videos of the robots that were used for 3D model generation.
* **\SSPE**
  * \training_data: Contains training datasets to train SSPE networks and the respective label files
  * \models: Contains .weight SSPE model files. The respective config file is provided within the SSPE folder of this repository
* **\Zed2_model_weights**: Contains YOLO weight files. The contents have to be moved to */zed2-perception/darknet_ros/darknet_ros/yolo_network_config/weights/*

## Overview

This is a short overview of the directories and contents included in this project. More information about the contents and a detailed description on how to setup and utilize the respective components is given in the specific subdirectories.

### 3d_models

Contains the generated 3D models for Pepper, TIAGo, and the KUKA youBot.

### Industrial6DPoseEstimation-SSPE

Contains all necessary files and data for training a 6D Pose Estimation SSPE.

### object_detection_realsense

An object detection system developed to be used together with the Intel Realsense D435i stereo camera in ROS. Also contains a full pipeline from the generation of training images to the deployment of a model within a ROS environment.

### zed2-perception

An object detection system developed to be used together with the Stereolabs ZED 2 stereo camera in ROS on the Nvidia Jetson workstation.

