# 3D-robot-localization-ros

Our project's objective was to develop a real-time object detection system integrated in ROS to detect and classify three different robot models, namely Pepper, TIAGo, and the KUKA youBot, and humans and locate them in relation to a camera mounted on a robot.

All the modules, implementations and resulting products of this project are included in this Repo.

## Weight files

Since the size of the weight files for the different models exceed GitHub's limits, they are stored on Google Drive and can be accessed via this link: https://drive.google.com/drive/folders/1tYig62OAzJDJrKIc3dWvHv2Ggf9e-bMg?usp=sharing
The files have to be downloaded and moved to their original folders. 
The Drive folder contains:
* \MMDetection
  * \models:
  * \training_data:
* \original_videos: Contains the videos of the robots that were used 3D model generation.
* \SSPE
  * \training_data:
  * \models: 
* \Zed2_model_weights: Contains YOLO weight. The contents have to be moved to /zed2-perception/darknet_ros/darknet_ros/yolo_network_config/weights/

## Overview

This is a short overview of the directories and contents included in this project. More detailed information about the contents and their usage are given in the specific subdirectories.

### 3d_models

Contains the generated 3D models for Pepper, TIAGo, and the KUKA youBot.

### Industrial6DPoseEstimation-SSPE

Contains all necessary files and data for training a 6D Pose Estimation SSPE.

### object_detection_realsense

An object detection system developed to be used together with the Intel Realsense D435i stereo camera in ROS.

### zed2-perception

An object detection system developed to be used together with the Stereolabs ZED 2 stereo camera in ROS on the Nvidia Jetson workstation.

