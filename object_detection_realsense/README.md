# RGB-D object detection for robot perception

This folder contains a full pipeline of software components which enable the user to train an object detection system and deploy it over ROS to make it derive predictions in real-time on an Intel Realsense D435i depth cameras image output. All program components are intended to run within a Linux operating system.
All command instructions given here are intended to be executed in the base path of this projects file structure where this README file is also located. If this is not the case it is specifically stated so in the respective instructions.
All commands including the keyword python refer to an installation of python 3 and need to be replaced by your local command to call python 3.

# Generating Training Data

The blender-gen tool has been developed by the IGNC chair to generate artificial training data for the training of 2D as well as 3D object detectors like the ones utilized in this project.
It outputs its label information in two formats, the first is the standard COCO Dataset format where all label information is bundled into one JSON file. This file contains a dictionary with labeling information of which a detailed description can be found here: https://cocodataset.org/#format-data<br/>
To run the tool it requires a 3D model of the object of interest in an .obj file format. While it should technically work with ply files as well, this has not been tested within the scope of this project.
We acquired the 3D models of the robots, which can also be found in the "3d_models" folder, using a tool named Autodesk ReCap<sup>TM</sup> which generates 3D models from video data, but this could be replaced by any other 3D model in the .obj format.
The tool is however restricted to currently only take one 3D model as input and generating the training data for this particular model.

## Installation requirements

The main installation requirement for the data generation tool is a working blender installation. The instructions on how to install blender and setup the blender python environment can be found in the respective README.md file of the blender-gen project.

## Configuring Setup

Before the image generation process can be started a few important configuration options need to be adjusted in the file "./blender-gen/config.py". This file contains a multitude of parameters for the generation process. The most important ones are listed in the following:

-  To instruct the tool on the location of the 3D model which should be used in the training images change line 20:<br/>
```
self.model_path = "{LOCAL_PATH_TO_3D_FILE_LOCATION}"
```

-  To specify the amount of training images to be generated change line 96:<br/>
```
self.numberOfRenders = {NUMBER_OF_TRAINING_IMAGES}
```

-  When the model is not correctly in view of or always to close or to far away from the camera then the following parameters need to be adjusted. This can require some experimenting with the "--test" option of the blender tool which is shown below. These parameters are:

      - To adjust the range of distances that the camera can be located away from the robot:
      <br/>
      ```
      self.cam_rmin
      ```
      <br/>
      ```
      self.cam_rmax
      ```

      - To adjust the range of viewing angles from which the camera can view the robot:
      <br/>
      ```
      self.cam_incmin
      ```
      <br/>
      ```
      self.cam_incmax
      ```
      <br/>
      ```
      self.cam_azimin
      ```
      <br/>
      ```
      self.cam_azimax
      ```
      <br/>


## Running the program

Run the following command from within the blender-gen directory to start the rendering process:<br/><br/>
```
blender --background --python main.py
```
<br/><br/>
Run the following command from within the blender-gen directory to render one scene and then visualize a bounding box using OpenCV (useful for debugging and configuration):<br/><br/>
```
blender --background --python main.py --test
```

The resulting training images should then be located within the folder "./blender-gen/DATASET/object/images/". The corresonding annotation.json can be found in the base path of the blender-gen folder.

## Common problems
 - When running the program it should be noted that interrupting its execution also prevents the annontation.JSON file from being saved to disk, which means that all progress in data generation will be effectively lost.<br/>
 - It can also happen that the tool gets stuck in a loop. This often happens because the object is either to small or to large from the current camera perspective. Adjusting the camera parameters can solve this problem.


# Training MMDetection models

The MMDetection framework has been developed to simplify the design and training process for neural network architectures. It is based on the MMCV library which have both been developed by the OpenMMLab project. The underlying framework is based on PyTorch but no knowledge of this is required as MMDetection already handles most steps of the training process automatically.
This project utilized the MMDetection tool version 2.10.0 and the mmcv-full library version 1.2.7.

## Installation requirements

The MMDetection tool requires an installation of PyTorch and CUDA, the official installation instructions for this can be found here: https://pytorch.org/get-started/locally/. An installation process for the conda environment specifically for MMDetection tool is also outlined in https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md.

The following example can be used to install the same mmcv version as has been used in this project. Other versions might work but are not guaranteed to function properly.
When these basic modules are installed the mmcv-full library can be installed with:
<br/><br/>
```
pip install mmcv-full==1.2.7
```
<br/><br/>
Or when the library should not be build locally or a specific cuda and/or torch version are required the following can be used to download pre-build library images:
<br/><br/>
```
pip install mmcv-full=={desired_mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
<br/><br/>
Then install the remaining libraries with:
<br/><br/>
```
pip install -r requirements/build.txt
```
<br/>
```
pip install -v -e .
```
<br/><br/>
More detailed installation instructions can also be found in https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md.
## Configuring Setup

Configuring the MMDetection tools training procedure is mainly performed by editing the specific python scripts which contain configuration code for the respective network architecture and how and from where data should be loaded for the training process. What needs to be edited and why is shown in the following for YOLO and FasterRCNN as these are the architectures which were tested in this project. For other network architectures this configuration needs to be derived from their respective documentations.
Because the annontation.json file still contains all label information in one list, it needs to be separated into several files to represent the training, validation and test set respectively. This was achieved using the cocosplit script developed by Artur Karaźniewicz. Instructions on how to install its required libraries and how to use this script are given in the respective directories README. We utilized this script to split the annotation.json file into a train.json, val.json and test.json and will refer to these files under these names in the next section. These files need to be located in the same directory level as the object directory containing the training images so it is recommended to relocate both the training images and the annotation files into a new directory before starting the training process.
The following instructions are given based on the files "./mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco_pepper.py"  "./mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_pepper.py" and it is recommended to base your networks on these configuration files to replicate our results.
### YOLO

 - To indicate the location of the respective sets file locations modify the following lines:
   - For the training dataset in lines 101 & 102:<br/><br/>
    ```
    ann_file='{PATH_TO_TRAINING_DATA_FOLDER}/train.json'
    ```
    <br/>
    ```
    img_prefix='{PATH_TO_TRAINING_DATA_FOLDER}/'
    ```
    <br/>
   - For the validation dataset in lines 107 & 108:<br/><br/>
    ```
    ann_file='{PATH_TO_TRAINING_DATA_FOLDER}/val.json'
    ```
    <br/>
    ```
    img_prefix='{PATH_TO_TRAINING_DATA_FOLDER}/'
    ```
    <br/>
   - For the test dataset in lines 113 & 114:<br/><br/>
    ```
    ann_file='{PATH_TO_TRAINING_DATA_FOLDER}/test.json'
    ```
    <br/>
    ```
    img_prefix='{PATH_TO_TRAINING_DATA_FOLDER}/'
    ```
    <br/>
 - To set the number of epochs for which the training process should run edit the max_epochs parameter in line 127
### FasterRCNN

ros
 - To indicate the location of the respective sets file locations modify the following lines:
   - For the training dataset in lines 19 & 21:<br/><br/>
    ```
    img_prefix='{PATH_TO_TRAINING_DATA_FOLDER}/'
    ```
    <br/>
    ```
    ann_file='{PATH_TO_TRAINING_DATA_FOLDER}/train.json'
    ```
    <br/>
   - For the validation dataset in lines 23 & 25:<br/><br/>
    ```
    img_prefix='{PATH_TO_TRAINING_DATA_FOLDER}/'
    ```
    <br/>
    ```
    ann_file='{PATH_TO_TRAINING_DATA_FOLDER}/val.json'
    ```
    <br/>
   - For the test dataset in lines 27 & 29:<br/><br/>
    ```
    img_prefix='{PATH_TO_TRAINING_DATA_FOLDER}/'
    ```
    <br/>
    ```
    ann_file='{PATH_TO_TRAINING_DATA_FOLDER}/test.json'
    ```
    <br/>

## Running the program
### Training a model
Once all configuration files are configured accordingly the training can be started by running:
<br/><br/>
```
python ./mmdetection/tools/train.py ./mmdetection/configs/{MODEL_TYPE}/{MODEL_SPECIFICATION}_{ROBOT_NAME}.py
```
<br/>
So for example for a YOLO network this would be achieved with:
<br/><br/>
```
python ./mmdetection/tools/train.py ./mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco_{ROBOT_NAME}.py
```
<br/>

### Evaluating a model
This compares the predictions against the ground truth values in the test dataset and calculates a set of metrics to measure its performance:
<br/><br/>
```
python ./mmdetection/tools/test.py ./mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco_{ROBOT_NAME}.py  ./work_dirs/yolov3_d53_mstrain-608_273e_coco_{ROBOT_NAME}/latest.pth --eval bbox
```
<br/>

## Common problems
 - If the training process interrupts with an out-of-memory exception for the GPU memory a common fix is to reduce the settings for the workers_per_gpu and samples_per_gpu parameters in line 96-97 of the configuration file.


# ros-perception
The ros-perception tool is designed to embed the detection models outputted by the MMDetection framework within a ROS node and therefore provide the detection systems results through ROS messages.


## Installation requirements
As the tool is packaged within a GPU-accelerated docker container it is required to install the docker engine as well as the nvidia-container-toolkit. Instructions on how to achieve this and where to download the required software can be found under: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker<br/>
<br/>
To run the roscore or the realsense-ros node which reads and sends the Intel Realsense depth cameras images, a full ROS installation on the respective systems is required. Instructions for this can be found here: http://wiki.ros.org/ROS/Installation<br/>
The realsense-ros integration presents a framework to interface realsense cameras with ROS. In this project we utilized this tool to publish the color and depth images acquired by the Intel Realsense depth camera over the network to be analyzed by the detection node. It is available under: https://github.com/IntelRealSense/realsense-ros <br/>
The tool can be installed using the following command on a system with a working ROS installation:
<br/><br/>
```
sudo apt-get install ros-$ROS_DISTRO-realsense2-camera
```
<br/>

## Configuring Setup
The following changes have to be performed before building the docker container:

 - Move the trained MMDetection chckpoint .pth file and the corresponding config.py file into the models folder

 - In the file mmdetection_ros_node.py change line 52 and 53 to the correct names of the aforementioned model files.

 - In the file start_ros_node.sh replace lines 1-4 according to the following:
 <br/><br/>
 ```
 source /opt/ros/{LOCAL_ROS_VERSION}/setup.bash
 ```
 <br/>
 ```
 export ROS_MASTER_URI=http://{ROSCORE_IP_ADDRESS}:11311
 ```
 <br/>
 ```
 export ROS_IP={ROSCORE_IP_ADDRESS}
 ```
 <br/>
 ```
 export ROS_HOSTNAME={ROSCORE_IP_ADDRESS}
 ```
 <br/>

- The docker container can then be build with the following command:
<br/><br/>
```
sudo docker build -t ros-perception:latest .
```
<br/>

## Running the program
To allow the different ROS nodes in this setup to communicate with each other it is important to either turn off the firewall on all participating systems or to configure specific exceptions for these programs and the respective ports beforehand.
### (Optional) Start roscore:
If the roscore should not run on the same system which extracts the camera frames start a roscore in an empty command line window on the intended host machine with the following. Otherwise the realsense2_camera launch file will automatically serve as a roscore.
<br/>
<br/>
```
source /opt/ros/{local_ros_version}/setup.bash
```
<br/>
```
export ROS_MASTER_URI=http://{ROSCORE_IP_ADDRESS}:11311
```
<br/>
```
export ROS_IP={ROSCORE_IP_ADDRESS}
```
<br/>
```
export ROS_HOSTNAME={ROSCORE_IP_ADDRESS}
```
<br/>
```
roscore
```
<br/>

### Start realsense-ros:

To run the roslaunch program that provides the cameras output images over ROS execute the following code on the host machine to which the Intel Realsense Depth camera will be connected:
<br/>
<br/>
```
source /opt/ros/{local_ros_version}/setup.bash
```
<br/>
```
export ROS_MASTER_URI=http://{ROSCORE_IP_ADDRESS}:11311
```
<br/>
```
export ROS_IP={ROSCORE_IP_ADDRESS}
```
<br/>
```
export ROS_HOSTNAME={ROSCORE_IP_ADDRESS}
```
<br/>
```
roslaunch realsense2_camera rs_camera.launch ~intial-reset:=true
```
<br/>


### Run docker container:
```
sudo docker run -it --gpus all --net=host ros-perception:latest
```

This node then publishes three topics which can be utilized in further projects. These are a Detection2D message, an image message which shows the detected bounding box as well as a Marker message to visualize the detection in RVIZ. This marker message contains the important 3D location information of the robot and could be easily transmitted in other message formats if this would be required.

## Common problems

 - When the nodes cannot connect to each other it might be necessary to adjust the ROS topic names in the mmdetection_ros_node.py file.

 - Each time when building the docker container it stores a docker container image with a size of several gigabytes on your system. When this becomes a problem the docker command prune can be used to free up space from unneeded docker containers. A detailed documentation of this can be found here: https://docs.docker.com/config/pruning/
