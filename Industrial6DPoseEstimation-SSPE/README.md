#SSPE Model
This folder contains all code files utilized to train the SSPE component of this project. It is a clone of the IGNC SSPE implementation and therefore this README is partly taken from the original README of that repository. As the training process of this network takes place within a docker container it is required to install docker beforehand.
Also the training data and/or model weights we produced in our project can be found in shared google folder provided in the general README of this project. The training data should be placed in the new_DB folder and the models should be placed in the backup folder before building the docker container. When a model has been trained within a docker container the resulting weights file needs to be copied out of the docker container into the regular file structure.

## How to run
### Training a new model


Step 1: `cd Industrial6DPoseEstimation-SSPE`

Step 2: Download dataset from link provided in general README of this project and put the required training data folder into the new_DB folder

Step 3: Download pre-trained model weights and put inside `backup` folder

Step 4: Build and run docker as below

`sudo docker build --network=host -t sspe .`

`sudo docker run --gpus all -it --rm --network=host sspe`

Step 5: Run `python3 blender_train.py mixed_pepper.data yolo-pose.cfg backup/init.weights`


### Testing model by a single image

`python draw_image.py --p "path_to_weight_file" --i "path_to_input_image"`

Example: `python draw_image.py --p model.weights --i input_image.png`

### Testing model by a video

`python draw_image.py --p "path_to_weight_file" --i "path_to_input_video"`

Example: `python draw_video.py --p model.weights --i input_video.mp4 `

### Testing model in jupyter notebook

This jupyter notebook is taken from the original SSPE implementation by Tekin et al.
It can be used to load a model file and test its performance on its defined test set by changing the variables datacfg, cfgfile, weightfile to point to your respective file locations.

`jupyter notebook valid.ipynb `
