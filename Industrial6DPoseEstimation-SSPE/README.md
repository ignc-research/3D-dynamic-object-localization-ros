# How to run
### Case 1: Training by Linh's data set only

Step 1: Clone this repo into your laptop/PC

Step 2: `cd Industrial6DPoseEstimation`

Step 3: Download dataset from link (https://files.slack.com/files-pri/TEL7B5P29-F01D610RAH3/download/doube-augmented.zip) and put inside this repo with all images inside JPEGImages and all labels inside labels folder

Step 4: Download pre-trained model weights and put inside `backup` folder

Step 5: Build and run docker as below

`sudo docker build --network=host -t sspe .`

`sudo docker run --gpus all -it --rm --network=host sspe`

Step 6: Run `python/python3 train.py 3dbox.data yolo-pose.cfg backup/init.weights`


### Case 2: Training by combined dataset

Step 1: Clone this repo into your laptop/PC

Step 2: `cd Industrial6DPoseEstimation`

Step 3: Download dataset from link (https://chairignc.slack.com/files/UEKS45J0L/F01DBB7QX3P/new-combined-data.zip) and put inside this repo with all images inside JPEGImages and all labels inside labels folder

Step 4: Download pre-trained model weights and put inside `backup` folder

Step 5: Build and run docker as below

`sudo docker build --network=host -t sspe .`

`sudo docker run --gpus all -it --rm --network=host sspe`

Step 6: Run `python/python3 combined_train.py combined_3dbox.data yolo-pose.cfg backup/init.weights`

### Case 3: Testing model by a single image

`python draw_image.py --p "path_to_weight_file" --i "path_to_input_image"`

Example: `python draw_image.py --p model.weights --i input_image.png`

### Case 4: Testing model by a video

`python draw_image.py --p "path_to_weight_file" --i "path_to_input_video"`

Example: `python draw_video.py --p model.weights --i input_video.mp4 `

### Case 5: Training by Blender's dataset

Step 1: Clone this repo into your laptop/PC

Step 2: `cd Industrial6DPoseEstimation`

Step 3: Download dataset from link (https://drive.google.com/file/d/1F_bZR9kxy7iDm1Cvg54xvJ-YIPL8MIbB/view?usp=sharing) and put inside this repo with all images inside JPEGImages and all labels inside labels folder

Step 4: Download pre-trained model weights and put inside `backup` folder

Step 5: Build and run docker as below

`sudo docker build --network=host -t sspe .`

`sudo docker run --gpus all -it --rm --network=host sspe`

Step 6: Run `python/python3 blender_train.py blender_3dbox.data yolo-pose.cfg backup/init.weights`


