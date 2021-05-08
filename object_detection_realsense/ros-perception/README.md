# ros-perception
## Setup
you need docker and nvidia-docker
move your mmdetection checkpoint .pth-file and config.py-file into the model/ folder
change topic names in faster_rcnn_node.py


## Build docker image:
```
sudo docker build -t ros-perception:latest .
```

## Run docker container:
```
sudo docker run -it --gpus all --net=host ros-perception:latest
```
