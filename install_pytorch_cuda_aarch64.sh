#!/usr/bin/env bash

# 
# script to build cuda enabled pytorch wheel for python 3.7
#

set -xe
ls -l

#cd pytorch
#git checkout 
#git submodule sync
#git submodule update --init --recursive --jobs 0

git clone --recursive --branch 1.7 http://github.com/pytorch/pytorch
cd /pytorch

python3.7 -m pip install -r requirements.txt

export PYTORCH_BUILD_VERSION=1.7.0
export PYTORCH_BUILD_NUMBER=1

#export MAX_JOBS=2 this makes it so slow 
export BUILD_TEST=0
#export USE_FFMPEG=1

# recommended by dusty nv
export USE_NCCL=0 # this is taking so long maybe without this and not required 
export USE_QNNPACK=0 # this is deprecated I think  
export USE_PYTORCH_QNNPACK=0 # this is deprecated I think 
export USE_CUDA=1

# try these two params (dusty nv)
# export USE_DISTRIBUTED=0
# export TORCH_CUYDA_ARCH_LIST="5.3;6.2;7.2"

# try this : disable use nnpack because deprecated and a lot of warnings 
# export USE_NNPACK=0

python3.7 setup.py build
python3.7 setup.py install
python3.7 setup.py bdist_wheel
