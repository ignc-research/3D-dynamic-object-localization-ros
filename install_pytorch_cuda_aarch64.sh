#!/usr/bin/env bash
 
# script to build cuda enabled pytorch wheel for python 3.7

set -xe
ls -l

git clone --recursive --branch 1.7 http://github.com/pytorch/pytorch
cd /pytorch

python3.7 -m pip install -r requirements.txt

export PYTORCH_BUILD_VERSION=1.7.0
export PYTORCH_BUILD_NUMBER=1
export BUILD_TEST=0
export USE_NCCL=0 
export USE_QNNPACK=0 
export USE_PYTORCH_QNNPACK=0
export USE_CUDA=1

python3.7 setup.py build
python3.7 setup.py install
python3.7 setup.py bdist_wheel
