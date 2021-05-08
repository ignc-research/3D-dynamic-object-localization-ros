FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx
RUN pip install opencv-python scipy argparse urllib3

COPY . /workspace/SSPE_TUBerlin
WORKDIR /workspace/SSPE_TUBerlin

#ENTRYPOINT ["python3", "train.py"]

# In one terminal
# sudo docker build --network=host -t sspe .
# sudo docker run -it --rm --network=host sspe


