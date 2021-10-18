FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

LABEL maintainer="Kauki.Yoshiyama@gmail.com"

ENV OS ubuntu1804
ENV CUDA 10.0.0
ENV VERSION 7.0.0-1+cuda10.0

RUN apt-get update
RUN apt-get install -y \
  wget \
  less \
  zip

# nvidia machine learning repo
RUN wget https://developer.download.nvidia.com/compute/machine-learning/repos/${OS}/x86_64/nvidia-machine-learning-repo-${OS}_1.0.0-1_amd64.deb && \
  dpkg -i nvidia-machine-learning-repo-*.deb && \
  apt-get update

# Bind version
RUN  apt-get install -y \
  libnvinfer7=${VERSION} \
  libnvonnxparsers7=${VERSION} \
  libnvparsers7=${VERSION} \
  libnvinfer-plugin7=${VERSION} \
  libnvinfer-dev=${VERSION} \
  libnvonnxparsers-dev=${VERSION} \
  libnvparsers-dev=${VERSION} \
  libnvinfer-plugin-dev=${VERSION} \
  python-libnvinfer=${VERSION} \
  python3-libnvinfer=${VERSION}
RUN apt-mark hold \
  libnvinfer7 \
  libnvonnxparsers7 \
  libnvparsers7 \
  libnvinfer-plugin7 \
  libnvinfer-dev \
  libnvonnxparsers-dev \
  libnvparsers-dev \
  libnvinfer-plugin-dev \
  python-libnvinfer \
  python3-libnvinfer

# python3
RUN unlink /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python

# pycuda
RUN apt install -y \
  python3-pip
RUN python -m pip install \
  pycuda \
  ipython

RUN mkdir /workspace
WORKDIR /workspace



