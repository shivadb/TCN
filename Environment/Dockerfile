FROM nvcr.io/nvidia/tensorrt:20.12-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN set -x \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        redis-tools \
        netcat \
        net-tools \
        openssh-client \
        openconnect \
        openvpn \
        sudo \
        build-essential \
        git \
        libboost-all-dev\
        file \
        ncdu \
        vim \
        curl \
        unzip \
        screen \
        ffmpeg \
        byobu \
        wget \
        tmux \
        python-numpy  \
        rsync


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
RUN cd torch2trt && python3 setup.py install --plugins
