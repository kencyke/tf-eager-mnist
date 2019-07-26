FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

SHELL [ "/bin/bash", "-c" ]
    
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        curl \
        dstat \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG _PY_VERSION=3.6
ARG PYTHON=python${_PY_VERSION}

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        ${PYTHON} \
        ${PYTHON}-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py | ${PYTHON}

ARG PIP=pip${_PY_SUFFIX}
ARG TENSORFLOW_PACKAGE=tensorflow-gpu==1.13.1

RUN ${PIP} --no-cache-dir install --upgrade \
        pip \
        setuptools \
    && ${PIP} --no-cache-dir install \
        ${TENSORFLOW_PACKAGE}

# for gpu full trace
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

# https://github.com/NVIDIA/nvidia-docker/issues/775
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

ENV PYTHONPATH ./