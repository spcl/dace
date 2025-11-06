ARG BASE_IMAGE=docker.io/nvidia/cuda
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ARG WORKDIR_PATH=/workspace
ENV WORKDIR=${WORKDIR_PATH}
WORKDIR ${WORKDIR}

RUN apt-get update -qq && apt-get install -qq -y \
    python3 \
    ca-certificates \
    git \
    build-essential && \
    apt-get clean

RUN python3 -m pip install --no-cache-dir pytest
