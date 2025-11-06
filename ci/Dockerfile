ARG BASE_IMAGE=docker.io/mambaorg/micromamba:latest
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ARG WORKDIR_PATH=/workspace
ENV WORKDIR=${WORKDIR_PATH}
WORKDIR ${WORKDIR}

USER root
RUN apt-get update -qq && apt-get install -qq -y git && \
    apt-get clean

USER $MAMBA_USER
ARG PY_VERSION=3.13
RUN micromamba create -n dace-env python=$PY_VERSION -c conda-forge -c nvidia && \
    micromamba clean -a -y

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=dace-env
RUN python -m pip install --no-cache-dir pytest
