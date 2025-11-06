FROM mambaorg/micromamba:cuda${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ARG WORKDIR_PATH=/workspace
ENV WORKDIR=${WORKDIR_PATH}
WORKDIR ${WORKDIR}

ARG DEBIAN_FRONTEND=noninteractive
ARG EXTRA_APTGET=""
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    ${EXTRA_APTGET} && \
    apt-get clean && \
    apt-get autoremove -y

RUN micromamba create -n dace-env python=${PY_VERSION} -c conda-forge -c nvidia && \
    micromamba clean -a -y

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=dace-env
RUN python -m pip install --no-cache-dir -e ./dace
RUN python -m pip install --no-cache-dir pytest
