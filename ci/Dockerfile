ARG BASE_IMAGE=docker.io/nvidia/cuda
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ARG WORKDIR_PATH=/workspace
ENV WORKDIR=${WORKDIR_PATH}
WORKDIR ${WORKDIR}

# Install basic dependencies
RUN apt-get update -qq && apt-get install -qq -y \
    ca-certificates \
    curl \
    git \
    build-essential && \
    apt-get clean

# Install uv
ARG UV_VERSION=0.6.12
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN UV_UNMANAGED_INSTALL="/opt/uv" sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/opt/uv:$PATH"

# Set up the environment vars with paths for uv
ARG CACHE_DIR=${WORKDIR_PATH}/cache/
ENV UV_CACHE_DIR="${CACHE_DIR}/uv"
ENV UV_PROJECT_ENVIRONMENT=${WORKDIR_PATH}/venv
ENV UV_PYTHON_INSTALL_DIR="/opt/uv-python"

# Create and activate the python venv
ENV VIRTUAL_ENV=${UV_PROJECT_ENVIRONMENT}
ARG PY_VERSION=3.10
RUN uv venv -p ${PY_VERSION} --link-mode copy ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Install test dependencies
RUN uv pip install \
    pytest-xdist \
    coverage \
    flake8 \
    cupy

# Install codecov
RUN curl -Os https://uploader.codecov.io/latest/linux/codecov && \
    chmod +x codecov && \
    mv codecov /usr/local/bin/codecov
