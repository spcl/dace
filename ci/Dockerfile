ARG BASE_IMAGE=docker.io/nvidia/cuda
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ARG WORKDIR_PATH=/workspace
ENV WORKDIR=${WORKDIR_PATH}
WORKDIR ${WORKDIR}

# Install basic dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -qq -y \
    ca-certificates \
    iputils-ping \
    curl \
    git \
    gnupg \
    build-essential && \
    apt-get clean

# HIP's NVIDIA platform: headers over the CUDA toolkit, built by nvcc. Not hip-runtime-nvidia --
# that metapackage depends on the whole cuda package, and so on a display driver.
#
# NOT via apt: repo.radeon.com publishes ``Architectures: amd64`` only, so on this aarch64 image
# apt cannot satisfy hip-dev/hipcc-nvidia at all ("held broken packages"). What those packages
# hold, though, is portable -- hip-dev is nothing but headers, CMake files and perl, and hipcc is
# a compiler driver whose NVIDIA path (src/hipBin_nvidia.h) just assembles an nvcc command line.
# So unpack the headers straight out of the .deb, and build the driver from its own source for
# this architecture. The one binary in hipcc-nvidia is x86-64 and is what apt could not have
# given us anyway.
ARG ROCM_VERSION=7.2.4
ARG HIP_DEV_DEB=hip-dev_7.2.53211.70204-93~24.04_amd64.deb
RUN apt-get update -qq && apt-get install -qq -y cmake && apt-get clean && \
    curl -fsSL -o /tmp/hip-dev.deb \
      "https://repo.radeon.com/rocm/apt/${ROCM_VERSION}/pool/main/h/hip-dev/${HIP_DEV_DEB}" && \
    dpkg-deb -x /tmp/hip-dev.deb / && \
    rm /tmp/hip-dev.deb && \
    git clone -q --depth 1 --filter=blob:none --sparse -b "rocm-${ROCM_VERSION}" \
      https://github.com/ROCm/llvm-project.git /tmp/hipcc-src && \
    git -C /tmp/hipcc-src sparse-checkout set amd/hipcc && \
    cmake -S /tmp/hipcc-src/amd/hipcc -B /tmp/hipcc-build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="/opt/rocm-${ROCM_VERSION}" && \
    cmake --build /tmp/hipcc-build -j "$(nproc)" && \
    cmake --install /tmp/hipcc-build && \
    rm -rf /tmp/hipcc-src /tmp/hipcc-build && \
    ln -sfn "/opt/rocm-${ROCM_VERSION}" /opt/rocm
ENV PATH="/opt/rocm/bin:$PATH"

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
    codecov-cli \
    flake8
RUN uv pip install cupy
