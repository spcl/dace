ARG BASE_IMAGE=docker.io/ubuntu:24.04
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ARG WORKDIR_PATH=/workspace
ENV WORKDIR=${WORKDIR_PATH}
WORKDIR ${WORKDIR}

ARG DEBIAN_FRONTEND=noninteractive
ARG EXTRA_APTGET=""
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    git \
    ${EXTRA_APTGET} && \
    apt-get clean && \
    apt-get autoremove -y