#!/bin/sh

set -e

# Synchronize submodules
git submodule update --init --recursive

# Erase old distribution, if exists
rm -rf dist dace.egg-info

# Make tarball
uv build --sdist --no-sources

# Upload to PyPI
uv publish dist/*
