#!/bin/sh

set -e

# Install dependencies
pip install --upgrade twine

# Synchronize submodules
git submodule update --init --recursive

# Erase old distribution, if exists
rm -rf dist dace.egg-info

# Make tarball
python -m build --sdist

# Upload to PyPI
twine upload dist/*

