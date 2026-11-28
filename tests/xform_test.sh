#!/bin/bash
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

DACE_call_hooks="dace.transformation.testing.test_transformations_hook"
DACE_TEST_NAME="Transformations"
# GPU transformations size grids by the target device; CPU-only runners describe one as tests/conftest.py does
DACE_compiler_cuda_device_multiprocessors=${DACE_compiler_cuda_device_multiprocessors:-304}

# Run the end-to-end tests with the TransformationTester
exec $SCRIPTPATH/e2e_test.sh $*
