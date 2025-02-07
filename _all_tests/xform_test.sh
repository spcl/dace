#!/bin/bash
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

DACE_call_hooks="dace.transformation.testing.test_transformations_hook"
DACE_TEST_NAME="Transformations"

# Run the end-to-end tests with the TransformationTester
exec $SCRIPTPATH/e2e_test.sh $*
