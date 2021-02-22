#!/bin/bash
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

DACE_optimizer_transform_on_call=1
DACE_optimizer_interface="dace.transformation.testing.TransformationTester"
DACE_TEST_NAME="Transformations"

# Run the end-to-end tests with the TransformationTester
exec $SCRIPTPATH/e2e_test.sh $*
