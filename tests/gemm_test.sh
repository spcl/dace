#!/bin/bash
# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHONPATH=$SCRIPTPATH/..
PYTHON_BINARY="${PYTHON_BINARY:-python3}"

DACE_debugprint="${DACE_debugprint:-0}"
DACE_optimizer_transform_on_call=${DACE_optimizer_transform_on_call:-1}
ERRORS=0
FAILED_TESTS=""
TESTS=0

TEST_TIMEOUT=10m

RED='\033[0;31m'
NC='\033[0m'

join_by_newline() {
    for a in $*; do
        echo $a        
    done
    echo 9999
}


runtestopt() {
    opts=$(join_by_newline ${@:3})
    echo "$opts\ny" | timeout $TEST_TIMEOUT $PYTHON_BINARY $PYTHONPATH/samples/simple/$1
    if [ $? -ne 0 ]; then
        /bin/echo -e "${RED}ERROR${NC} in test $1 ($2, optimized)" 1>&2
        ERRORS=`expr $ERRORS + 1`
        FAILED_TESTS="${FAILED_TESTS}    $1 ($2, optimized)\n"
    fi
    TESTS=`expr $TESTS + 1`
}


runall() {
    echo "Running $PYTHON_BINARY"
    runtestopt gemm.py $1
    runtestopt gemm.py $1 'MapReduceFusion$0'
}

DACE_compiler_use_cache=0

runall "CPU"

PASSED=`expr $TESTS - $ERRORS`
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
    printf "Failed tests:\n${FAILED_TESTS}"
    exit 1
fi
