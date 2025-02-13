#!/bin/bash
# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHONPATH=$SCRIPTPATH/..
PYTHON_BINARY="${PYTHON_BINARY:-python3}"

ERRORS=0
FAILED_TESTS=""
TESTS=0

TEST_TIMEOUT=10m

RED='\033[0;31m'
NC='\033[0m'
        

runtest() {
    yes | timeout $TEST_TIMEOUT $PYTHON_BINARY $PYTHONPATH/samples/$1
    if [ $? -ne 0 ]; then
        /bin/echo -e "${RED}ERROR${NC} in test $1 ($2)" 1>&2
        ERRORS=`expr $ERRORS + 1`
        FAILED_TESTS="${FAILED_TESTS}    $1 ($2)\n"
    fi
    TESTS=`expr $TESTS + 1`
}

runtestopt() {
    echo "$3\ny" | timeout $TEST_TIMEOUT $PYTHON_BINARY $PYTHONPATH/samples/$1
    if [ $? -ne 0 ]; then
        /bin/echo -e "${RED}ERROR${NC} in test $1 ($2, optimized)" 1>&2
        ERRORS=`expr $ERRORS + 1`
        FAILED_TESTS="${FAILED_TESTS}    $1 ($2, optimized)\n"
    fi
    TESTS=`expr $TESTS + 1`
}


runall() {
    echo "Running $PYTHON_BINARY"
    runtest simple/axpy.py $1
    runtest simple/laplace.py $1
    runtest simple/mandelbrot.py $1
    runtest simple/spmv.py $1
    runtest explicit/fibonacci.py $1
    runtest explicit/filter.py $1
    runtest explicit/histogram.py $1
    runtest optimization/matmul.py $1  
}

DACE_compiler_use_cache=0
DACE_TEST_NAME="${DACE_TEST_NAME:-CPU}"

runall $DACE_TEST_NAME

PASSED=`expr $TESTS - $ERRORS`
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
    printf "Failed tests:\n${FAILED_TESTS}"
    exit 1
fi
