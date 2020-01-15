#!/bin/bash

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHONPATH=$SCRIPTPATH/..
PYTHON_BINARY="${PYTHON_BINARY:python3}"

ERRORS=0
FAILED_TESTS=""
TESTS=0

TEST_TIMEOUT=10m

RED='\033[0;31m'
NC='\033[0m'
        

runtest() {
    yes | timeout $TEST_TIMEOUT $1 $PYTHONPATH/samples/simple/$2
    if [ $? -ne 0 ]; then
        /bin/echo -e "${RED}ERROR${NC} in test $1 $2 ($3)" 1>&2
        ERRORS=`expr $ERRORS + 1`
        FAILED_TESTS="${FAILED_TESTS}    $1 $2 ($3)\n"
    fi
    TESTS=`expr $TESTS + 1`
}

runtestopt() {
    echo "$4\ny" | timeout $TEST_TIMEOUT $1 $PYTHONPATH/samples/simple/$2
    if [ $? -ne 0 ]; then
        /bin/echo -e "${RED}ERROR${NC} in test $1 $2 ($3, optimized)" 1>&2
        ERRORS=`expr $ERRORS + 1`
        FAILED_TESTS="${FAILED_TESTS}    $1 $2 ($3, optimized)\n"
    fi
    TESTS=`expr $TESTS + 1`
}


runone() {
    echo "Running $1"
    runtest $1 axpy.py $2
    runtest $1 ddot.py $2
    runtest $1 fibonacci.py $2
    runtest $1 filter.py $2
    runtest $1 gemm.py $2
    runtest $1 histogram.py $2
    runtest $1 histogram_declarative.py $2
    runtest $1 jacobi.py $2
    runtest $1 mandelbrot.py $2
    runtest $1 mat_add.py $2
    runtest $1 spmv.py $2
    runtest $1 sum.py $2
    runtest $1 transpose.py $2
}

runall() {
    runone $PYTHON_BINARY $1
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
