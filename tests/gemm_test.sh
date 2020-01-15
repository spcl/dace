#!/bin/bash

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHONPATH=$SCRIPTPATH/..
PYTHON_BINARY="${PYTHON_BINARY:python3}"

DACE_debugprint="${DACE_debugprint:-0}"
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
    opts=$(join_by_newline ${@:4})
    echo "$opts\ny" | timeout $TEST_TIMEOUT $1 $PYTHONPATH/samples/simple/$2
    if [ $? -ne 0 ]; then
        /bin/echo -e "${RED}ERROR${NC} in test $1 $2 ($3, optimized)" 1>&2
        ERRORS=`expr $ERRORS + 1`
        FAILED_TESTS="${FAILED_TESTS}    $1 $2 ($3, optimized)\n"
    fi
    TESTS=`expr $TESTS + 1`
}


runone() {
    echo "Running $1"
    runtestopt $1 gemm.py $2
    runtestopt $1 gemm.py $2 'MapReduceFusion$0'
}

runall() {
    runone $PYTHON_BINARY $1
}

DACE_compiler_use_cache=0

runall "CPU"

PASSED=`expr $TESTS - $ERRORS`
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
    printf "Failed tests:\n${FAILED_TESTS}"
    exit 1
fi
