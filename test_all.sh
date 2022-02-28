#!/bin/bash
# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHONPATH=$SCRIPTPATH

DACE_debugprint="${DACE_debugprint:-0}"
DACE_testing_serialization="${DACE_testing_serialization:-1}"
DACE_cache="${DACE_cache:-single}"
DACE_optimizer_interface="${DACE_optimizer_interface:-dace.transformation.optimizer.SDFGOptimizer}"
DACE_optimizer_transform_on_call="${DACE_optimizer_transform_on_call:-1}"
NOSTATUSBAR="${NOSTATUSBAR:-0}"
ERRORS=0
FAILED_TESTS=""
SKIPS=0
SKIPPED_TESTS=""
TESTS=0
CURTEST=""
TESTPREFIX=""
TOTAL_TESTS=0
PYTHON_BINARY="${PYTHON_BINARY:-python3}"

TEST_TIMEOUT=10

RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'
TGAP="                                                                                  "

join_by_newline() {
    for a in $*; do
        echo $a        
    done
    echo 9999
}

bail() {
    ERRORSTR="$TESTPREFIX$1"
    /bin/echo -e "${RED}ERROR${NC} in $ERRORSTR" 1>&2
    ERRORS=`expr $ERRORS + 1`
    FAILED_TESTS="${FAILED_TESTS} $ERRORSTR\n"
}

bail_skip() {
    ERRORSTR="$TESTPREFIX$1"
    /bin/echo -e "${YELLOW}SKIPPING${NC} $ERRORSTR" 1>&2
    SKIPS=`expr $SKIPS + 1`
    SKIPPED_TESTS="${SKIPPED_TESTS} $ERRORSTR\n"
}


test_start() {
    TESTS=`expr $TESTS + 1`
    CURTEST="$TESTPREFIX$1"
    echo "---------- TEST: $TESTPREFIX$1 ----------"
}

testcmd() {
    if [ $NOSTATUSBAR -ne 0 ]; then
        $*
        return
    fi
    #$* | tee -a test.log
    TESTCNT=`expr $TESTS - 1`
    MSG="($TESTCNT / $TOTAL_TESTS) $CURTEST (Fails: $ERRORS)"
    ($* || echo "_TFAIL_ $?") |& awk "BEGIN{printf \"$MSG\r\"} /_TFAIL_/{printf \"$TGAP\r\"; exit \$NF} {printf \"$TGAP\r\"; print; printf \"$MSG\r\";} END{printf \"$TGAP\r\"}"
}

################################################

runtest_cpp() {
    test_start $1
    testcmd g++ -std=c++14 -Wall -Wextra -O3 -march=native -ffast-math -fopenmp -fPIC \
        -I $SCRIPTPATH/dace/runtime/include $1 -o ./$1.out
    if [ $? -ne 0 ]; then bail "$1 (compilation)"; fi
    testcmd ./$1.out
    retval=$?
    rm -f $1.out
    if [ $retval -ne 0 ]; then bail $1; fi
}

runtest_cu() {
    test_start $1
    testcmd nvcc -O3 -I $SCRIPTPATH/dace/runtime/include $1 -o ./$1.out
    if [ $? -ne 0 ]; then bail "$1 (compilation)"; fi

    # Check if GPU tests can be run
    nvidia-smi >/dev/null 2>&1
    if [ $? -ne 0 ]; then bail_skip $1; return; fi
    
    testcmd ./$1.out
    retval=$?
    rm -f $1.out
    if [ $? -ne 0 ]; then bail $1; fi
}

runtest_octave() {
    test_start $1
    testcmd $PYTHON_BINARY $SCRIPTPATH/dace/frontend/octave/dacelab.py $1
    if [ $? -ne 0 ]; then bail $1; fi
}

runtest_py() {
    test_start $1
    yes | testcmd $PYTHON_BINARY $1
    if [ $? -ne 0 ]; then bail $1; fi
}

runtest_sh() {
    test_start $1
    testcmd ./$1
    retval=$?
    if [ $retval -eq 99 ]; then bail_skip $1; return; fi
    if [ $retval -ne 0 ]; then bail $1; fi
}

endreport() {
    PASSED=`expr $TESTS - $ERRORS - $SKIPS`
    TOTAL=`expr $TESTS - $SKIPS`
    echo "$PASSED / $TOTAL tests passed"
    if [ $SKIPS -ne 0 ]; then
        printf "Skipped tests:\n${SKIPPED_TESTS}"
    fi    
    if [ $ERRORS -ne 0 ]; then
        printf "Failed tests:\n${FAILED_TESTS}"
        exit 1
    fi
}

echo "====== All-Inclusive Test Runner ======"

cd $SCRIPTPATH/tests

SUBTESTS=`find . -type d -not -path '*/\.*' | cut -c3- | grep -v "/[._]" | grep -v '^[._]'`

DACE_compiler_use_cache=0
DACE_optimizer_detect_control_flow=1

# Specific test(s)
if [ $# -ne 0 ]; then
    TOTAL_TESTS=$#
    for arg in "$@"; do
        if [[ $arg == *_test.cpp ]]; then
            runtest_cpp $arg
        elif [[ $arg == *_test.cu ]]; then
            runtest_cu $arg
        elif [[ $arg == *_test.py ]]; then
            runtest_py $arg
        elif [[ $arg == *.m ]]; then
            runtest_octave $arg
        elif [[ $arg == *_test.sh ]]; then
            runtest_sh $arg
        fi
    done
    endreport
    exit 0
fi

# Count tests first
counttests() {
    for file in *_test.cpp; do
        if [ $file == '*_test.cpp' ]; then break; fi # No files found
        TOTAL_TESTS=`expr $TOTAL_TESTS + 1`
    done

    for file in *_test.cu; do
        if [ $file == '*_test.cu' ]; then break; fi # No files found
        TOTAL_TESTS=`expr $TOTAL_TESTS + 1`
    done

    for file in *_test.py; do
        if [ $file == '*_test.py' ]; then break; fi # No files found
        TOTAL_TESTS=`expr $TOTAL_TESTS + 1`
    done

    for file in *.m; do
        if [ $file == '*.m' ]; then break; fi # No files found
        TOTAL_TESTS=`expr $TOTAL_TESTS + 1`
    done

    for file in *_test.sh; do
        if [ $file == '*_test.sh' ]; then break; fi # No files found
        TOTAL_TESTS=`expr $TOTAL_TESTS + 1`
    done
}

# Count tests in top-level folder
cd $SCRIPTPATH/tests
counttests

# Count sub-tests
for test in $SUBTESTS; do
    cd $SCRIPTPATH/tests/$test
    counttests
done

################################################################

runtests() {
    for file in *_test.cpp; do
        if [ $file == '*_test.cpp' ]; then break; fi # No files found
        runtest_cpp $file
    done

    for file in *_test.cu; do
        if [ $file == '*_test.cu' ]; then break; fi # No files found
        runtest_cu $file
    done

    for file in *_test.py; do
        if [ $file == '*_test.py' ]; then break; fi # No files found
        runtest_py $file
    done

    for file in *.m; do
        if [ $file == '*.m' ]; then break; fi # No files found
        runtest_octave $file
    done

    for file in *_test.sh; do
        if [ $file == '*_test.sh' ]; then break; fi # No files found
        runtest_sh $file
    done
}

cd $SCRIPTPATH/tests
TESTPREFIX=""
runtests

for test in $SUBTESTS; do
    cd $SCRIPTPATH/tests/$test
    TESTPREFIX="$test/"
    runtests
done

endreport
