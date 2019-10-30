#!/bin/bash

# Executes a bunch of small test for the intel fpga backend
# This are not intended to be performance test: they just check that everything compiles
# and the correct result is produced

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHONPATH=$SCRIPTPATH/..

DACE_debugprint="${DACE_debugprint:-0}"
ERRORS=0
FAILED_TESTS=""
TESTS=0

TEST_TIMEOUT=10

RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'


################################################

bail() {
    ERRORSTR=$1
    /bin/echo -e "${RED}ERROR${NC} in $ERRORSTR" 1>&2
    ERRORS=`expr $ERRORS + 1`
    FAILED_TESTS="${FAILED_TESTS} $ERRORSTR\n"
}

run_sample() {
    # Args:
    #  1 - Relative path of FPGA test starting from test folder
    #  2 - Name of the DAPP program
    #  3 - a string indicating the list of input to pass to the python program (the trasnformation sequence)
    #  4 - program command line argument (if any)

    TESTS=`expr $TESTS + 1`
    echo -e "${YELLOW}Running test $1...${NC}"

    #1: generate the opencl
    #trick: use type scripting to mask the first segfault due to the missing aocx file
    command='echo -e "'"${3}"'" |python3 '"${1}"'.py '"${@:4}"''
    script -c  "$command" /tmp/test_intelfpga > /dev/null
    #echo -e $3 | python3 $1.py ${@:5} &> /dev/null

    #2: compile for emulation
    cd .dacecache/$2/build
    make intelfpga_compile_$2_emulator
    if [ $? -ne 0 ]; then
      bail "$1 (${RED}high-level synthesis failed${NC})"
      return 1
    fi

    #3: execute the emulation
    cd ../../../
    echo -e $3 | python3 $1.py ${@:4}

    if [ $? -ne 0 ]; then
        bail "$1 (${RED}Wrong emulation result${NC})"
    fi

    #4 cleanup
    cd .dacecache/$2/build
    rm -fr $1_*
    cd -



    return 0
}

run_all() {
    # #### VECTORIZATION ####
    #Vectorization 1: first vectorize and then transform for FPGA
    run_sample intel_fpga/vec_sum vec_sum "11\n1\n"
    #Vectorization 2: first transform for FPGA then vectorize
    run_sample intel_fpga/vec_sum vec_sum "1\n15\n"
    #Vectorization 3: TODO non vectorizable N

    # #### WCR ####
    #simple WCR (accumulates on scalar)
    run_sample intel_fpga/dot dot "1\n"

    # histogram (WCR on array)
    run_sample ../samples/simple/histogram histogram "1\n"

    # #### REDUCE ####
    # Simple reduce
    run_sample intel_fpga/vector_reduce vector_reduce "1\n"

    # GEMM sample
    run_sample ../samples/simple/gemm gemm "1\n"

    # #### TYPE INFERENCE ####
    run_sample ../samples/simple/mandelbrot mandelbrot "1\n"

    # type inference for statements with annotation
    run_sample intel_fpga/type_inference type_inference "1\n"


    # #### SYSTOLIC ARRAY ###
    run_sample intel_fpga/simple_systolic_array simple_systolic_array_4 "\n" 128 4
    run_sample intel_fpga/gemm_systolic_array gemm_fpga_systolic_4_NxKx256 "\n" 256 256 256 4



    # #### MISCELLANEA ####
    # Execute some of the compatible tests in samples/fpga (some of them have C++ code in tasklet)
    # They contain streams
    run_sample ../samples/fpga/filter_fpga filter_fpga "\n" 1000 0.2
    run_sample ../samples/fpga/gemm_fpga_stream gemm_fpga_stream_NxKx128 "\n" 128 128 128
    run_sample ../samples/fpga/spmv_fpga_stream spmv_fpga_stream "\n" 128 128 64
}

# Check if aoc is vailable
which aoc
if [ $? -ne 0 ]; then
  echo "aocc not available"
  exit 99
fi

echo "====== Target: INTEL FPGA ======"

DACE_compiler_use_cache=0
DACE_compiler_fpga_vendor="intel_fpga"

TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $TEST_DIR
run_all ${1:-"0"}

PASSED=`expr $TESTS - $ERRORS`
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
    printf "Failed tests:\n${FAILED_TESTS}"
    exit 1
fi
