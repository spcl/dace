#!/bin/bash

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHONPATH=$SCRIPTPATH/..

DACE_debugprint="${DACE_debugprint:-0}"
ERRORS=0
FAILED_TESTS=""
TESTS=0
PYTHON_BINARY="${PYTHON_BINARY:-python3}"

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
    #  1 - Name of FPGA sample located in samples/fpga
    #  2 - Boolean flag whether to assert II=1 in all loops
    #  3-x - Other args to forward to kernel
    TESTS=`expr $TESTS + 1`
    echo -e "${YELLOW}Running test $1...${NC}"
    yes | $PYTHON_BINARY $1.py ${@:4}
    if [ $? -ne 0 ]; then
      bail "$1 (${RED}simulation failed${NC})"
      return 1
    fi
    (cd .dacecache/$2/build && make xilinx_compile_hardware)
    if [ $? -ne 0 ]; then
      bail "$1 (${RED}high-level synthesis failed${NC})"
      return 1
    fi
    if [ $3 -ne 0 ]; then
      grep -n .dacecache/$2/build/xocc_*_hw.log -e "Final II = \([2-9]\|1[0-9]+\)"
      if [ $? == 0 ]; then
        bail "$1 (${RED}design was not fully pipelined${NC})"
      fi
    fi
    return 0
}

run_all() {
    # Args:
    #  0: Boolean flag that runs all (1) or a reduced set (0) of samples
    run_sample remove_degenerate_loop remove_degenerate_loop_test 0
    run_sample pipeline_scope pipeline_test 1
    run_sample veclen_copy_conversion veclen_copy_conversion 1
    run_sample ../samples/fpga/axpy_transformed axpy_fpga_24 0 24
    run_sample ../samples/fpga/spmv_fpga_stream spmv_fpga_stream 0 64 64 640
    run_sample ../samples/fpga/gemm_fpga_systolic gemm_fpga_systolic_4_64x64x64 1 64 64 64 4 -specialize 
    run_sample ../samples/fpga/filter_fpga_vectorized filter_fpga_vectorized_4 1 8192 4 0.25
    # run_sample jacobi_fpga_systolic jacobi_fpga_systolic_4_Hx128xT 1 128 128 8 4
    # TODO: this doesn't pipeline. Should it? Why doesn't it?
    run_sample ../samples/fpga/gemv_transposed_fpga gemv_transposed_1024xM 0 1024 1024
    if [ "$1" -ne "0" ]; then
      run_sample ../samples/fpga/histogram_fpga histogram_fpga 0 128 128
      run_sample ../samples/fpga/spmv_fpga spmv_fpga 0 64 64 640
      run_sample ../samples/fpga/gemm_fpga_pipelined gemm_fpga_pipelined_NxKx128 1 128 128 128
      run_sample ../samples/fpga/gemm_fpga_stream gemm_fpga_stream_NxKx64 1 64 64 64
      run_sample ../samples/fpga/filter_fpga filter_fpga 1 8192 0.5
      run_sample ../samples/fpga/jacobi_fpga_stream jacobi_fpga_stream_Hx128xT 1 128 128 8
    fi
}

# Check if xocc is vailable
which v++ 
if [ $? -ne 0 ]; then
  which xocc
  if [ $? -ne 0 ]; then
    echo "v++/xocc not available"
    exit 99
  fi
fi

echo "====== Target: Xilinx ======"

DACE_compiler_use_cache=0
DACE_testing_single_cache=0
DACE_compiler_fpga_vendor="xilinx"
DACE_compiler_xilinx_mode="simulation"

TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $TEST_DIR
run_all ${1:-"0"}

PASSED=`expr $TESTS - $ERRORS`
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
    printf "Failed tests:\n${FAILED_TESTS}"
    exit 1
fi
