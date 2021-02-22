#!/bin/bash
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

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
    (cd .dacecache/$2/build && make xilinx_synthesis)
    if [ $? -ne 0 ]; then
      bail "$1 (${RED}high-level synthesis failed${NC})"
      return 1
    fi
    if [ $3 -ne 0 ]; then
      grep -n .dacecache/$2/build/*_hls.log -e "Final II = \([2-9]\|1[0-9]+\)"
      if [ $? == 0 ]; then
        bail "$1 (${RED}design was not fully pipelined${NC})"
      fi
    fi
    return 0
}

run_all() {

    # Args:
    #  0: Boolean flag that runs all (1) or a reduced set (0) of samples
    run_sample fpga/remove_degenerate_loop remove_degenerate_loop_test 0
    run_sample fpga/pipeline_scope pipeline_test 1
    run_sample fpga/veclen_copy_conversion veclen_copy_conversion 1
    run_sample ../samples/fpga/axpy_transformed axpy_fpga_24 1 24
    run_sample ../samples/fpga/spmv_fpga_stream spmv_fpga_stream 0 64 64 640
    run_sample ../samples/fpga/matrix_multiplication_systolic mm_fpga_systolic_4_64x64x64 1 64 64 64 4 -specialize
    run_sample ../samples/fpga/filter_fpga_vectorized filter_fpga_vectorized_4 1 8192 4 0.25
    # run_sample jacobi_fpga_systolic jacobi_fpga_systolic_4_Hx128xT 1 128 128 8 4
    run_sample ../samples/fpga/gemv_transposed_fpga gemv_transposed_1024xM 0 1024 1024
    if [ "$1" -ne "0" ]; then
      run_sample ../samples/fpga/histogram_fpga histogram_fpga 0 128 128
      run_sample ../samples/fpga/spmv_fpga spmv_fpga 0 64 64 640
      run_sample ../samples/fpga/matrix_multiplication_pipelined mm_fpga_pipelined_NxKx128 1 128 128 128
      run_sample ../samples/fpga/matrix_multiplication_stream mm_fpga_stream_NxKx64 1 64 64 64
      run_sample ../samples/fpga/filter_fpga filter_fpga 1 8192 0.5
      run_sample ../samples/fpga/jacobi_fpga_stream jacobi_fpga_stream_Hx128xT 1 128 128 8
    fi

    run_sample fpga/multiple_kernels multiple_kernels 0
    run_sample fpga/unique_nested_sdfg_fpga two_vecAdd 0
    run_sample fpga/nested_sdfg_as_kernel nested_sdfg_kernels 0
    run_sample fpga/streaming_memory streamingcomp_1 1

    ## BLAS
    run_sample blas/nodes/axpy_test axpy_test_fpga_1_w4_1 1 --target fpga 
    run_sample blas/nodes/dot_test dot_FPGA_PartialSums_float_w16_1 1 --target xilinx
    run_sample blas/nodes/gemv_test gemv_FPGA_TilesByColumn_float_True_w4_1 1 --target tiles_by_column --transpose --vectorize 4
    run_sample blas/nodes/gemv_test gemv_FPGA_Accumulate_float_False_w4_1 1 --target accumulate --vectorize 4 
    run_sample blas/nodes/ger_test ger_test_1 1 --target fpga 

    # Multiple gearboxing
    run_sample fpga/multiple_veclen_conversions multiple_veclen_conversions 0
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
