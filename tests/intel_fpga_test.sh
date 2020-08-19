#!/bin/bash

# Executes a bunch of small test for the Intel FPGA backend
# These are not intended to be performance test: they just check that everything compiles
# and the correct result is produced

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHONPATH=$SCRIPTPATH/..
PYTHON_BINARY="${PYTHON_BINARY:-python3}"

DACE_debugprint="${DACE_debugprint:-0}"
DACE_optimizer_transform_on_call=${DACE_optimizer_transform_on_call:-1}
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
    #  2 - Name of the DaCe program
    #  3 - a string indicating the list of input to pass to the python program (the transformation sequence)
    #  4 - program command line argument (if any)

    TESTS=`expr $TESTS + 1`
    echo -e "${YELLOW}Running test $1...${NC}"

    echo -e ${3} | $PYTHON_BINARY ${1}.py ${@:4}

    if [ $? -ne 0 ]; then
        bail "$1"
    fi

    return 0
}

run_all() {

    #### Vectorization ####
    # TODO: These tests require getting access to how types are generated in
    # process_out_memlets on a fine-grained level, which is not yet possible.
    # This will be implement as part of the new codegen.
    # run_sample intel_fpga/vec_sum vec_sum "\n" true
    # run_sample intel_fpga/vec_sum vec_sum "\n" false
    # run_sample fpga/veclen_conversion_connector "\n"
    run_sample fpga/veclen_conversion "\n"
    run_sample fpga/veclen_copy_conversion "\n"

    # Throw error when kernel names are too long
    run_sample intel_fpga/name_too_long name_too_long "\n"

    # Test removing degenerate loops that only have a single iteration
    run_sample fpga/remove_degenerate_loop remove_degenerate_loop_test "\n" 

    # Test pipeline scopes 
    run_sample fpga/pipeline_scope pipeline_scope "\n" 

    # Test shift register abstraction with stencil code
    run_sample fpga/fpga_stencil fpga_stencil_test "\n"

    ### MAP TILING ####
    # First tile then transform
    run_sample intel_fpga/dot dot "MapTiling\$0\nFPGATransformSDFG\$0\n"
    # Other way around
    run_sample intel_fpga/dot dot "FPGATransformSDFG\$0\nMapTiling\$0\n"

    #### WCR ####
    # simple WCR (accumulates on scalar)
    run_sample intel_fpga/dot dot "FPGATransformSDFG\$0\n"

    #### REDUCE ####
    # Simple reduce
    run_sample intel_fpga/vector_reduce vector_reduce "FPGATransformSDFG\$0\n"

    # GEMM sample
    run_sample ../samples/simple/gemm gemm "FPGATransformSDFG\$0\n"

    #### TYPE INFERENCE ####
    run_sample ../samples/simple/mandelbrot mandelbrot "FPGATransformSDFG\$0\n"

    # type inference for statements with annotation
    run_sample intel_fpga/type_inference type_inference "FPGATransformSDFG\$0\n"

    #### SYSTOLIC ARRAY ###
    run_sample intel_fpga/simple_systolic_array simple_systolic_array_4 "\n" 128 4
    run_sample ../samples/fpga/gemm_fpga_systolic gemm_fpga_systolic_4_NxKx256 "\n" 256 256 256 4
    run_sample ../samples/fpga/jacobi_fpga_systolic jacobi_fpga_systolic_8_Hx8192xT "\n"

    #### MISCELLANEOUS ####
    # Execute some of the compatible tests in samples/fpga (some of them have C++ code in tasklet)
    # They contain streams
    run_sample intel_fpga/async async_test "\n" 
    run_sample ../samples/fpga/filter_fpga filter_fpga "\n" 1000 0.2
    run_sample ../samples/fpga/gemm_fpga_stream gemm_fpga_stream_NxKx128 "\n" 128 128 128
    run_sample ../samples/fpga/spmv_fpga_stream spmv_fpga_stream "\n" 128 128 64
    run_sample ../samples/fpga/axpy_transformed axpy_fpga_24 "\n" 24
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
