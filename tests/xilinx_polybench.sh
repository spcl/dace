#!/bin/bash

# Executes polybench kernels for correctness check
# Tests have also a timeout time (defined in variable) TEST_TIMEOUT
# this must be invoked on test directory

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHONPATH=$SCRIPTPATH/..

DACE_debugprint="${DACE_debugprint:-0}"
ERRORS=0
FAILED_TESTS=""
TIMEDOUT_TESTS=""
POLYBENCH_INPUT="small"
TESTS=0

TEST_TIMEOUT="30s"

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

timedout() {
    ERRORSTR=$1
    /bin/echo -e "${RED}ERROR${NC} in $ERRORSTR" 1>&2
    ERRORS=`expr $ERRORS + 1`
    TIMEDOUT_TESTS="${TIMEDOUT} $ERRORSTR\n"
}

run_sample() {
    # Args:
    #  1 - Relative path of FPGA test starting from test folder

    TESTS=`expr $TESTS + 1`
    echo -e "${YELLOW}Running test $1...${NC}"

    #0: dirty trick to get the number for SDFG transformation
#    python3 $1.py &> out <<- EOF
#EOF
#
#    transf_numb="$(grep "Transformation FPGATransformSDFG" out | sed -E 's/[ \t]*([0-9]+).*/\1/')"
#    echo "Transformation number for $1 is $transf_numb"

    #1: execute the benchmark with timeout
    echo "echo -e "FPGATransformSDFG$0\n\n" | timeout $TEST_TIMEOUT python3 $1.py -size ${POLYBENCH_INPUT} ${@:3}"
    echo -e "FPGATransformSDFG\$0\ny" | timeout $TEST_TIMEOUT python3 $1.py -size ${POLYBENCH_INPUT} ${@:3}
    ret_status=$?
    if [ $ret_status -ne 0 ]; then
      echo "Result " $ret_status
      if [ $ret_status -eq "124" ]; then #timeout test
        timedout "$1 (${RED}Test timeout${NC})"
      else
        bail "$1 (${RED}Result not correct${NC})"
      fi
      return 1
    fi



    return 0
}

run_all() {

    run_sample ../samples/polybench/2mm
    run_sample ../samples/polybench/3mm
    run_sample ../samples/polybench/adi
    run_sample ../samples/polybench/atax
    run_sample ../samples/polybench/bicg
    run_sample ../samples/polybench/cholesky
    run_sample ../samples/polybench/correlation
    run_sample ../samples/polybench/covariance
    run_sample ../samples/polybench/deriche
    run_sample ../samples/polybench/doitgen
    run_sample ../samples/polybench/durbin
    run_sample ../samples/polybench/fdtd-2d
    run_sample ../samples/polybench/floyd-warshall
    run_sample ../samples/polybench/gemm
    run_sample ../samples/polybench/gemver
    run_sample ../samples/polybench/gesummv
    run_sample ../samples/polybench/gramschmidt
    run_sample ../samples/polybench/heat-3d
    run_sample ../samples/polybench/jacobi-1d
    run_sample ../samples/polybench/jacobi-2d
    run_sample ../samples/polybench/ludcmp
    run_sample ../samples/polybench/lu
    run_sample ../samples/polybench/mvt
    run_sample ../samples/polybench/nussinov
    run_sample ../samples/polybench/polybench
    run_sample ../samples/polybench/seidel-2d
    run_sample ../samples/polybench/symm
    run_sample ../samples/polybench/syr2k
    run_sample ../samples/polybench/syrk
    run_sample ../samples/polybench/trisolv
    run_sample ../samples/polybench/trmm

}
# Check if xocc is vailable
which xocc
if [ $? -ne 0 ]; then
  echo "xocc not available"
  exit 99
fi

echo "====== Target: Xilinx ======"

DACE_compiler_use_cache=0
DACE_compiler_xilinx_mode="simulation"
DACE_compiler_fpga_vendor="xilinx"

echo "Attention: this will cleanup cache in 5 seconds..."
#cleanup
rm -fr .dacecache/

TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $TEST_DIR
run_all ${1:-"0"}

PASSED=`expr $TESTS - $ERRORS`
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
    printf "Failed tests:\n${FAILED_TESTS}"
    exit 1
fi
