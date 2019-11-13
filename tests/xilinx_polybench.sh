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

    #1: execute the benchmark with timeout
    echo -e "FPGATransformSDFG\$${3}\ny" | timeout $TEST_TIMEOUT python3 $1.py -size ${POLYBENCH_INPUT} ${@:3}
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

    run_sample 2mm k2mm 0
    run_sample 3mm k3mm 0
    run_sample adi adi 0
    run_sample atax atax 0
    run_sample bicg bicg 0
    run_sample cholesky cholesky 0
    run_sample correlation correlation 0
    run_sample covariance covariance 0
    run_sample deriche deriche 0
    run_sample doitgen doitgen 1
    run_sample durbin durbin 0
    run_sample fdtd-2d fdtd2d 0
    run_sample  floyd-warshall floyd_warshall 0
    run_sample gemm gemm 0
    run_sample gemver gemver 0
    run_sample gesummv gesummv 0
    run_sample gramschmidt gramschmidt 1
    run_sample heat-3d heat3d 0
    run_sample jacobi-1d jacobi1d 0
    run_sample jacobi-2d jacobi2d 0
    run_sample ludcmp ludcmp 0
    run_sample lu lu 0
    run_sample mvt mvt 0
    run_sample nussinov nussinov 0
    run_sample seidel-2d seidel2d 0
    run_sample symm symm 1
    run_sample syr2k syr2k 0
    run_sample syrk syrk 0
    run_sample trisolv trisolv 0
    run_sample trmm trmm 1

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
cd $TEST_DIR/../samples/polybench
run_all

PASSED=`expr $TESTS - $ERRORS`
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
    printf "Failed tests:\n${FAILED_TESTS}"
    exit 1
fi
