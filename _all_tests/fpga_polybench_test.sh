#!/bin/bash
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Executes polybench kernels for correctness check, by default on both Xilinx and Intel FPGA
# Tests have also a timeout time (defined in variable) TEST_TIMEOUT
# this must be invoked on test directory
# usage: fpga_polybench.sh <intel_fpga/xilinx>

set -a

SCRIPTPATH="$(
  cd "$(dirname "$0")"
  pwd -P
)"
PYTHONPATH=$SCRIPTPATH/..

DACE_debugprint="${DACE_debugprint:-0}"
ERRORS=0
FAILED_TESTS=""
TIMEDOUT_TESTS=""
POLYBENCH_INPUT="mini"
TESTS=0
PYTHON_BINARY="${PYTHON_BINARY:-python3}"

TEST_TIMEOUT="30s"

RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

################################################

bail() {
  ERRORSTR=$1
  /bin/echo -e "${RED}ERROR${NC} in $ERRORSTR" 1>&2
  ERRORS=$(expr $ERRORS + 1)
  FAILED_TESTS="${FAILED_TESTS} $ERRORSTR\n"
}

timedout() {
  ERRORSTR=$1
  /bin/echo -e "${RED}ERROR${NC} in $ERRORSTR" 1>&2
  ERRORS=$(expr $ERRORS + 1)
  TIMEDOUT_TESTS="${TIMEDOUT_TESTS} $ERRORSTR\n"
}

run_sample_intel() {
  # Args:
  #  1 - Relative path of FPGA test starting from test folder
  #  2 - name of the build folder
  #  3 - number of the SDFG transformation to apply
  TESTS=$(expr $TESTS + 1)
  echo -e "${YELLOW}Running test $1...${NC}"

  #1: generate the opencl
  #remove previously built version. This helps to avoid stall in case the program does not terminates
  rm -fr .dacecache/$2 2>/dev/null
  echo -e "FPGATransformSDFG\$${3}\ny" | $PYTHON_BINARY $1.py -size ${POLYBENCH_INPUT} 2>/dev/null | :

  #2: compile for emulation
  cd .dacecache/$2/build
  if [ $? -ne 0 ]; then
    bail "$1 (${RED}Code generation failed${NC})"
    return 1
  fi

  make intelfpga_compile_$2_emulator
  cd ../../../

  if [ $? -ne 0 ]; then
    bail "$1 (${RED}high-level synthesis failed${NC})"
    return 1
  fi

  #3: execute the emulation with timeout
  echo -e "FPGATransformSDFG\$${3}\ny" | timeout $TEST_TIMEOUT $PYTHON_BINARY $1.py -size ${POLYBENCH_INPUT}

  ret_status=$?
  if [ $ret_status -ne 0 ]; then
    echo "Result " $ret_status
    if [ $ret_status -eq "124" ]; then #timeout test
      timedout "Intel_FPGA: $1 (${RED}Test timeout${NC})"
    else
      bail "Intel_FPGA: $1 (${RED}Result not correct${NC})"
    fi
    return 1
  fi

  #4 cleanup
  cd .dacecache/$2/build
  rm -fr $1_*
  cd -

  return 0
}

run_sample_xilinx() {
  # Args:
  #  1 - Relative path of FPGA test starting from test folder
  #  2 - name of the build folder
  #  3 - number of the SDFG transformation to apply
  TESTS=$(expr $TESTS + 1)
  echo -e "${YELLOW}Running test $1...${NC}"

  #1: execute the benchmark with timeout
  echo -e "FPGATransformSDFG\$${3}\ny" | timeout $TEST_TIMEOUT $PYTHON_BINARY $1.py -size ${POLYBENCH_INPUT} ${@:3}
  ret_status=$?
  if [ $ret_status -ne 0 ]; then
    echo "Result " $ret_status
    if [ $ret_status -eq "124" ]; then #timeout test
      timedout "Xilinx: $1 (${RED}Test timeout${NC})"
    else
      bail "Xilinx: $1 (${RED}Result not correct${NC})"
    fi
    return 1
  fi

  return 0
}

run_all() {
  # Args:
  # - run_sample function to invoke

  echo "Removing cache..."
  sleep 5
  rm -fr .dacecache

  $1 2mm k2mm 0
  $1 3mm k3mm 0
  $1 adi adi 0
  $1 atax atax 0
  $1 bicg bicg 0
  $1 cholesky cholesky 0
  $1 correlation correlation 0
  $1 covariance covariance 0
  $1 deriche deriche 0
  $1 doitgen doitgen 1
  $1 durbin durbin 0
  $1 fdtd-2d fdtd2d 0
  $1 floyd-warshall floyd_warshall 0
  $1 gemm gemm 0
  $1 gemver gemver 0
  $1 gesummv gesummv 0
  $1 gramschmidt gramschmidt 1
  $1 heat-3d heat3d 0
  $1 jacobi-1d jacobi1d 0
  $1 jacobi-2d jacobi2d 0
  $1 ludcmp ludcmp 0
  $1 lu lu 0
  $1 mvt mvt 0
  $1 nussinov nussinov 0
  $1 seidel-2d seidel2d 0
  $1 symm symm 1
  $1 syr2k syr2k 0
  $1 syrk syrk 0
  $1 trisolv trisolv 0
  $1 trmm trmm 1
}

if [ "$1" == "intel_fpga" ]; then
  # Check if aoc is vailable
  which aoc
  if [ $? -ne 0 ]; then
    echo "aocc not available"
    exit 99
  fi

  echo "====== Target: INTEL FPGA ======"
  export DACE_compiler_use_cache=0
  export DACE_compiler_fpga_vendor="intel_fpga"
  export DACE_compiler_intel_fpga_mode="emulator"

  TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
  cd $TEST_DIR/polybench

  run_all run_sample_intel

else
  # assuming xilinx
  # Check if xocc is vailable
  which xocc
  if [ $? -ne 0 ]; then
    echo "xocc not available"
    exit 99
  fi

  echo "====== Target: Xilinx ======"

  export DACE_compiler_use_cache=0
  export DACE_compiler_xilinx_mode="simulation"
  export DACE_compiler_fpga_vendor="xilinx"

  echo "Attention: this will cleanup cache in 5 seconds..."
  #cleanup
  rm -fr .dacecache/

  TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
  cd $TEST_DIR/polybench
  run_all run_sample_xilinx
fi

PASSED=$(expr $TESTS - $ERRORS)
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
  printf "Failed tests:\n${FAILED_TESTS}"
  exit 1
fi
