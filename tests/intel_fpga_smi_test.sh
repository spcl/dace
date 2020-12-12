#!/bin/bash

# Executes a bunch of small test for the Intel FPGA backend
# These are not intended to be performance test: they just check that everything compiles
# and the correct result is produced

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHONPATH=$SCRIPTPATH/..
PYTHON_BINARY="${PYTHON_BINARY:-python3}"

DACE_debugprint="${DACE_debugprint:-0}"
ERRORS=0
FAILED_TESTS=""
TESTS=0
TEST_TIMEOUT=10
TESTS_DIR=intel_fpga_smi/

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

run_sample0() {
    TEST_NAME=smi_sample0
    TESTS=`expr $TESTS + 1`
    echo -e "${YELLOW}Running test SMI sample 0...${NC}"

    #0: cleanup
    rm -fr .dacecache/${TEST_NAME}*
    rm *emulated_channel* 2> /dev/null

    #1: generate the opencl for both ranks

    # (Provisional) We use mpi4py
    # TODO: remove this dependency
    mpirun -n 2  $PYTHON_BINARY ${TESTS_DIR}/${TEST_NAME}.py

    #2: compile both for emulation
    # Sender
    cd .dacecache/${TEST_NAME}_sender/build
    make intelfpga_smi_compile_${TEST_NAME}_sender_emulator_0 -j4
    if [ $? -ne 0 ]; then
      bail "$1 (${RED}high-level synthesis failed (${TEST_NAME}_sender)${NC})"
      return 1
    fi
    make intelfpga_smi_${TEST_NAME}_sender_codegen_host
    if [ $? -ne 0 ]; then
      bail "$1 (${RED}Compilation of host program failed (${TEST_NAME}_sender)${NC})"
      return 1
    fi
    cd -
    #receiver
    cd .dacecache/${TEST_NAME}_receiver/build
    make intelfpga_smi_compile_${TEST_NAME}_receiver_emulator_1 -j4
    if [ $? -ne 0 ]; then
      bail "$1 (${RED}high-level synthesis failed (${TEST_NAME}_receiver)${NC})"
      return 1
    fi
    make intelfpga_smi_${TEST_NAME}_receiver_codegen_host
    if [ $? -ne 0 ]; then
      bail "$1 (${RED}Compilation of host program failed (${TEST_NAME}_receiver)${NC})"
      return 1
    fi
    cd -

    #3: execute the emulation
    mpirun -n 2  $PYTHON_BINARY ${TESTS_DIR}/${TEST_NAME}.py

    if [ $? -ne 0 ]; then
        bail "$1 (${RED}Wrong emulation result${NC})"
    fi

    cd -
    return 0
}


run_all() {
    #Axpy with ping pong between sender receiver
    run_sample0


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
DACE_compiler_intel_fpga_board="p520_max_sg280l"

TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $TEST_DIR
run_all ${1:-"0"}

PASSED=`expr $TESTS - $ERRORS`
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
    printf "Failed tests:\n${FAILED_TESTS}"
    exit 1
fi
