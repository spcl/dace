#!/bin/bash
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

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

TIMEOUTCMD="timeout -s9 30"

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
        $TIMEOUTCMD $*
        return
    fi
    #$* | tee -a test.log
    TESTCNT=`expr $TESTS - 1`
    MSG="($TESTCNT / $TOTAL_TESTS) $CURTEST (Fails: $ERRORS)"
    ($TIMEOUTCMD $* || echo "_TFAIL_ $?") |& awk "BEGIN{printf \"$MSG\r\"} /_TFAIL_/{printf \"$TGAP\r\"; exit \$NF} {printf \"$TGAP\r\"; print; printf \"$MSG\r\";} END{printf \"$TGAP\r\"}"
}

################################################

runtest_py() {
    test_start $1
    yes | testcmd $PYTHON_BINARY $1 --size=mini
    if [ $? -ne 0 ]; then bail $1; fi
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

echo "====== Polybench Test Runner ======"

cd $SCRIPTPATH/polybench

# Specific test(s)
if [ $# -ne 0 ]; then
    TOTAL_TESTS=$#
    for arg in "$@"; do
        if [[ $arg == *.py ]]; then
            runtest_py $arg
        fi
    done
    endreport
    exit 0
fi

# Count tests first
counttests() {
    for file in *.py; do
        if [ $file == '*.py' ]; then break; fi # No files found
        TOTAL_TESTS=`expr $TOTAL_TESTS + 1`
    done
}

# Count tests in top-level folder
cd $SCRIPTPATH/polybench
counttests


################################################################

runtests() {
    for file in *.py; do
        if [ $file == '*.py' ]; then break; fi # No files found
        runtest_py $file
    done
}

cd $SCRIPTPATH/polybench
TESTPREFIX=""
runtests

endreport
