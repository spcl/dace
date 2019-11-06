#!/bin/bash

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHONPATH=$SCRIPTPATH/..

DACE_debugprint="${DACE_debugprint:-0}"
DACE_optimizer_automatic_strict_transformations=${DACE_optimizer_automatic_strict_transformations:-1}
ERRORS=0
FAILED_TESTS=""
TESTS=0

TEST_TIMEOUT=60

RED='\033[0;31m'
NC='\033[0m'

# From http://mywiki.wooledge.org/BashFAQ/003
get_latest_file() {
    dir=$1
    unset -v latest
    for file in "$dir"/*/build/*.so; do
        [[ $file -nt $latest ]] && latest=$file
    done
    # If there is no such file, return with an error
    if [ -z "$latest" ]; then
        return 2
    fi
    echo $latest
}

join_by_newline() {
    for a in $*; do
        echo $a        
    done
    echo 9999
}

################################################

bail() {
    ERRORSTR=$1
    /bin/echo -e "${RED}ERROR${NC} in $ERRORSTR" 1>&2
    ERRORS=`expr $ERRORS + 1`
    FAILED_TESTS="${FAILED_TESTS} $ERRORSTR\n"
}

runtestopt() {
    TESTS=`expr $TESTS + 1`
    opts=$(join_by_newline ${@:4})
    echo "$opts\ny" | timeout $TEST_TIMEOUT $1 $PYTHONPATH/tests/$2
    if [ $? -ne 0 ]; then
        bail "$1 $2 ($3, MPI)"
        return 1
    fi
    
    return 0
}

runone() {
    echo "Running $1"
    
    # Use cache (do not recompile every rank) in distributed test
    runtestopt $1 immaterial_test.py $2 'MPITransformMap$0'
    DACE_compiler_use_cache=1
    runtestopt "mpirun -np 4 $1" immaterial_test.py
    DACE_compiler_use_cache=0
    
    runtestopt $1 immaterial_range_test.py $2 'MPITransformMap$0'
    DACE_compiler_use_cache=1
    runtestopt "mpirun -np 4 $1" immaterial_range_test.py
    DACE_compiler_use_cache=0
}

which mpirun
if [ $? -ne 0 ]; then
    echo "MPI not available"
    exit 99
fi

DACE_compiler_use_cache=0

runone python3 "MPI"

PASSED=`expr $TESTS - $ERRORS`
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
    printf "Failed tests:\n${FAILED_TESTS}"
    exit 1
fi
