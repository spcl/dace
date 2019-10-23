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
# Check for generated bad code

badcode_check() {
    FILENAME=$1
    cuobjdump -sass $FILENAME > tmp.sass
    grep ' STL' tmp.sass
    if [ $? -eq 0 ]; then
        echo "Possible register spill found! (store)"
        rm -f tmp.sass
        return 1
    fi
    echo $OBJ | grep ' LDL'
    if [ $? -eq 0 ]; then
        echo "Possible register spill found! (load)"
        rm -f tmp.sass
        return 1
    fi
    rm -f tmp.sass
    echo "SUCCESS: Code contains no spills"
    return 0
}

find_128() {
    FILENAME=$1
    cuobjdump -sass $FILENAME | grep '\.128 '
    if [ $? -ne 0 ]; then
        echo "ERROR: 128-bit memory operations not found"
        return 1
    fi
    echo "SUCCESS: Code contains wide memory operations"
    return 0
}

################################################

checkoutput() {
    OBJFILE=$(get_latest_file ".dacecache")
    if [ $? -ne 0 ]; then
        echo "ERROR: Shared object file not found"
        return 2
    fi
    badcode_check $OBJFILE
}

check_vectorization() {
    OBJFILE=$(get_latest_file ".dacecache")
    if [ $? -ne 0 ]; then
        echo "ERROR: Shared object file not found"
        return 2
    fi
    find_128 $OBJFILE
}

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
        bail "$1 $2 ($3, optimized)"
        return 1
    fi
    
    checkoutput # Check for spills in the assembly
    if [ $? -ne 0 ]; then bail "$1 $2 ($3, assembly)"; return 1; fi
    return 0
}

runopt() {
    TESTS=`expr $TESTS + 1`
    opts=$(join_by_newline ${@:4})
    echo "$opts\ny" | timeout $TEST_TIMEOUT $1 $PYTHONPATH/$2
    if [ $? -ne 0 ]; then
        bail "$1 $2 ($3, optimized)"
        return 1
    fi
    
    checkoutput # Check for spills in the assembly
    if [ $? -ne 0 ]; then bail "$1 $2 ($3, assembly)"; return 1; fi
    return 0
}

runone() {
    echo "Running $1"
    runtestopt $1 cuda_grid_test.py $2
    runtestopt $1 cuda_grid_test.py $2 'GPUTransformMap$0'

    runtestopt $1 cuda_grid2d_test.py $2
    runtestopt $1 cuda_grid2d_test.py $2 'GPUTransformMap$0'
    
    runtestopt $1 cuda_grid_test.py $2 'GPUTransformMap$0' 'Vectorization$0'
    # Check that output was vectorized
    if [ $? -eq 0 ] && [ $DACE_optimizer_automatic_strict_transformations -ne 0 ]; then 
        check_vectorization
        if [ $? -ne 0 ]; then bail "$1 cuda_grid_test.py ($2, wideload)"; fi
    fi

    runtestopt $1 cuda_block_test.py $2
    runtestopt $1 cuda_block_test.py $2 'GPUTransformMap$0'

    runtestopt $1 cuda_smem_test.py $2
    runtestopt $1 cuda_smem_test.py $2 'GPUTransformMap$0'
    runtestopt $1 cuda_smem_test.py $2 'GPUTransformMap$0' 'InLocalStorage$0(array="gpu_A")'
    
    runtestopt $1 cuda_smem2d_test.py $2
    runtestopt $1 cuda_smem2d_test.py $2 'GPUTransformMap$0'
    runtestopt $1 cuda_smem2d_test.py $2 'GPUTransformMap$0' 'InLocalStorage$0(array="gpu_V")'
    
    runopt $1 samples/simple/sum.py $2
    runopt $1 samples/simple/sum.py $2 'GPUTransformMap$0'
    
    runtestopt $1 cuda_blockreduce.py $2 'GPUTransformMap$0'
    
    runtestopt $1 cuda_highdim_kernel_test.py $2 'GPUTransformMap$0(fullcopy=True)'
    
    runtestopt $1 multistream_copy_cudatest.py $2
    runtestopt $1 multistream_kernel_cudatest.py $2
    runtestopt $1 multistream_custom_cudatest.py $2

    runtestopt $1 multiprogram_cudatest.py $2
}

runall() {
    #runone python2 $1
    runone python3 $1
}

# Check if GPU tests can be run
nvidia-smi >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "GPUs not available or unusable"
    exit 99
fi


echo "====== Target: GPU ======"

DACE_compiler_use_cache=0

runall "GPU"

PASSED=`expr $TESTS - $ERRORS`
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
    printf "Failed tests:\n${FAILED_TESTS}"
    exit 1
fi
