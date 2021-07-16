#!/bin/bash
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# options
red="reductions"
ignore_directories=("batchnorm")
ignore_files=("${red}/aCPU_aGPU_test.py" "${red}/aGPU_aCPU_test.py")
deselect_tests=("reductions/mGPU_CPU_test.py::test_multi_gpu_reduction_max")

# Cleanup in gpu_multi
for f in ".dacecache" "__pycache__"; do
    if [[ -d "$f" && ! -L "$f" ]]; then
        rm -rf $f
    fi
done

# Cleanup in subfolders of gpu_multi
for d in */ ; do
    for f in ".dacecache" "__pycache__"; do
        if [[ -d "$d$f" && ! -L "$d$f" ]]; then
            rm -rf $d$f
        fi
    done
done

# Compute ignore directories string
ignore_directories_string=""
for dir in ${ignore_directories[*]} ; do
    if [[ -d "$dir" && ! -L "$dir" ]]; then
        ignore_directories_string+="--ignore=$dir "
    fi
done

# Compute ignore glob string
ignore_files_string=""
for file in ${ignore_files[*]} ; do
    ignore_files_string+="--ignore=$file "
done
# for file in ${ignore_files[*]} ; do
#     ignore_files_string+="--ignore-glob='*$file' "
# done

# Compute deselect tests string
deselect_tests_string=""
for test in ${deselect_tests[*]} ; do
    deselect_tests_string+="--deselect=$test "
done
pystr="pytest ${ignore_directories_string} ${ignore_files_string} ${deselect_tests_string}"
echo $pystr
$pystr

