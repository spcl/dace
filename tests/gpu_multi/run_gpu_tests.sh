#!/bin/bash
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

cd ..
find . -name *.dacecache -type d -exec rm -rf {} +
find . -name *__pycache__ -type d -exec rm -rf {} +
export DACE_cache=single
export DACE_optimizer_interface=" "



pytest -o log_cli=1 -m gpu | tee gpu_multi/gpu_test.log

source ./cuda_test.sh | tee -a gpu_multi/gpu_tests.log