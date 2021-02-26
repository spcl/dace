#!/bin/bash
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

cp /tmp/gemmz_test.py tests/library/gemmz_test.py

for i in `seq 1 10`; do

    pytest --cov-report=xml --cov=dace --tb=short -m "not gpu and not verilator and not tensorflow"
    if [ "$?" -ne '0' ]; then
        cat .dacecache/test/src/program.cpp
        exit 127
    fi

done
