#!/bin/bash
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

pytest --cov-report=xml --cov=dace --tb=short -m "not gpu and not verilator and not tensorflow"
if [ "$?" -ne '0' ]; then
    exit 127
fi

pytest --cov-report=xml --cov=dace --tb=short -m "not gpu and not verilator and not tensorflow"
if [ "$?" -ne '0' ]; then
    exit 127
fi

pytest --cov-report=xml --cov=dace --tb=short -m "not gpu and not verilator and not tensorflow"
if [ "$?" -ne '0' ]; then
    exit 127
fi

pytest --cov-report=xml --cov=dace --tb=short -m "not gpu and not verilator and not tensorflow"
if [ "$?" -ne '0' ]; then
    exit 127
fi

pytest --cov-report=xml --cov=dace --tb=short -m "not gpu and not verilator and not tensorflow"
if [ "$?" -ne '0' ]; then
    exit 127
fi
