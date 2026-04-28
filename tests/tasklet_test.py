# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math as mt
import numpy as np
import pytest
from dace.frontend.python.common import DaceSyntaxError


def test_decorator_syntax():

    @dace.program
    def myprint(input, N, M):

        @dace.tasklet
        def myprint():
            a << input
            n << N
            m << M
            for i in range(0, n):
                for j in range(0, m):
                    mt.sin(a[i, j])

    input = dace.ndarray([10, 10], dtype=dace.float32)
    input[:] = np.random.rand(10, 10).astype(dace.float32.type)

    myprint(input, 10, 10)


def test_invalid_array_access():

    @dace.program
    def tester(input: dace.float64[20], output: dace.float64[20]):
        with dace.tasklet:
            out >> output[0]
            out = input[1]  # Invalid access, must go through memlet

    with pytest.raises(DaceSyntaxError):
        tester.to_sdfg()


def test_invalid_scalar_access():

    @dace.program
    def tester(input: dace.float64, output: dace.float64[20]):
        with dace.tasklet:
            out >> output[0]
            out = input  # Invalid access, must go through memlet

    with pytest.raises(DaceSyntaxError):
        tester.to_sdfg()


def test_invalid_array_access_decorator_syntax():

    @dace.program
    def tester(input: dace.float64[20], output: dace.float64[20]):

        @dace.tasklet
        def tasklet():
            out >> output[0]
            out = input[1]  # Invalid access, must go through memlet

    with pytest.raises(DaceSyntaxError):
        tester.to_sdfg()


def test_invalid_scalar_access_decorator_syntax():

    @dace.program
    def tester(input: dace.float64, output: dace.float64[20]):

        @dace.tasklet
        def tasklet():
            out >> output[0]
            out = input  # Invalid access, must go through memlet

    with pytest.raises(DaceSyntaxError):
        tester.to_sdfg()


def test_store_into_symbol_memlet():
    N = dace.symbol('N')

    @dace.program
    def tester(input: dace.int64[N]):

        @dace.tasklet
        def tasklet():
            inp << input[0]
            out = inp
            out >> N

    with pytest.raises(DaceSyntaxError, match="Symbolic variables"):
        tester.to_sdfg()


def test_store_into_symbol():
    N = dace.symbol('N')

    @dace.program
    def tester(input: dace.int64[N]):

        @dace.tasklet
        def tasklet():
            inp << input[0]
            N = inp  # Invalid access, cannot store into symbol

    with pytest.raises(DaceSyntaxError):
        tester.to_sdfg()


if __name__ == "__main__":
    test_decorator_syntax()
    test_invalid_array_access()
    test_invalid_scalar_access()
    test_invalid_array_access_decorator_syntax()
    test_invalid_scalar_access_decorator_syntax()
    test_store_into_symbol_memlet()
    test_store_into_symbol()
