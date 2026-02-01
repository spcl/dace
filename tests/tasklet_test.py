# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math as mt
import numpy as np
import pytest
from dace.sdfg.validation import InvalidSDFGError


def test_decorator_syntax():

    @dace.program
    def myprint(input, N, M):

        @dace.tasklet
        def myprint():
            a << input
            for i in range(0, N):
                for j in range(0, M):
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

    with pytest.raises(InvalidSDFGError):
        tester.to_sdfg(simplify=False)


def test_invalid_scalar_access():

    @dace.program
    def tester(input: dace.float64, output: dace.float64[20]):
        with dace.tasklet:
            out >> output[0]
            out = input  # Invalid access, must go through memlet

    with pytest.raises(InvalidSDFGError):
        tester.to_sdfg(simplify=False)


def test_invalid_array_access_decorator_syntax():

    @dace.program
    def tester(input: dace.float64[20], output: dace.float64[20]):

        @dace.tasklet
        def tasklet():
            out >> output[0]
            out = input[1]  # Invalid access, must go through memlet

    with pytest.raises(InvalidSDFGError):
        tester.to_sdfg(simplify=False)


def test_invalid_scalar_access_decorator_syntax():

    @dace.program
    def tester(input: dace.float64, output: dace.float64[20]):

        @dace.tasklet
        def tasklet():
            out >> output[0]
            out = input  # Invalid access, must go through memlet

    with pytest.raises(InvalidSDFGError):
        tester.to_sdfg(simplify=False)


if __name__ == "__main__":
    test_decorator_syntax()
    test_invalid_array_access()
    test_invalid_scalar_access()
    test_invalid_array_access_decorator_syntax()
    test_invalid_scalar_access_decorator_syntax()
