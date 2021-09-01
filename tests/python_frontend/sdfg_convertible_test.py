# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests custom SDFG-convertible objects. """
import dace
import numpy as np
from dace.frontend.python.common import SDFGConvertible
from types import SimpleNamespace


def test_daceprogram_constants_in_signature():
    @dace.program
    def convertible(grid: dace.constant, arr: dace.float64[10]):
        arr[grid.start:grid.end] = 7.0

    grid = SimpleNamespace(start=2, end=-2)

    @dace.program
    def program(grid: dace.constant, arr: dace.float64[10]):
        convertible(grid, arr)

    A = np.ones((10, ))
    program(grid, A)
    assert np.allclose(A[grid.start:grid.end], 7.0)


def test_constants_in_signature():
    class AConvertible(SDFGConvertible):
        def __sdfg__(self, grid, arr):
            @dace.program
            def func(arr: dace.float64[10]):
                arr[grid.start:grid.end] = 7.0

            return func.to_sdfg(grid, arr)

        def __sdfg_signature__(self):
            return (['grid', 'arr'], ['grid'])

        def __sdfg_closure__(self, reevaluate=None):
            return {}

    grid = SimpleNamespace(start=2, end=-2)
    convertible = AConvertible()

    @dace.program
    def program(grid: dace.constant, arr: dace.float64[10]):
        convertible(grid, arr)

    A = np.ones((10, ))
    program(grid, A)
    assert np.allclose(A[grid.start:grid.end], 7.0)


if __name__ == '__main__':
    test_daceprogram_constants_in_signature()
    test_constants_in_signature()
