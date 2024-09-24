# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests custom SDFG-convertible objects. """
import dace
import numpy as np
from dace.frontend.python.common import SDFGConvertible
from types import SimpleNamespace
import pytest


def test_daceprogram_constants_in_signature():
    @dace.program
    def convertible(grid: dace.compiletime, arr: dace.float64[10]):
        arr[grid.start:grid.end] = 7.0

    grid = SimpleNamespace(start=2, end=-2)

    @dace.program
    def program(grid: dace.compiletime, arr: dace.float64[10]):
        convertible(grid, arr)

    A = np.ones((10, ))
    program(grid, A)
    assert np.allclose(A[grid.start:grid.end], 7.0)


def test_constants_in_signature():
    class AConvertible(SDFGConvertible):
        def __sdfg__(self, grid, arr):
            @dace.program
            def func(_: dace.compiletime, arr: dace.float64[10]):
                arr[grid.start:grid.end] = 7.0

            return func.to_sdfg(grid, arr)

        def __sdfg_signature__(self):
            return (['grid', 'arr'], ['grid'])

        def __sdfg_closure__(self, reevaluate=None):
            return {}

    grid = SimpleNamespace(start=2, end=-2)
    convertible = AConvertible()

    @dace.program
    def program(grid: dace.compiletime, arr: dace.float64[10]):
        convertible(grid, arr)

    A = np.ones((10, ))
    program(grid, A)
    assert np.allclose(A[grid.start:grid.end], 7.0)


@pytest.mark.parametrize(('raise_error', 'nested_decorator'), [(False, True), (False, True)])
def test_nested_convertible_parse_fail(raise_error, nested_decorator):
    raised_exception = None

    class AConvertible(SDFGConvertible):
        def __call__(self, arr):
            nonlocal raised_exception
            raised_exception = FileNotFoundError('Expected')
            raise raised_exception

        def __sdfg__(self, arr):
            raise RuntimeError('Expected')

        def __sdfg_signature__(self):
            return (['arr'], [])

        def __sdfg_closure__(self, reevaluate=None):
            return {}

    convertible = AConvertible()

    def program2(arr):
        convertible(arr)

    if nested_decorator:
        program2 = dace.program(program2)

    @dace.program
    def program(arr: dace.float64[10]):
        program2(arr)

    A = np.ones((10, ))

    if raise_error:
        with dace.config.set_temporary('frontend', 'raise_nested_parsing_errors', value=True):
            with pytest.raises(RuntimeError):
                program(A)
    else:
        if nested_decorator:
            with pytest.raises(RuntimeError):
                program(A)
        else:
            with pytest.raises(FileNotFoundError):
                program(A)
                if raised_exception is not None:
                    raise raised_exception


if __name__ == '__main__':
    test_daceprogram_constants_in_signature()
    test_constants_in_signature()
    test_nested_convertible_parse_fail(False, False)
    test_nested_convertible_parse_fail(False, True)
    test_nested_convertible_parse_fail(True, False)
    test_nested_convertible_parse_fail(True, True)
