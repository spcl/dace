# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests constants, optional, and keyword arguments. """
import dace
import numpy as np
import pytest
from dace.frontend.python.common import DaceSyntaxError


def test_kwargs():
    @dace.program
    def kwarg(A: dace.float64[20], kw: dace.float64[20]):
        A[:] = kw + 1

    A = np.random.rand(20)
    kw = np.random.rand(20)
    kwarg(A, kw=kw)
    assert np.allclose(A, kw + 1)


def test_kwargs_with_default():
    @dace.program
    def kwarg(A: dace.float64[20], kw: dace.float64[20] = np.ones([20])):
        A[:] = kw + 1

    # Call without argument
    A = np.random.rand(20)
    kwarg(A)
    assert np.allclose(A, 2.0)

    # Call with argument
    kw = np.random.rand(20)
    kwarg(A, kw)
    assert np.allclose(A, kw + 1)


if __name__ == '__main__':
    test_kwargs()
    test_kwargs_with_default()
