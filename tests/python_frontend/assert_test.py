# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests static and runtime assertions in dace programs. """
import dace
import pytest
import numpy as np


def test_static_assert():
    N = 5

    @dace.program
    def prog_static(A: dace.float64[20]):
        assert N > 1

    A = np.random.rand(20)
    prog_static(A)


def test_static_assert_fail():
    N = 5

    @dace.program
    def prog_static(A: dace.float64[20]):
        assert N > 5

    A = np.random.rand(20)
    with pytest.raises(AssertionError):
        prog_static(A)


def test_runtime_assert():
    @dace.program
    def prog_runtime(A: dace.float64[20]):
        assert A[0] >= 0

    A = np.random.rand(20)
    prog_runtime(A)


if __name__ == '__main__':
    test_static_assert()
    test_static_assert_fail()
    test_runtime_assert()
