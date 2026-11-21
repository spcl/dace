# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A nested program's typed array parameter must not be bound to an array of another element type.

The binding used to become a view that REINTERPRETS the caller's buffer: a float32 array read and written as float64,
twice its size -- mixed_precision_ir's float32 LU factors overran their buffer and corrupted the heap.
"""

import numpy as np
import pytest

import dace
from dace.frontend.python.common import DaceSyntaxError

N = dace.symbol("N")


@dace.program
def fill(a: dace.float64[N]):
    for i in range(N):
        a[i] = 2.0 * i


@dace.program
def fills_a_float32_buffer(out: dace.float64[N]):
    t = np.zeros((N,), dtype=np.float32)
    fill(t)
    out[:] = t


@dace.program
def fills_a_float64_buffer(out: dace.float64[N]):
    t = np.zeros((N,), dtype=np.float64)
    fill(t)
    out[:] = t


def test_an_array_of_another_element_type_is_refused():
    with pytest.raises(DaceSyntaxError, match="declared float64 but"):
        fills_a_float32_buffer.to_sdfg()


def test_an_array_of_the_declared_element_type_is_bound():
    out = np.zeros(6)
    fills_a_float64_buffer(out=out, N=6)
    assert np.array_equal(out, 2.0 * np.arange(6))


if __name__ == "__main__":
    test_an_array_of_another_element_type_is_refused()
    test_an_array_of_the_declared_element_type_is_bound()
