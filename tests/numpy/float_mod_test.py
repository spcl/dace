# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Float ``%`` follows Python: the sign of the denominator, exact remainders, and a zero that takes that sign."""

import math

import numpy as np
import pytest

import dace

N = 12


@dace.program
def float_mod(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        out[i] = a[i] % b[i]


@dace.program
def float_mod32(a: dace.float32[N], b: dace.float32[N], out: dace.float32[N]):
    for i in range(N):
        out[i] = a[i] % b[i]


A = [-0.0, 0.0, -6.0, 6.0, -7.5, 7.5, 1.0, 1e20, 5.0, -5.0, 5.0, -0.0]
B = [3.0, -3.0, 3.0, -3.0, 2.0, -2.0, 0.1, 3.0, math.inf, math.inf, -math.inf, -3.0]


@pytest.mark.parametrize("program, dtype", [(float_mod, np.float64), (float_mod32, np.float32)])
def test_float_mod_matches_python(program, dtype):
    a, b, out = np.array(A, dtype=dtype), np.array(B, dtype=dtype), np.zeros(N, dtype=dtype)
    program(a=a, b=b, out=out)
    expected = np.array([float(x) % float(y) for x, y in zip(a, b)], dtype=dtype)
    assert np.array_equal(out, expected)
    assert np.array_equal(np.signbit(out), np.signbit(expected))


if __name__ == "__main__":
    test_float_mod_matches_python(float_mod, np.float64)
    test_float_mod_matches_python(float_mod32, np.float32)
