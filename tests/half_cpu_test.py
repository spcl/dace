# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for half-precision (float16) syntax quirks on the CPU target.

The GPU counterparts live in ``half_cuda_test.py``; on the GPU path ``dace::half``
is the backend's native (default-constructible, IEEE-correct) half type. On the
CPU path it is a small fallback struct, so these tests guard that host-side
float16 default-construction, arithmetic, and fp32<->fp16 conversion are correct.
"""

import dace
import numpy as np

N = dace.symbol('N')

# Edge values covering zero, subnormals, the normal range, the max half, the
# overflow-to-inf boundary, inf, nan, and round-to-nearest-even halfway cases.
_EDGE_F32 = np.array(
    [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        3.140625,
        6.103515625e-05,  # smallest normal half (2^-14)
        6.0e-05,  # subnormal half
        3.0e-08,  # underflows to +0
        65504.0,  # max finite half
        65519.0,  # rounds down to 65504
        65520.0,  # rounds up to inf
        70000.0,  # overflow -> inf
        -70000.0,
        np.inf,
        -np.inf,
        np.nan,
        1.0009765625,  # exactly representable (1 + 2^-10)
        1.00048828125,  # exact halfway between 1.0 and 1+2^-10 -> round to even (1.0)
    ],
    dtype=np.float32,
)


def _bits(a):
    return a.view(np.uint16 if a.dtype == np.float16 else np.uint32)


def _to_half(a):
    # The edge values deliberately include out-of-range magnitudes (-> inf);
    # numpy's reference cast warns on those, which is expected here.
    with np.errstate(over='ignore'):
        return a.astype(np.float16)


def test_half_cast_from_float_matches_numpy():
    """float32 -> float16 conversion (dace::half(float)) is bit-exact with numpy,
    including zero, subnormals, overflow->inf, nan, and round-to-nearest-even."""

    @dace.program
    def cast_to_half(x: dace.float32[N], out: dace.float16[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << x[i]
                o >> out[i]
                o = a

    x = _EDGE_F32.copy()
    n = x.size
    out = np.zeros(n, dtype=np.float16)
    cast_to_half(x=x, out=out, N=n)

    expected = _to_half(x)
    nan = np.isnan(expected)
    assert np.all(np.isnan(out[nan]))  # nan stays nan
    assert np.array_equal(_bits(out[~nan]), _bits(expected[~nan]))  # everything else bit-exact


def test_half_cast_to_float_matches_numpy():
    """float16 -> float32 conversion (operator float()) is exact with numpy,
    including zero, subnormals, inf, and nan."""

    @dace.program
    def cast_from_half(a: dace.float16[N], out: dace.float32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                x << a[i]
                o >> out[i]
                o = x

    a = _to_half(_EDGE_F32)
    n = a.size
    out = np.zeros(n, dtype=np.float32)
    cast_from_half(a=a, out=out, N=n)

    expected = a.astype(np.float32)
    nan = np.isnan(expected)
    assert np.all(np.isnan(out[nan]))
    assert np.array_equal(_bits(out[~nan]), _bits(expected[~nan]))


def test_half_elementwise_add():
    """z = x + y on float16 arrays (host arithmetic emits a default-constructed
    temporary, which requires dace::half to be default-constructible)."""

    @dace.program
    def halfadd(x: dace.float16[N], y: dace.float16[N], z: dace.float16[N]):
        z[:] = x + y

    n = 24
    x = np.random.rand(n).astype(np.float16)
    y = np.random.rand(n).astype(np.float16)
    z = np.zeros(n, dtype=np.float16)
    halfadd(x=x, y=y, z=z, N=n)

    assert z.dtype == np.float16
    assert np.allclose(z.astype(np.float32), (x.astype(np.float32) + y.astype(np.float32)), atol=1e-2)


def test_half_return():
    """A float16 return value is allocated and returned as a float16 array."""

    @dace.program
    def halfinc(A: dace.float16[N]):
        return A + dace.float16(1.0)

    n = 20
    A = np.random.rand(n).astype(np.float16)
    out = halfinc(A=A, N=n)

    assert out.dtype == np.float16
    assert np.allclose(out.astype(np.float32), A.astype(np.float32) + 1.0, atol=1e-2)


def test_half_relu():
    """Relu over a float16 array via an explicit map/tasklet (CPU). Compares
    against float16(0), which requires the conversion of 0.0f to be correct."""

    @dace.program
    def halfrelu(A: dace.float16[N]):
        out = np.ndarray([N], dace.float16)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                o >> out[i]
                o = a if a > dace.float16(0) else dace.float16(0)
        return out

    n = 20
    A = (np.random.rand(n) - 0.5).astype(np.float16)  # mix of positive and negative
    out = halfrelu(A=A, N=n)

    assert out.dtype == np.float16
    assert np.allclose(out.astype(np.float32), np.maximum(A.astype(np.float32), 0), atol=1e-2)


if __name__ == '__main__':
    test_half_cast_from_float_matches_numpy()
    test_half_cast_to_float_matches_numpy()
    test_half_elementwise_add()
    test_half_return()
    test_half_relu()
