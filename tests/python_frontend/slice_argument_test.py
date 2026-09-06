# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Passing a slice of an array as an argument to another ``@dace.program``.

The callee is parsed on its own, so its parameters are described as contiguous arrays of the
declared shape. The call site then hands it a slice of a larger container, which reaches the callee
as a view: the argument keeps its shape but takes the enclosing container's strides, and the nested
SDFG contract asks that every connector standing for it be restated the same way -- inside the
callee, and in the view the call site leaves behind.
"""
import numpy as np

import dace

I, J, K = 4, 3, 6


@dace.program
def _scale(src: dace.float64[I, J, K - 1], dst: dace.float64[I, J, K - 1]):
    for i, j in dace.map[0:I, 0:J]:
        for k in range(K - 1):
            dst[i, j, k] = src[i, j, k] * 2.0 + 1.0


@dace.program
def _scale2d(src: dace.float64[J, K], dst: dace.float64[J, K]):
    for j in dace.map[0:J]:
        for k in range(K):
            dst[j, k] = src[j, k] * 2.0 + 1.0


@dace.program
def sliced_input(q: dace.float64[I, J, K], out: dace.float64[I, J, K - 1]):
    _scale(q[:, :, 1:], out)


@dace.program
def sliced_output(q: dace.float64[I, J, K - 1], out: dace.float64[I, J, K]):
    _scale(q, out[:, :, :K - 1])


@dace.program
def sliced_both(q: dace.float64[I, J, K], out: dace.float64[I, J, K]):
    _scale(q[:, :, :K - 1], out[:, :, 1:])


@dace.program
def sliced_plane(q: dace.float64[I, J, K], out: dace.float64[I, J, K]):
    _scale2d(q[0], out[1])


def test_sliced_input():
    q = np.random.rand(I, J, K)
    out = np.zeros((I, J, K - 1))
    sliced_input(q, out)
    assert np.allclose(out, q[:, :, 1:] * 2.0 + 1.0)


def test_sliced_output():
    q = np.random.rand(I, J, K - 1)
    out = np.zeros((I, J, K))
    sliced_output(q, out)
    assert np.allclose(out[:, :, :K - 1], q * 2.0 + 1.0)
    assert np.allclose(out[:, :, K - 1], 0.0)


def test_sliced_both():
    q = np.random.rand(I, J, K)
    out = np.zeros((I, J, K))
    sliced_both(q, out)
    assert np.allclose(out[:, :, 1:], q[:, :, :K - 1] * 2.0 + 1.0)
    assert np.allclose(out[:, :, 0], 0.0)


def test_sliced_plane():
    """A slice that drops a dimension, so the view is one the caller squeezes."""
    q = np.random.rand(I, J, K)
    out = np.zeros((I, J, K))
    sliced_plane(q, out)
    assert np.allclose(out[1], q[0] * 2.0 + 1.0)
    assert np.allclose(out[0], 0.0)
    assert np.allclose(out[2:], 0.0)


def test_sliced_both_unvalidated():
    """The same call with validation turned off, as orchestrated callers (e.g. NDSL) build it.

    Without validation a connector that has drifted off its container is not reported: the write
    simply lands somewhere else, so this checks the numbers rather than the graph.
    """
    sdfg = sliced_both.to_sdfg(validate=False)
    csdfg = sdfg.compile(validate=False)

    q = np.random.rand(I, J, K)
    out = np.zeros((I, J, K))
    csdfg(q=q, out=out)
    assert np.allclose(out[:, :, 1:], q[:, :, :K - 1] * 2.0 + 1.0)
    assert np.allclose(out[:, :, 0], 0.0)


if __name__ == '__main__':
    test_sliced_input()
    test_sliced_output()
    test_sliced_both()
    test_sliced_plane()
    test_sliced_both_unvalidated()
