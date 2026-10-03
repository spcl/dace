# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Batched ``numpy.matmul``: the >= 3-D operand forms, which lower to the ``BatchedMatMul`` node.

Two things are pinned here. Batches that the node can express must be computed, with NumPy's output
shape and not just NumPy's numbers. Batches it cannot express -- a leading dimension broadcast from
1, or batch ranks that differ -- must be refused, because the node walks the batch at a fixed stride
and would otherwise read the smaller operand out of bounds and return a plausible wrong answer.
"""
import numpy as np
import pytest

import dace
from dace.frontend.python.common import DaceSyntaxError
from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul

BATCH, BATCH2, M, K, N = 3, 2, 8, 5, 4

rng = np.random.default_rng(0)


@dace.program
def bmm_3d_3d(a: dace.float64[BATCH, M, K], b: dace.float64[BATCH, K, N]):
    return np.matmul(a, b)


@dace.program
def bmm_4d_4d(a: dace.float64[BATCH2, BATCH, M, K], b: dace.float64[BATCH2, BATCH, K, N]):
    return np.matmul(a, b)


@dace.program
def bmm_3d_2d(a: dace.float64[BATCH, M, K], b: dace.float64[K, N]):
    return np.matmul(a, b)


@dace.program
def bmm_2d_3d(a: dace.float64[M, K], b: dace.float64[BATCH, K, N]):
    return np.matmul(a, b)


@dace.program
def bmm_per_batch_slice(a: dace.float64[BATCH, M, K], b: dace.float64[BATCH, K, N], c: dace.float64[BATCH, M, N]):
    for __bm0 in dace.map[0:BATCH]:
        c[__bm0] = np.matmul(a[__bm0], b[__bm0])


def test_batched_matmul_3d_3d():
    a, b = rng.random((BATCH, M, K)), rng.random((BATCH, K, N))
    c = bmm_3d_3d(a, b)
    assert c.shape == (BATCH, M, N)
    assert np.allclose(c, np.matmul(a, b))


def test_batched_matmul_4d_4d():
    a, b = rng.random((BATCH2, BATCH, M, K)), rng.random((BATCH2, BATCH, K, N))
    c = bmm_4d_4d(a, b)
    assert c.shape == (BATCH2, BATCH, M, N)
    assert np.allclose(c, np.matmul(a, b))


def test_batched_matmul_3d_2d():
    """A plain matrix on the right is reused for every batch element -- NumPy's degenerate broadcast."""
    a, b = rng.random((BATCH, M, K)), rng.random((K, N))
    c = bmm_3d_2d(a, b)
    assert c.shape == (BATCH, M, N)
    assert np.allclose(c, np.matmul(a, b))


def test_batched_matmul_2d_3d():
    a, b = rng.random((M, K)), rng.random((BATCH, K, N))
    c = bmm_2d_3d(a, b)
    assert c.shape == (BATCH, M, N)
    assert np.allclose(c, np.matmul(a, b))


def test_batched_matmul_per_batch_slice():
    """The already-sliced spelling: 2-D operands under a map, one GEMM per batch element."""
    a, b = rng.random((BATCH, M, K)), rng.random((BATCH, K, N))
    c = np.zeros((BATCH, M, N))
    bmm_per_batch_slice(a, b, c)
    assert c.shape == (BATCH, M, N)
    assert np.allclose(c, np.matmul(a, b))


def test_batched_matmul_lowers_to_batched_matmul_node():
    """The point of the batched path: it must reach ``BatchedMatMul``, not a loop of GEMMs."""
    sdfg = bmm_3d_3d.to_sdfg(simplify=True)
    sdfg.expand_library_nodes(recursive=False)
    assert any(isinstance(node, BatchedMatMul) for node, _ in sdfg.all_nodes_recursive())


def test_broadcast_batch_dimension():
    """``(B, 1, M, K) @ (1, C, K, N)`` is NumPy-legal and now lowered, not refused.

    The pure expansion reads a stretched dimension at 0; only the fixed-stride BLAS expansions
    cannot express it, and they refuse for themselves.
    """

    @dace.program
    def bmm_broadcast(a: dace.float64[BATCH2, 1, M, K], b: dace.float64[1, BATCH, K, N]):
        return np.matmul(a, b)

    a = np.random.rand(BATCH2, 1, M, K)
    b = np.random.rand(1, BATCH, K, N)
    res = bmm_broadcast(a=a, b=b)
    expected = a @ b
    assert res.shape == expected.shape
    assert np.allclose(res, expected)


def test_mismatched_batch_rank():
    """``(B, C, M, K) @ (C, K, N)``: NumPy right-aligns the shorter batch, so this is legal.

    It used to produce the correct SHAPE with wrong numbers -- a silent miscompile -- because the
    operand was indexed with the leading output batch index instead of the right-aligned one.
    """

    @dace.program
    def bmm_rank(a: dace.float64[BATCH2, BATCH, M, K], b: dace.float64[BATCH, K, N]):
        return np.matmul(a, b)

    a = np.random.rand(BATCH2, BATCH, M, K)
    b = np.random.rand(BATCH, K, N)
    res = bmm_rank(a=a, b=b)
    expected = a @ b
    assert res.shape == expected.shape
    assert np.allclose(res, expected), f'max|diff| = {np.max(np.abs(res - expected))}'


def test_unequal_batch_size_refused():
    """The plain disagreement, which used to reach the library node as a wrong-sized batch."""

    @dace.program
    def bmm_unequal(a: dace.float64[BATCH, M, K], b: dace.float64[BATCH2, K, N]):
        return np.matmul(a, b)

    with pytest.raises(DaceSyntaxError, match='do not broadcast'):
        bmm_unequal.to_sdfg()


if __name__ == '__main__':
    test_batched_matmul_3d_3d()
    test_batched_matmul_4d_4d()
    test_batched_matmul_3d_2d()
    test_batched_matmul_2d_3d()
    test_batched_matmul_per_batch_slice()
    test_batched_matmul_lowers_to_batched_matmul_node()
    test_broadcast_batch_dimension()
    test_mismatched_batch_rank()
    test_unequal_batch_size_refused()
