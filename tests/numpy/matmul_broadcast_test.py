# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Broadcast batch dimensions in ``np.matmul``.

NumPy aligns batch dimensions to the RIGHT and stretches any of extent 1. The pure expansion of
``BatchedMatMul`` reproduces that by reading a stretched dimension at 0. Reading it at the output
batch index instead -- or aligning from the left -- runs off the end of the smaller operand, which
surfaced as an out-of-bounds memlet on ``(B,1,M,K) @ (1,C,K,N)`` and, for a rank mismatch, as a
silently wrong result.
"""
import numpy as np
import pytest

import dace


def run(prog, expected, **arrays):
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.expand_library_nodes()
    res = np.zeros_like(expected)
    sdfg(**arrays, res=res)
    assert res.shape == expected.shape, f'shape {res.shape} != numpy {expected.shape}'
    assert np.allclose(res, expected), f'max|diff| = {np.max(np.abs(res - expected))}'


def test_broadcast_both_sides():
    """``(B,1,M,K) @ (1,C,K,N) -> (B,C,M,N)``: each side stretches a different axis."""

    @dace.program
    def kernel(a: dace.float64[2, 1, 8, 5], b: dace.float64[1, 3, 5, 4], res: dace.float64[2, 3, 8, 4]):
        res[:] = np.matmul(a, b)

    a = np.random.rand(2, 1, 8, 5)
    b = np.random.rand(1, 3, 5, 4)
    run(kernel, a @ b, a=a, b=b)


def test_broadcast_leading_one():
    """``(1,M,K) @ (B,K,N)``: the single matrix is reused for every batch element."""

    @dace.program
    def kernel(a: dace.float64[1, 8, 5], b: dace.float64[4, 5, 6], res: dace.float64[4, 8, 6]):
        res[:] = np.matmul(a, b)

    a = np.random.rand(1, 8, 5)
    b = np.random.rand(4, 5, 6)
    run(kernel, a @ b, a=a, b=b)


def test_broadcast_rank_mismatch():
    """``(B,C,M,K) @ (C,K,N)``: right-alignment makes the shorter batch line up with C, not B.

    Aligning from the left instead produced the right shape and wrong numbers -- a silent
    miscompile, which is why this asserts values and not just the shape.
    """

    @dace.program
    def kernel(a: dace.float64[2, 3, 8, 5], b: dace.float64[3, 5, 4], res: dace.float64[2, 3, 8, 4]):
        res[:] = np.matmul(a, b)

    a = np.random.rand(2, 3, 8, 5)
    b = np.random.rand(3, 5, 4)
    run(kernel, a @ b, a=a, b=b)


def test_equal_batches_still_work():
    """The already-supported path must be unchanged by broadcast handling."""

    @dace.program
    def kernel(a: dace.float64[3, 8, 5], b: dace.float64[3, 5, 4], res: dace.float64[3, 8, 4]):
        res[:] = np.matmul(a, b)

    a = np.random.rand(3, 8, 5)
    b = np.random.rand(3, 5, 4)
    run(kernel, a @ b, a=a, b=b)


def test_incompatible_batches_refused():
    """Dimensions that neither match nor are 1 have no broadcast, so they must be refused."""
    from dace.frontend.python.common import DaceSyntaxError

    @dace.program
    def kernel(a: dace.float64[3, 8, 5], b: dace.float64[2, 5, 4], res: dace.float64[3, 8, 4]):
        res[:] = np.matmul(a, b)

    with pytest.raises((DaceSyntaxError, ValueError), match='broadcast'):
        kernel.to_sdfg(simplify=True)


def _batch_check(prog):
    """The fixed-stride lowerings' broadcast check, bound to the program's matmul node.

    The frontend emits the generic ``MatMul``; it is ``BatchedMatMul``'s MKL / cuBLAS expansions
    that run this check, and it reads the operands off the node either way."""
    from dace.libraries.blas.nodes.matmul import MatMul
    from dace.libraries.blas.nodes.batched_matmul import refuse_broadcast_batches

    sdfg = prog.to_sdfg(simplify=True)
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, MatMul):
                return lambda: refuse_broadcast_batches(node, state, sdfg)
    raise AssertionError('the parsed SDFG carries no matmul node to check')


def test_strided_lowering_refuses_a_stretched_batch():
    """Extent 1 against extent 3 is a real broadcast: one fixed stride cannot express it."""

    @dace.program
    def kernel(a: dace.float64[1, 8, 5], b: dace.float64[3, 5, 4], res: dace.float64[3, 8, 4]):
        res[:] = np.matmul(a, b)

    with pytest.raises(ValueError, match='broadcast'):
        _batch_check(kernel)()


def test_strided_lowering_accepts_distinct_batch_symbols():
    """``[B,M,K] @ [L,K,N]`` is not a broadcast -- numpy pairs the two only when they are equal at
    run time, which is what the result's batch shape already assumes. Refusing it sends every
    symbolically-batched matmul to a failed expansion (it did, under auto-optimization)."""
    B, L, M, K, N = (dace.symbol(s) for s in ('B', 'L', 'M', 'K', 'N'))

    @dace.program
    def kernel(a: dace.float64[B, M, K], b: dace.float64[L, K, N], res: dace.float64[B, M, N]):
        res[:] = np.matmul(a, b)

    _batch_check(kernel)()  # must not raise

    a = np.random.rand(3, 8, 5)
    b = np.random.rand(3, 5, 4)
    res = np.zeros((3, 8, 4))
    sdfg = kernel.to_sdfg(simplify=True)
    sdfg.expand_library_nodes()
    sdfg(a=a, b=b, res=res, B=3, L=3, M=8, K=5, N=4)
    assert np.allclose(res, a @ b), f'max|diff| = {np.max(np.abs(res - a @ b))}'


if __name__ == '__main__':
    test_broadcast_both_sides()
    test_broadcast_leading_one()
    test_broadcast_rank_mismatch()
    test_equal_batches_still_work()
    test_incompatible_batches_refused()
    test_strided_lowering_refuses_a_stretched_batch()
    test_strided_lowering_accepts_distinct_batch_symbols()
