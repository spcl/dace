# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness of the experimental-CUDA thread-block tree reduction.

A scalar WCR accumulator written from a GPU thread-block map folds via one ``cub::BlockReduce``
+ one atomic per block (gated by ``compiler.tree_reduction``) instead of one atomic per thread.
These tests cover the base one-element-per-thread case, a sequential tile per thread
(multiple WCR writes per thread), a non-sum operator, and the atomic fallback -- checking both
bit-close results against numpy and that the ``cub::BlockReduce`` is (or is not) emitted."""
import numpy as np
import pytest

import dace
from dace.transformation.interstate import GPUTransformSDFG

pytestmark = pytest.mark.new_gpu_codegen_only

N = 128


def _gpu_sdfg(prog: dace.frontend.python.parser.DaceProgram) -> dace.SDFG:
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)
    return sdfg


@pytest.mark.gpu
def test_block_reduction_sum():
    """One element per thread: tree reduction emitted and bit-close to numpy."""

    @dace.program
    def blockred(A: dace.float64[N], s: dace.float64[1]):
        s[0] = 0.0
        for i in dace.map[0:N]:
            s[0] += A[i]

    sdfg = _gpu_sdfg(blockred)
    code = "\n".join(c.code for c in sdfg.generate_code())
    assert "cub::BlockReduce" in code

    A = np.random.rand(N)
    s = np.zeros(1)
    sdfg(A=A, s=s)
    assert np.allclose(s[0], A.sum(), rtol=0, atol=1e-9)


@pytest.mark.gpu
def test_block_reduction_sequential_tile():
    """Each thread accumulates a sequential tile (many WCR writes per thread) before the fold."""

    @dace.program
    def tiledred(A: dace.float64[8, N], s: dace.float64[1]):
        s[0] = 0.0
        for i in dace.map[0:N]:
            for j in range(8):
                s[0] += A[j, i]

    sdfg = _gpu_sdfg(tiledred)
    code = "\n".join(c.code for c in sdfg.generate_code())
    assert "cub::BlockReduce" in code

    A = np.random.rand(8, N)
    s = np.zeros(1)
    sdfg(A=A, s=s)
    assert np.allclose(s[0], A.sum(), rtol=0, atol=1e-9)


@pytest.mark.gpu
def test_block_reduction_max():
    """A non-sum built-in operator (max via an explicit WCR out-connector) also block-reduces."""

    @dace.program
    def maxred(A: dace.float64[N], s: dace.float64[1]):
        s[0] = -1e30
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                o >> s(1, lambda x, y: max(x, y))[0]
                o = a

    sdfg = _gpu_sdfg(maxred)
    code = "\n".join(c.code for c in sdfg.generate_code())
    assert "cub::BlockReduce" in code

    A = np.random.rand(N)
    s = np.zeros(1)
    sdfg(A=A, s=s)
    assert np.allclose(s[0], A.max(), rtol=0, atol=1e-12)


@pytest.mark.gpu
def test_block_reduction_subset():
    """A length-m subset (vector) accumulator folds element-wise via a per-element for-loop of
    ``cub::BlockReduce`` instead of one atomic per thread per element."""
    M = 4

    @dace.program
    def vecred(B: dace.float64[N, M], acc: dace.float64[M]):
        acc[:] = 0.0
        for i in dace.map[0:N]:
            acc[:] += B[i, :]

    sdfg = _gpu_sdfg(vecred)
    code = "\n".join(c.code for c in sdfg.generate_code())
    assert "cub::BlockReduce" in code

    B = np.random.rand(N, M)
    acc = np.zeros(M)
    sdfg(B=B, acc=acc)
    assert np.allclose(acc, B.sum(axis=0), rtol=0, atol=1e-9)


@pytest.mark.gpu
def test_block_reduction_disabled_falls_back_to_atomic():
    """With ``compiler.tree_reduction`` off, no ``cub::BlockReduce`` is emitted, but the plain
    per-thread atomic WCR still produces the correct result."""

    @dace.program
    def blockred(A: dace.float64[N], s: dace.float64[1]):
        s[0] = 0.0
        for i in dace.map[0:N]:
            s[0] += A[i]

    with dace.config.set_temporary('compiler', 'tree_reduction', value=False):
        sdfg = _gpu_sdfg(blockred)
        code = "\n".join(c.code for c in sdfg.generate_code())
        assert "cub::BlockReduce" not in code

        A = np.random.rand(N)
        s = np.zeros(1)
        sdfg(A=A, s=s)
        assert np.allclose(s[0], A.sum(), rtol=0, atol=1e-9)


if __name__ == '__main__':
    test_block_reduction_sum()
    test_block_reduction_sequential_tile()
    test_block_reduction_max()
    test_block_reduction_subset()
    test_block_reduction_disabled_falls_back_to_atomic()
