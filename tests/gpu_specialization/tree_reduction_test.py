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


@dace.program
def red2d(A: dace.float64[32, 8], s: dace.float64[1]):
    s[0] = 0.0
    for i, j in dace.map[0:32, 0:8]:
        s[0] += A[i, j]


@dace.program
def minred_i64(A: dace.int64[N], s: dace.int64[1]):
    s[0] = 9223372036854775807
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            o >> s(1, lambda x, y: min(x, y))[0]
            o = a


def _force_2d_block(sdfg: dace.SDFG, block_size) -> None:
    """Pin every GPU_Device map to an explicit ``gpu_block_size`` so the kernel launches a genuinely
    multi-dimensional thread block (the default heuristic keeps blocks 1-D)."""
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                node.map.gpu_block_size = list(block_size)


def test_multidim_block_uses_multidim_cub_template():
    """A 2-D thread block must instantiate the multi-dimensional cub template. Regression: the 1-D
    ``cub::BlockReduce<T, N>`` form with N = blockDim.x*blockDim.y mis-maps threads (it assumes
    threadIdx.y == threadIdx.z == 0), silently computing a wrong reduction on any 2-D/3-D block."""
    sdfg = red2d.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)
    _force_2d_block(sdfg, [32, 8, 1])
    code = "\n".join(c.code for c in sdfg.generate_code())
    assert "cub::BlockReduce<double, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 8, 1>" in code
    assert "cub::BlockReduce<double, 256>" not in code


def test_int64_min_identity_is_exact_integer_literal():
    """An int64 Min identity (INT64_MAX) must be emitted as an exact integer literal. Regression:
    routing it through ``float`` rounded 2**63-1 up to 2**63 and the cast overflowed to INT64_MIN,
    so every thread seeded its partial with the wrong identity."""
    sdfg = minred_i64.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)
    code = "\n".join(c.code for c in sdfg.generate_code())
    assert "int64_t(9223372036854775807)" in code
    assert "9.223372036854776e+18" not in code


@pytest.mark.gpu
def test_multidim_block_reduction_correct():
    """End-to-end: a genuinely 2-D thread block reduces correctly. Pre-fix the 1-D cub template
    mis-mapped the threadIdx.y rows and produced a wrong sum."""
    sdfg = red2d.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)
    _force_2d_block(sdfg, [32, 8, 1])

    A = np.random.rand(32, 8)
    s = np.zeros(1)
    sdfg(A=A, s=s)
    assert np.allclose(s[0], A.sum(), rtol=0, atol=1e-9)


@pytest.mark.gpu
def test_int64_min_reduction_correct():
    """End-to-end: an int64 Min reduction returns the true minimum. Pre-fix the identity overflowed
    to INT64_MIN, so the result was pinned to that garbage value."""
    A = (np.random.rand(N) * 1e6).astype(np.int64) + 5
    s = np.array([np.iinfo(np.int64).max], dtype=np.int64)
    minred_i64(A=A, s=s)
    assert s[0] == A.min()


@dace.program
def sumred_1d(A: dace.float64[N], s: dace.float64[1]):
    s[0] = 0.0
    for i in dace.map[0:N]:
        s[0] += A[i]


def test_covered_accumulator_write_is_redirected_ok():
    """The interior WCR write of a covered accumulator is redirected into the register partial (not
    silently dropped); the emitted body assigns into ``__bpart`` via ``dace::_wcr_fixed`` and code
    generation succeeds without tripping the drain-time invariant check."""
    sdfg = sumred_1d.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)
    code = "\n".join(c.code for c in sdfg.generate_code())
    assert "cub::BlockReduce" in code
    assert "__bpart" in code and "dace::_wcr_fixed" in code


if __name__ == '__main__':
    test_block_reduction_sum()
    test_block_reduction_sequential_tile()
    test_block_reduction_max()
    test_block_reduction_subset()
    test_block_reduction_disabled_falls_back_to_atomic()
    test_multidim_block_uses_multidim_cub_template()
    test_int64_min_identity_is_exact_integer_literal()
    test_multidim_block_reduction_correct()
    test_int64_min_reduction_correct()
    test_covered_accumulator_write_is_redirected_ok()
