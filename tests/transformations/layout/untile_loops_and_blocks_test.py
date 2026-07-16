# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.layout.untile_loops_and_blocks.UntileLoopsAndBlocks`.

The kernels here GENUINELY arrive blocked+tiled: a manually-tiled two-level nest
``for i in range(0, N, K): for ii in range(0, K)`` whose body reaches an array through its
matching block dimension -- an array of physical shape ``[N/K, K]`` (or ``[M, N/K, K]``) accessed
as ``A[..., i // K, ii]``. These prove the gap the pass fills:

  * plain ``UntileLoops`` REFUSES such a nest (its combined-access audit rejects the split
    ``A[int_floor(i, K), ii]`` subset) -- the array stays blocked AND the loop stays tiled;
  * ``UntileLoopsAndBlocks`` (and ``prepare_for_layout``, which runs it first) untiles the loop
    AND unblocks the array in one coordinated rewrite, returning both to packed-C, bit-exact vs
    a plain numpy reference.

Compiled kernels run in an isolated build folder (unique cache under a temp dir) so the shared
repo ``.dacecache`` is never touched.
"""
import os

os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import contextlib
import tempfile

import numpy
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops
from dace.transformation.layout.untile_loops_and_blocks import UntileLoopsAndBlocks
from dace.transformation.layout.prepare import prepare_for_layout

N = dace.symbol('N')
M = dace.symbol('M')
K = 4


@contextlib.contextmanager
def isolated_build():
    """Compile/run in a unique cache under a throwaway temp dir -- never the repo .dacecache."""
    with tempfile.TemporaryDirectory(prefix='untile_blocks_') as td:
        with dace.config.set_temporary('cache', value='unique'), \
             dace.config.set_temporary('default_build_folder', value=os.path.join(td, 'dc')):
            yield


def run_isolated(sdfg, **kwargs):
    with isolated_build():
        sdfg(**kwargs)


def loop_vars(sdfg):
    return [r.loop_variable for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


# -----------------------------------------------------------------------------
# The blocked+tiled kernel: A is physically [N/K, K], accessed A[i//K, ii] while
# the loop is for i in range(0, N, K): for ii in range(0, K).
# -----------------------------------------------------------------------------


@dace.program
def tiled_blocked(A: dace.float64[N // K, K], B: dace.float64[N]):
    for i in range(0, N, K):
        for ii in range(0, K):
            A[i // K, ii] = B[i + ii] * 2.0


@dace.program
def tiled_blocked_offset_start(A: dace.float64[N // K, K], B: dace.float64[N]):
    # Same blocked+tiled shape, but the outer loop starts at 1 -- NOT a multiple of K.
    for i in range(1, N, K):
        for ii in range(0, K):
            A[i // K, ii] = B[i + ii] * 2.0


def test_blocked_untile_refuses_a_start_that_is_not_a_multiple_of_the_tile():
    """The case-A fold rewrites the unblocked element ``int_floor(i,K)*K + ii`` to ``k``, which is
    the original element only when ``int_floor(i,K)*K == i`` -- i.e. when the outer start is a
    multiple of K. With start=1, K=4 the first blocked access reads logical element
    int_floor(1,4)*4+0 = 0 while the untiled one would read k=1: every element shifts by
    (start mod K), silently and bit-inexactly. The pass must REFUSE and leave the SDFG untouched
    rather than miscompile."""
    sdfg = tiled_blocked_offset_start.to_sdfg(simplify=True)
    assert len(sdfg.arrays['A'].shape) == 2  # blocked
    assert len(loop_vars(sdfg)) == 2  # tiled

    UntileLoopsAndBlocks().apply_pass(sdfg, {})

    # refused: nothing untiled, nothing unblocked, no half-applied rewrite
    assert len(sdfg.arrays['A'].shape) == 2, 'A must stay blocked -- the fold is invalid for this start'
    assert len(loop_vars(sdfg)) == 2, 'loop must stay tiled -- the fold is invalid for this start'
    sdfg.validate()


def test_blocked_untile_still_applies_for_a_multiple_start():
    """The guard must not over-refuse: the ordinary start=0 blocked nest still untiles+unblocks."""
    sdfg = tiled_blocked.to_sdfg(simplify=True)
    UntileLoopsAndBlocks().apply_pass(sdfg, {})
    assert len(sdfg.arrays['A'].shape) == 1, 'A should have been unblocked for a start of 0'
    assert len(loop_vars(sdfg)) == 1, 'loop should have been untiled for a start of 0'


def test_untile_loops_alone_leaves_array_blocked_gap():
    """(a) Plain ``UntileLoops`` alone does NOT unblock the array -- it refuses the whole nest
    because the split ``A[int_floor(i, K), ii]`` subset fails its combined-access audit. The
    array stays 2-D (blocked) and the loop stays two-level. This proves the gap is real."""
    sdfg = tiled_blocked.to_sdfg(simplify=True)
    assert len(sdfg.arrays['A'].shape) == 2  # blocked [N/K, K]
    assert len(loop_vars(sdfg)) == 2  # tiled two-level nest

    res = UntileLoops().apply_pass(sdfg, {})

    assert res is None, 'plain UntileLoops must refuse the blocked nest (gap)'
    assert len(sdfg.arrays['A'].shape) == 2, 'A must remain blocked (UntileLoops does not unblock)'
    assert len(loop_vars(sdfg)) == 2, 'loop must remain tiled (UntileLoops refused)'


def test_untile_loops_and_blocks_untiles_and_unblocks():
    """(b) ``UntileLoopsAndBlocks`` untiles the loop AND unblocks the array: A returns to a flat
    1-D descriptor and a single unit-stride loop remains. (c) The compiled result is bit-exact
    vs a plain numpy reference."""
    sdfg = tiled_blocked.to_sdfg(simplify=True)

    res = UntileLoopsAndBlocks().apply_pass(sdfg, {})
    sdfg.validate()

    assert res == 1, 'UntileLoopsAndBlocks must collapse the tile pair'
    assert len(sdfg.arrays['A'].shape) == 1, f'A must be unblocked to 1-D, got {sdfg.arrays["A"].shape}'
    lv = loop_vars(sdfg)
    assert len(lv) == 1 and lv[0].startswith('_untile_k_'), f'expected one collapsed loop, got {lv}'

    n = 32  # divisible by K=4 (clean tile)
    rng = numpy.random.default_rng(0)
    B = rng.standard_normal(n)
    A = numpy.zeros(n)  # A is now flat [N]
    run_isolated(sdfg, A=A, B=B, N=n)
    assert numpy.allclose(A, B * 2.0), f'value mismatch: {A} vs {B * 2.0}'


def test_prepare_for_layout_untiles_and_unblocks():
    """``prepare_for_layout`` (which runs ``UntileLoopsAndBlocks`` first, before canonicalize)
    also lands the array + schedule in packed-C, bit-exact."""
    sdfg = tiled_blocked.to_sdfg(simplify=True)

    prepare_for_layout(sdfg, validate=True)

    assert len(sdfg.arrays['A'].shape) == 1, f'A must be unblocked to 1-D, got {sdfg.arrays["A"].shape}'

    n = 32
    rng = numpy.random.default_rng(1)
    B = rng.standard_normal(n)
    A = numpy.zeros(n)
    run_isolated(sdfg, A=A, B=B, N=n)
    assert numpy.allclose(A, B * 2.0)


# -----------------------------------------------------------------------------
# 2-D array blocked on the last logical dimension: [M, N/K, K] accessed A[m, i//K, ii].
# -----------------------------------------------------------------------------


@dace.program
def tiled_blocked_2d(A: dace.float64[M, N // K, K], B: dace.float64[M, N]):
    for m in range(0, M):
        for i in range(0, N, K):
            for ii in range(0, K):
                A[m, i // K, ii] = B[m, i + ii] + 3.0


def test_2d_blocked_last_dim_untiles_and_unblocks():
    """A 2-D logical array blocked on its last dim (physical ``[M, N/K, K]``, block index adjacent
    to the extent-K inner dim) is unblocked to ``[M, N]`` while the tile loop collapses -- the
    outer non-tiled ``m`` loop is untouched. Bit-exact."""
    sdfg = tiled_blocked_2d.to_sdfg(simplify=True)
    assert len(sdfg.arrays['A'].shape) == 3  # [M, N/K, K]

    res = UntileLoopsAndBlocks().apply_pass(sdfg, {})
    sdfg.validate()

    assert res == 1
    assert len(sdfg.arrays['A'].shape) == 2, f'A must be unblocked to [M, N], got {sdfg.arrays["A"].shape}'

    mm, nn = 3, 32
    rng = numpy.random.default_rng(2)
    B = rng.standard_normal((mm, nn))
    A = numpy.zeros((mm, nn))
    run_isolated(sdfg, A=A, B=B, N=nn, M=mm)
    assert numpy.allclose(A, B + 3.0)


# -----------------------------------------------------------------------------
# Integer kernel -- exact equality (array_equal) rather than allclose.
# -----------------------------------------------------------------------------


@dace.program
def tiled_blocked_int(A: dace.int64[N // K, K], B: dace.int64[N]):
    for i in range(0, N, K):
        for ii in range(0, K):
            A[i // K, ii] = B[i + ii] + 7


def test_int_kernel_untile_unblock_array_equal():
    """Integer blocked+tiled kernel: untile+unblock and assert exact integer equality."""
    sdfg = tiled_blocked_int.to_sdfg(simplify=True)

    res = UntileLoopsAndBlocks().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert len(sdfg.arrays['A'].shape) == 1

    n = 32
    rng = numpy.random.default_rng(3)
    B = rng.integers(-100, 100, size=n, dtype=numpy.int64)
    A = numpy.zeros(n, dtype=numpy.int64)
    run_isolated(sdfg, A=A, B=B, N=n)
    assert numpy.array_equal(A, B + 7), f'int mismatch: {A} vs {B + 7}'


# -----------------------------------------------------------------------------
# No-op / superset contracts.
# -----------------------------------------------------------------------------


def test_noop_on_plain_untiled_kernel():
    """A plain (non-tiled) map kernel is a strict no-op: the pass declines and leaves the SDFG
    untouched (proving it fires only on the genuine blocked+tiled pattern)."""

    @dace.program
    def plain(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            A[i] = B[i] + 1.0

    sdfg = plain.to_sdfg(simplify=True)
    shape_before = tuple(str(s) for s in sdfg.arrays['A'].shape)

    res = UntileLoopsAndBlocks().apply_pass(sdfg, {})

    assert res is None, 'plain kernel must be a no-op'
    assert tuple(str(s) for s in sdfg.arrays['A'].shape) == shape_before


def test_superset_plain_combined_tile_untiles_without_unblock():
    """A plain combined tile ``for i in range(0, N, K): for ii in range(K): a[i+ii] = b[i+ii]``
    (no blocked array) collapses exactly like ``UntileLoops`` -- the array stays 1-D, no unblock
    triggers -- so the pass is a strict superset."""

    @dace.program
    def plain_tile(a: dace.float64[N], b: dace.float64[N]):
        for i in range(0, N, K):
            for ii in range(K):
                a[i + ii] = b[i + ii]

    sdfg = plain_tile.to_sdfg(simplify=True)
    res = UntileLoopsAndBlocks().apply_pass(sdfg, {})
    sdfg.validate()

    assert res == 1
    lv = loop_vars(sdfg)
    assert len(lv) == 1 and lv[0].startswith('_untile_k_')
    assert len(sdfg.arrays['a'].shape) == 1  # never blocked, so never unblocked

    n = 32
    rng = numpy.random.default_rng(4)
    b = rng.standard_normal(n)
    a = numpy.zeros(n)
    run_isolated(sdfg, a=a, b=b, N=n)
    assert numpy.allclose(a, b)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
