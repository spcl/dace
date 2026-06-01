# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that ``LoopToMap`` parallelizes a loop only when its writes are
    provably non-overlapping across iterations.

    Each iteration of a loop must write disjoint locations for the loop to be a
    valid map. Two affine write subscripts ``a1*i + b1`` and ``a2*i + b2`` into
    the same container collide on some pair of iterations if and only if
    ``gcd(a1, a2)`` divides ``b2 - b1``; otherwise they are provably disjoint
    for any iteration range.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.interstate import LoopToMap

N = dace.symbol('N')
M = dace.symbol('M')


def _has_map(sdfg: dace.SDFG) -> bool:
    """ True if any state in the (recursively expanded) SDFG contains a Map. """
    return any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())


@dace.program
def overlapping_writes(A: dace.int64[5 * N]):
    for i in range(N):
        A[5 * i] = 1
        A[3 * i] = 2


@dace.program
def injective_write(A: dace.int64[2 * N]):
    for i in range(N):
        A[2 * i] = i


@dace.program
def disjoint_stride_writes(A: dace.int64[2 * N]):
    for i in range(N):
        A[2 * i] = 1
        A[2 * i + 1] = 2


@dace.program
def shifted_writes(A: dace.int64[N + 1]):
    for i in range(N):
        A[i] = 1
        A[i + 1] = 2


@dace.program
def disjoint_outer_dim(B: dace.int64[2 * N, 4]):
    for i in range(N):
        B[2 * i, :] = 1
        B[2 * i + 1, :] = 2


def _applies(program) -> int:
    sdfg = program.to_sdfg(simplify=False)
    return sdfg.apply_transformations_repeated(LoopToMap)


def test_rejects_overlapping_writes():
    """ ``A[5*i]`` and ``A[3*i]`` collide at ``A[15]`` (i=3 and i=5). """
    sdfg = overlapping_writes.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) == 0

    n = 64
    a = np.full(5 * n, -1, dtype=np.int64)
    sdfg(A=a, N=n)
    ref = np.full(5 * n, -1, dtype=np.int64)
    for i in range(n):
        ref[5 * i] = 1
        ref[3 * i] = 2
    assert np.array_equal(a, ref)


def test_accepts_injective_write():
    """ A single ``a*i + b`` write is injective in ``i`` and parallelizable. """
    sdfg = injective_write.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) >= 1

    n = 64
    a = np.full(2 * n, -1, dtype=np.int64)
    sdfg(A=a, N=n)
    ref = np.full(2 * n, -1, dtype=np.int64)
    for i in range(n):
        ref[2 * i] = i
    assert np.array_equal(a, ref)


def test_accepts_disjoint_strides():
    """ ``A[2*i]`` (even) and ``A[2*i+1]`` (odd) never collide, for any range. """
    sdfg = disjoint_stride_writes.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) >= 1

    n = 64
    a = np.full(2 * n, -1, dtype=np.int64)
    sdfg(A=a, N=n)
    ref = np.full(2 * n, -1, dtype=np.int64)
    for i in range(n):
        ref[2 * i] = 1
        ref[2 * i + 1] = 2
    assert np.array_equal(a, ref)


def test_rejects_shifted_writes():
    """ ``A[i]`` and ``A[i+1]`` collide between consecutive iterations. """
    assert _applies(shifted_writes) == 0


def test_accepts_disjoint_outer_dimension():
    """ A provably disjoint leading dimension makes the whole access disjoint. """
    sdfg = disjoint_outer_dim.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) >= 1

    n = 32
    b = np.full((2 * n, 4), -1, dtype=np.int64)
    sdfg(B=b, N=n)
    ref = np.full((2 * n, 4), -1, dtype=np.int64)
    for i in range(n):
        ref[2 * i, :] = 1
        ref[2 * i + 1, :] = 2
    assert np.array_equal(b, ref)


# ---------------------------------------------------------------------------
# Indirect / nonlinear subscripts must NOT be certified disjoint by the new
# affine fast path. ``idx`` is not known to be a permutation, and ``i*i`` /
# ``i % k`` fall outside the affine ``a*i + b`` model, so the loop must stay
# sequential (no Map) regardless of the gcd-disjointness reasoning.
# ---------------------------------------------------------------------------


@dace.program
def indirect_write_vs_affine(A: dace.int64[5 * N], idx: dace.int64[N]):
    for i in range(N):
        A[idx[i]] = 1
        A[5 * i] = 2


@dace.program
def indirect_read_vs_affine(A: dace.int64[3 * N], idx: dace.int64[N], out: dace.int64[N]):
    for i in range(N):
        out[i] = A[idx[i]]
        A[3 * i] = i


@dace.program
def nonlinear_square_write(A: dace.int64[N * N]):
    for i in range(N):
        A[i * i] = 1
        A[i] = 2


@dace.program
def nonlinear_mod_write(A: dace.int64[N]):
    for i in range(N):
        A[i % 4] = 1
        A[i] = 2


def test_rejects_indirect_write_vs_affine():
    """ ``A[idx[i]]`` could equal ``A[5*i]``; ``idx`` is not a known
        permutation, so this is a possible cross-iteration write-write
        dependence and the loop must stay sequential. """
    sdfg = indirect_write_vs_affine.to_sdfg(simplify=False)
    ref_sdfg = copy.deepcopy(sdfg)
    assert sdfg.apply_transformations_repeated(LoopToMap) == 0
    assert not _has_map(sdfg)

    n = 32
    # Adversarial index: many entries alias the affine ``5*i`` targets.
    idx = (np.arange(n) % 5).astype(np.int64)

    a = np.full(5 * n, -1, dtype=np.int64)
    a_ref = a.copy()
    sdfg(A=a, idx=idx, N=n)
    ref_sdfg(A=a_ref, idx=idx.copy(), N=n)
    assert np.array_equal(a, a_ref)


def test_rejects_indirect_read_vs_affine():
    """ ``... = A[idx[i]]`` reads while ``A[3*i]`` writes the same container;
        a RAW/WAR hazard the affine model cannot rule out. """
    sdfg = indirect_read_vs_affine.to_sdfg(simplify=False)
    ref_sdfg = copy.deepcopy(sdfg)
    assert sdfg.apply_transformations_repeated(LoopToMap) == 0
    assert not _has_map(sdfg)

    n = 24
    idx = (np.arange(n) % 3 * 3).astype(np.int64)
    A = np.arange(3 * n, dtype=np.int64)
    A_ref = A.copy()
    out = np.full(n, -1, dtype=np.int64)
    out_ref = out.copy()
    sdfg(A=A, idx=idx, out=out, N=n)
    ref_sdfg(A=A_ref, idx=idx.copy(), out=out_ref, N=n)
    assert np.array_equal(out, out_ref)
    assert np.array_equal(A, A_ref)


def test_rejects_nonlinear_square_write():
    """ ``A[i*i]`` is nonlinear in ``i`` and outside the affine model; with a
        second write to ``A`` the loop must stay sequential. """
    sdfg = nonlinear_square_write.to_sdfg(simplify=False)
    ref_sdfg = copy.deepcopy(sdfg)
    assert sdfg.apply_transformations_repeated(LoopToMap) == 0
    assert not _has_map(sdfg)

    n = 16
    a = np.full(n * n, -1, dtype=np.int64)
    a_ref = a.copy()
    sdfg(A=a, N=n)
    ref_sdfg(A=a_ref, N=n)
    assert np.array_equal(a, a_ref)


def test_rejects_nonlinear_mod_write():
    """ ``A[i % 4]`` is nonlinear in ``i``; with a second write to ``A`` the
        loop must stay sequential. """
    sdfg = nonlinear_mod_write.to_sdfg(simplify=False)
    ref_sdfg = copy.deepcopy(sdfg)
    assert sdfg.apply_transformations_repeated(LoopToMap) == 0
    assert not _has_map(sdfg)

    n = 16
    a = np.full(n, -1, dtype=np.int64)
    a_ref = a.copy()
    sdfg(A=a, N=n)
    ref_sdfg(A=a_ref, N=n)
    assert np.array_equal(a, a_ref)


# ---------------------------------------------------------------------------
# A dimension that both writes index by the same injective function of the loop
# variable pins any collision to a single iteration, so the writes are disjoint
# across iterations even when the *other* indices are opaque symbols the affine
# model cannot certify. This is the CloudSC scatter pattern
# ``zsolqa[0, imelt, i]`` / ``zsolqa[imelt, 0, i]`` (``i`` the parallel column).
# ---------------------------------------------------------------------------


@dace.program
def shared_iteration_dim(A: dace.int64[8, 8, N]):
    for i in range(N):
        A[0, M, i] = 1
        A[M, 0, i] = 2


@dace.program
def shared_constant_dim_shifted(A: dace.int64[8, N + 1]):
    for i in range(N):
        A[M, i] = 1
        A[M, i + 1] = 2


def test_accepts_shared_iteration_dimension():
    """ ``A[0, M, i]`` and ``A[M, 0, i]`` share the loop variable ``i`` in their
        last dimension, so each iteration owns column ``i`` and they never
        collide across iterations -- parallelizable despite the opaque ``M``. """
    sdfg = shared_iteration_dim.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) >= 1
    assert _has_map(sdfg)

    n, m = 16, 3
    a = np.full((8, 8, n), -1, dtype=np.int64)
    sdfg(A=a, N=n, M=m)
    ref = np.full((8, 8, n), -1, dtype=np.int64)
    for i in range(n):
        ref[0, m, i] = 1
        ref[m, 0, i] = 2
    assert np.array_equal(a, ref)


def test_rejects_shared_constant_dimension_with_shift():
    """ Guard: the shared dimension here is the *constant* ``M`` (no dependence
        on ``i``), so it does not pin iterations together; ``A[M, i]`` and
        ``A[M, i+1]`` still collide between consecutive iterations. """
    assert _applies(shared_constant_dim_shifted) == 0


def test_positive_control_disjoint_strides_becomes_map():
    """ Positive control: ``A[2*i]`` / ``A[2*i+1]`` are affine and
        gcd-disjoint, so the new fast path SHOULD parallelize (Map present). """
    sdfg = disjoint_stride_writes.to_sdfg(simplify=False)
    assert sdfg.apply_transformations_repeated(LoopToMap) >= 1
    assert _has_map(sdfg)

    n = 64
    a = np.full(2 * n, -1, dtype=np.int64)
    sdfg(A=a, N=n)
    ref = np.full(2 * n, -1, dtype=np.int64)
    for i in range(n):
        ref[2 * i] = 1
        ref[2 * i + 1] = 2
    assert np.array_equal(a, ref)


_CR_N = dace.symbol('CR_N')


@dace.program
def _forward_dep_recurrence(a: dace.float64[_CR_N], b: dace.float64[_CR_N], c: dace.float64[_CR_N],
                            d: dace.float64[_CR_N], e: dace.float64[_CR_N]):
    """Forward loop-carried dependency: ``a[i] = ... + a[i+1] * ...`` reads
    the value at a position another iteration WRITES. Mirrors the TSVC s243
    body."""
    for i in range(_CR_N - 1):
        a[i] = b[i] + c[i] * d[i]
        b[i] = a[i] + d[i] * e[i]
        a[i] = b[i] + a[i + 1] * d[i]


def test_loop_to_map_refuses_forward_carried_read():
    """``LoopToMap`` must refuse a loop where a read of an array references a
    position that another iteration WRITES (forward / backward loop-carried
    dependency, even when each iteration's write is per-iteration unique).

    Pre-fix the write-pattern check alone accepted the loop because every
    write ``a[i]`` matched ``a*i + b``; the read ``a[i+1]`` slipped through
    because the same-iteration disjoint check found no overlap WITHIN one
    iteration. The new carried-read check refuses when any read references
    the iter symbol at an offset that doesn't match the write pattern."""
    sdfg = _forward_dep_recurrence.to_sdfg(simplify=True)
    applied = sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert applied == 0, ('LoopToMap must refuse a forward-carried recurrence '
                          f'(``a[i] = ... a[i+1]``); got applied={applied}.')

    n = 32
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    d = rng.standard_normal(n)
    e = rng.standard_normal(n)
    ra, rb = a.copy(), b.copy()
    for i in range(n - 1):
        ra[i] = rb[i] + c[i] * d[i]
        rb[i] = ra[i] + d[i] * e[i]
        ra[i] = rb[i] + ra[i + 1] * d[i]
    sa, sb = a.copy(), b.copy()
    sdfg(a=sa, b=sb, c=c.copy(), d=d.copy(), e=e.copy(), CR_N=n)
    assert np.allclose(sa, ra), f'a mismatch: max diff {np.abs(sa - ra).max():.2e}'
    assert np.allclose(sb, rb), f'b mismatch: max diff {np.abs(sb - rb).max():.2e}'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
