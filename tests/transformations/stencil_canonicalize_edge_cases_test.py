# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Edge-case audit for the loop-centric canonicalization passes, exercised
    with indirect / semi-indirect stencil kernels and guarded stencils.

    Two invariants per test:
      * *always correct* -- the post-pass SDFG is numerically identical to a
        deep-copied pre-pass run, for the condition taken and not-taken;
      * *always applied when possible* -- the pass fires (returns non-``None``,
        structure changed) whenever its precondition holds, and provably
        refuses (no-op, still correct) when it does not.

    Kernels use the dace Python frontend with ``range`` so every loop is a
    ``LoopRegion`` (``MoveIfIntoLoop`` / ``LoopFission`` operate on loops).
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion, ConditionalBlock
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.loop_fission import LoopFission

N = dace.symbol('N')
M = dace.symbol('M')


def _loops(sdfg):
    return [c for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion)]


def _conds(sdfg):
    return [c for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, ConditionalBlock)]


def _top_level_conds(sdfg):
    return [b for b in sdfg.nodes() if isinstance(b, ConditionalBlock)]


# --------------------------------------------------------------------------- #
# MoveIfIntoLoop -- guarded stencils (dimension guard / whole stencil in if)   #
# --------------------------------------------------------------------------- #

@dace.program
def guarded_stencil_1d(a: dace.float64[N], b: dace.float64[N], active: dace.int32[1]):
    # Whole 1-D stencil wrapped in a guard; the boundary is handled by the
    # loop range, the guard is loop-invariant.
    if active[0] > 0:
        for i in range(1, N - 1):
            b[i] = a[i - 1] + a[i] + a[i + 1]


@dace.program
def guarded_gather_stencil(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N], active: dace.int32[1]):
    # Indirect (gather) stencil: every read of ``a`` is through ``idx``.
    if active[0] > 0:
        for i in range(1, N - 1):
            b[i] = a[idx[i] - 1] + a[idx[i]] + a[idx[i] + 1]


@dace.program
def guarded_scatter(a: dace.float64[N], idx: dace.int32[N], out: dace.float64[N], active: dace.int32[1]):
    # Indirect (scatter) write through a permutation index.
    if active[0] > 0:
        for i in range(N):
            out[idx[i]] = a[i] * 2.0


@dace.program
def guarded_semi_indirect(a: dace.float64[N, M], col: dace.int32[M], b: dace.float64[N, M], active: dace.int32[1]):
    # Semi-indirect: dim 0 (``i``) is structured/direct, dim 1 (``j``) is
    # gathered through ``col`` -- one structured dim, one unstructured.
    if active[0] > 0:
        for i in range(N):
            for j in range(1, M - 1):
                b[i, j] = a[i, col[j] - 1] + a[i, col[j]] + a[i, col[j] + 1]


@dace.program
def nested_dim_guards(a: dace.float64[N, M], b: dace.float64[N, M], gi: dace.int32[1], gj: dace.int32[1]):
    # Two stacked dimension guards over a 2-D interior stencil; the fixpoint
    # must push both into the (now top-level) i-loop.
    if gi[0] > 0:
        if gj[0] > 0:
            for i in range(1, N - 1):
                for j in range(1, M - 1):
                    b[i, j] = 0.25 * (a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1])


def _move_if_ok(sdfg):
    """The single guarding loop is hoisted to top level and now wraps a
    ConditionalBlock; no ConditionalBlock remains at SDFG top level."""
    loops = _loops(sdfg)
    assert len(loops) >= 1
    top = next(l for l in loops if l.parent_graph is sdfg)
    assert any(isinstance(r, ConditionalBlock) for r in top.all_control_flow_regions(recursive=True))
    assert not _top_level_conds(sdfg)


def test_move_if_into_guarded_1d_stencil():
    """`if active: for i in 1:N-1: b=a[i-1]+a[i]+a[i+1]` -> guard pushed into
    the loop; value-preserving for active taken and not-taken."""
    n = 32
    a = np.random.rand(n)
    base = guarded_stencil_1d.to_sdfg(simplify=True)
    for av in (1, 0):
        ref = np.full(n, 5.0)
        copy.deepcopy(base)(a=a.copy(), b=ref, active=np.array([av], np.int32), N=n)

        sdfg = guarded_stencil_1d.to_sdfg(simplify=True)
        assert MoveIfIntoLoop().apply_pass(sdfg, {}) is not None
        sdfg.validate()
        _move_if_ok(sdfg)

        out = np.full(n, 5.0)
        sdfg(a=a.copy(), b=out, active=np.array([av], np.int32), N=n)
        assert np.allclose(out, ref), f"mismatch active={av}"
        if av > 0:
            exp = np.full(n, 5.0)
            exp[1:n - 1] = a[0:n - 2] + a[1:n - 1] + a[2:n]
            assert np.allclose(out, exp)
        else:
            assert np.allclose(out, 5.0)


def test_move_if_into_guarded_gather_stencil():
    """Indirect (gather) stencil under a guard: pushed in, value-preserving."""
    n = 24
    a = np.random.rand(n)
    idx = np.random.randint(1, n - 1, size=n).astype(np.int32)
    base = guarded_gather_stencil.to_sdfg(simplify=True)
    for av in (1, 0):
        ref = np.full(n, 3.0)
        copy.deepcopy(base)(a=a.copy(), idx=idx.copy(), b=ref, active=np.array([av], np.int32), N=n)

        sdfg = guarded_gather_stencil.to_sdfg(simplify=True)
        assert MoveIfIntoLoop().apply_pass(sdfg, {}) is not None
        sdfg.validate()
        _move_if_ok(sdfg)

        out = np.full(n, 3.0)
        sdfg(a=a.copy(), idx=idx.copy(), b=out, active=np.array([av], np.int32), N=n)
        assert np.allclose(out, ref), f"mismatch active={av}"
        if av == 0:
            assert np.allclose(out, 3.0)


def test_move_if_into_guarded_scatter():
    """Indirect (scatter) write under a guard: pushed in, value-preserving."""
    n = 20
    a = np.random.rand(n)
    idx = np.random.permutation(n).astype(np.int32)
    base = guarded_scatter.to_sdfg(simplify=True)
    for av in (1, 0):
        ref = np.full(n, -1.0)
        copy.deepcopy(base)(a=a.copy(), idx=idx.copy(), out=ref, active=np.array([av], np.int32), N=n)

        sdfg = guarded_scatter.to_sdfg(simplify=True)
        assert MoveIfIntoLoop().apply_pass(sdfg, {}) is not None
        sdfg.validate()
        _move_if_ok(sdfg)

        out = np.full(n, -1.0)
        sdfg(a=a.copy(), idx=idx.copy(), out=out, active=np.array([av], np.int32), N=n)
        assert np.allclose(out, ref), f"mismatch active={av}"
        if av > 0:
            exp = np.full(n, -1.0)
            exp[idx] = a * 2.0
            assert np.allclose(out, exp)
        else:
            assert np.allclose(out, -1.0)


def test_move_if_into_guarded_semi_indirect_stencil():
    """Semi-indirect 2-D stencil (structured i, gathered j) under a guard:
    the outer loop is hoisted and wraps the guard; value-preserving."""
    n, m = 12, 18
    a = np.random.rand(n, m)
    col = np.random.randint(1, m - 1, size=m).astype(np.int32)
    base = guarded_semi_indirect.to_sdfg(simplify=True)
    for av in (1, 0):
        ref = np.full((n, m), 2.0)
        copy.deepcopy(base)(a=a.copy(), col=col.copy(), b=ref, active=np.array([av], np.int32), N=n, M=m)

        sdfg = guarded_semi_indirect.to_sdfg(simplify=True)
        assert MoveIfIntoLoop().apply_pass(sdfg, {}) is not None
        sdfg.validate()
        _move_if_ok(sdfg)

        out = np.full((n, m), 2.0)
        sdfg(a=a.copy(), col=col.copy(), b=out, active=np.array([av], np.int32), N=n, M=m)
        assert np.allclose(out, ref), f"mismatch active={av}"
        if av == 0:
            assert np.allclose(out, 2.0)


def test_move_if_into_nested_dimension_guards():
    """`if gi: if gj: for i: for j: jacobi` -> the fixpoint pushes BOTH
    guards into the top-level i-loop; value-preserving for all guard
    combinations."""
    n, m = 10, 14
    a = np.random.rand(n, m)
    base = nested_dim_guards.to_sdfg(simplify=True)
    for gi, gj in ((1, 1), (1, 0), (0, 1), (0, 0)):
        ref = np.full((n, m), 8.0)
        copy.deepcopy(base)(a=a.copy(), b=ref, gi=np.array([gi], np.int32),
                            gj=np.array([gj], np.int32), N=n, M=m)

        sdfg = nested_dim_guards.to_sdfg(simplify=True)
        assert MoveIfIntoLoop().apply_pass(sdfg, {}) is not None
        sdfg.validate()
        loops = _loops(sdfg)
        top = next(l for l in loops if l.parent_graph is sdfg)
        inner_conds = [r for r in top.all_control_flow_regions(recursive=True)
                       if isinstance(r, ConditionalBlock)]
        assert len(inner_conds) >= 2, "both guards must end up inside the loop"
        assert not _top_level_conds(sdfg)

        out = np.full((n, m), 8.0)
        sdfg(a=a.copy(), b=out, gi=np.array([gi], np.int32),
             gj=np.array([gj], np.int32), N=n, M=m)
        assert np.allclose(out, ref), f"mismatch gi={gi} gj={gj}"
        if gi > 0 and gj > 0:
            exp = np.full((n, m), 8.0)
            exp[1:n - 1, 1:m - 1] = 0.25 * (a[0:n - 2, 1:m - 1] + a[2:n, 1:m - 1] +
                                            a[1:n - 1, 0:m - 2] + a[1:n - 1, 2:m])
            assert np.allclose(out, exp)
        else:
            assert np.allclose(out, 8.0)


# --------------------------------------------------------------------------- #
# MoveIfIntoLoop -- precondition refusals (no-op, still correct)               #
# --------------------------------------------------------------------------- #

@dace.program
def guard_then_loop_then_stmt(a: dace.float64[N], b: dace.float64[N], active: dace.int32[1]):
    # Branch region is `loop ; trailing-state` -- it does NOT end in a single
    # LoopRegion, so the precondition fails and the pass must no-op.
    if active[0] > 0:
        for i in range(N):
            b[i] = a[i] + 1.0
        b[0] = 99.0


@dace.program
def guarded_stencil_with_else(a: dace.float64[N], b: dace.float64[N], active: dace.int32[1]):
    # Two branches (if/else) -> not a single-branch guard -> refuse.
    if active[0] > 0:
        for i in range(1, N - 1):
            b[i] = a[i - 1] + a[i + 1]
    else:
        for i in range(1, N - 1):
            b[i] = a[i] * 2.0


@pytest.mark.parametrize('prog,extra', [
    (guard_then_loop_then_stmt, {}),
    (guarded_stencil_with_else, {}),
])
def test_move_if_into_loop_refuses_and_stays_correct(prog, extra):
    """When the precondition does not hold the pass is a provable no-op and
    the SDFG remains numerically correct for the condition taken/not-taken."""
    n = 16
    a = np.random.rand(n)
    base = prog.to_sdfg(simplify=True)
    for av in (1, 0):
        ref = np.full(n, 4.0)
        copy.deepcopy(base)(a=a.copy(), b=ref, active=np.array([av], np.int32), N=n, **extra)

        sdfg = prog.to_sdfg(simplify=True)
        assert MoveIfIntoLoop().apply_pass(sdfg, {}) is None, "must not fire"
        sdfg.validate()

        out = np.full(n, 4.0)
        sdfg(a=a.copy(), b=out, active=np.array([av], np.int32), N=n, **extra)
        assert np.allclose(out, ref), f"mismatch active={av}"


# --------------------------------------------------------------------------- #
# LoopFission -- independent split, recurrence kept whole, indirect inputs     #
# --------------------------------------------------------------------------- #

@dace.program
def two_independent_stencils(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N - 1):
        b[i] = a[i - 1] + a[i] + a[i + 1]
        d[i] = c[i - 1] + c[i] + c[i + 1]


@dace.program
def two_independent_gathers(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N],
                            c: dace.float64[N], e: dace.float64[N]):
    # ``idx`` is a read-only shared input: it must NOT force a merge.
    for i in range(N):
        b[i] = a[idx[i]]
        e[i] = c[idx[i]]


@dace.program
def recurrence_plus_independent(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]   # loop-carried recurrence on b -- must stay whole
        d[i] = c[i] * 2.0        # independent -- may split off


@dace.program
def prefix_sum_only(a: dace.float64[N], b: dace.float64[N]):
    # Pure recurrence, single group: nothing to fission -> no-op.
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]


def test_loop_fission_splits_two_independent_stencils():
    """Two data-independent stencils in one loop fission into two loops;
    numerically identical to the pre-pass run."""
    n = 28
    a = np.random.rand(n)
    c = np.random.rand(n)
    base = two_independent_stencils.to_sdfg(simplify=True)
    assert len(_loops(base)) == 1

    ref_b, ref_d = np.full(n, 6.0), np.full(n, 6.0)
    copy.deepcopy(base)(a=a.copy(), b=ref_b, c=c.copy(), d=ref_d, N=n)

    sdfg = two_independent_stencils.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    assert len(_loops(sdfg)) == 2, "expected two fissioned loops"

    out_b, out_d = np.full(n, 6.0), np.full(n, 6.0)
    sdfg(a=a.copy(), b=out_b, c=c.copy(), d=out_d, N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_d, ref_d)
    exp_b, exp_d = np.full(n, 6.0), np.full(n, 6.0)
    exp_b[1:n - 1] = a[0:n - 2] + a[1:n - 1] + a[2:n]
    exp_d[1:n - 1] = c[0:n - 2] + c[1:n - 1] + c[2:n]
    assert np.allclose(out_b, exp_b) and np.allclose(out_d, exp_d)


def test_loop_fission_indirect_shared_input_does_not_merge():
    """Two independent gathers sharing a read-only index array fission into
    two loops (the shared read-only input must not couple them)."""
    n = 22
    a = np.random.rand(n)
    c = np.random.rand(n)
    idx = np.random.permutation(n).astype(np.int32)
    base = two_independent_gathers.to_sdfg(simplify=True)

    ref_b, ref_e = np.zeros(n), np.zeros(n)
    copy.deepcopy(base)(a=a.copy(), idx=idx.copy(), b=ref_b, c=c.copy(), e=ref_e, N=n)

    sdfg = two_independent_gathers.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    assert len(_loops(sdfg)) == 2, "shared read-only index must not force a merge"

    out_b, out_e = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), b=out_b, c=c.copy(), e=out_e, N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_e, ref_e)
    assert np.allclose(out_b, a[idx]) and np.allclose(out_e, c[idx])


def test_loop_fission_keeps_recurrence_whole_splits_independent():
    """A loop-carried recurrence stays in one loop while the independent
    statement splits off; the recurrence must remain correct."""
    n = 30
    a = np.random.rand(n)
    c = np.random.rand(n)
    base = recurrence_plus_independent.to_sdfg(simplify=True)

    ref_b, ref_d = np.zeros(n), np.zeros(n)
    ref_b[0] = 1.0
    copy.deepcopy(base)(a=a.copy(), b=ref_b, c=c.copy(), d=ref_d, N=n)

    sdfg = recurrence_plus_independent.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    assert len(_loops(sdfg)) == 2, "recurrence group and independent group split"

    out_b, out_d = np.zeros(n), np.zeros(n)
    out_b[0] = 1.0
    sdfg(a=a.copy(), b=out_b, c=c.copy(), d=out_d, N=n)
    assert np.allclose(out_b, ref_b) and np.allclose(out_d, ref_d)
    # Analytic recurrence oracle: b[i] = b[i-1] + a[i], b[0] = 1.0
    exp_b = np.zeros(n)
    exp_b[0] = 1.0
    for i in range(1, n):
        exp_b[i] = exp_b[i - 1] + a[i]
    assert np.allclose(out_b, exp_b)
    # ``d`` is written only for i in 1:N (loop starts at 1); d[0] keeps its
    # init value of 0.0.
    assert out_d[0] == 0.0
    assert np.allclose(out_d[1:], c[1:] * 2.0)


def test_loop_fission_pure_recurrence_is_noop():
    """A single-group recurrence loop has nothing to fission: no-op, correct."""
    n = 15
    a = np.random.rand(n)
    base = prefix_sum_only.to_sdfg(simplify=True)
    ref = np.zeros(n)
    ref[0] = 2.0
    copy.deepcopy(base)(a=a.copy(), b=ref, N=n)

    sdfg = prefix_sum_only.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is None, "single group -> no-op"
    sdfg.validate()
    out = np.zeros(n)
    out[0] = 2.0
    sdfg(a=a.copy(), b=out, N=n)
    assert np.allclose(out, ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
