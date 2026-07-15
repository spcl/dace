# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`LiftLoopCarriedReduction`.

The pass turns a loop-carried in-place array accumulation ``for k: A[i] = A[i] +
delta_k[i]`` into a WCR write ``A[i] (wcr: +)= delta_k[i]`` so ``LoopToMap`` can
parallelize the enclosing ``k`` loop as a reduction. It must lift only genuine
reductions -- a pure accumulator read at a loop-invariant subset -- and refuse
recurrences (accumulator read for the increment, or a loop-indexed write).
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.canonicalize.lift_loop_carried_reduction import LiftLoopCarriedReduction
from dace.transformation.interstate import LoopToMap

K = dace.symbol('K')
N = dace.symbol('N')


def _apply(sdfg) -> int:
    res = Pipeline([LiftLoopCarriedReduction()]).apply_pass(sdfg, {})
    return (res or {}).get('LiftLoopCarriedReduction', 0) or 0


def _num_wcr(sdfg) -> int:
    return sum(1 for e, _ in sdfg.all_edges_recursive() if getattr(getattr(e, 'data', None), 'wcr', None) is not None)


def _num_loops(sdfg) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive()
               if isinstance(n, LoopRegion) and not getattr(n, 'pinned_sequential', False))


@dace.program
def _rowsum(B: dace.float64[K, N], out: dace.float64[N]):
    out[:] = 0.0
    for k in range(K):
        for i in dace.map[0:N]:
            out[i] = out[i] + B[k, i]


@dace.program
def _impure_accumulator(B: dace.float64[K, N], out: dace.float64[N]):
    out[:] = 1.0
    for k in range(K):
        for i in dace.map[0:N]:
            # out[i] is read BOTH as the accumulator AND inside the increment ->
            # a recurrence, not a pure reduction. Must be refused.
            out[i] = out[i] + B[k, i] * out[i]


@dace.program
def _loop_indexed_scan(a: dace.float64[N]):
    for k in range(1, N):
        # write subset depends on the loop variable -> a scan, not an
        # invariant-subset reduction. Must be refused.
        a[k] = a[k - 1] + a[k]


def test_lifts_rowsum_reduction_and_parallelizes():
    """``for k: out[i] = out[i] + B[k, i]`` is a reduction of ``out`` over ``k``;
    lifting the in-place accumulation to a WCR lets ``LoopToMap`` parallelize the
    ``k`` loop. Value-preserving (column sums). Concrete sizes -- the pass refuses
    symbolic extents."""
    k, n = 5, 7
    sdfg = _rowsum.to_sdfg(simplify=True)
    _specialize(sdfg, K=k, N=n)
    assert _apply(sdfg) >= 1
    assert _num_wcr(sdfg) >= 1, 'the accumulation should become a WCR write'
    sdfg.validate()
    # the k loop is now parallelizable
    assert sdfg.apply_transformations_repeated(LoopToMap) >= 1
    assert _num_loops(sdfg) == 0

    rng = np.random.default_rng(0)
    B = rng.standard_normal((k, n))
    out = np.full(n, -123.0)
    ref = B.sum(axis=0)
    sdfg(B=B.copy(), out=out)
    assert np.allclose(out, ref)


def test_refuses_impure_accumulator():
    """The accumulator is also read to form the increment -- a recurrence, not a
    pure reduction -- so the WCR lift (which drops the read-back) would change the
    value. Must refuse."""
    sdfg = _impure_accumulator.to_sdfg(simplify=True)
    _specialize(sdfg, K=5, N=7)  # concrete: prove the refusal is the recurrence gate, not the size guard
    assert _apply(sdfg) == 0
    assert _num_wcr(sdfg) == 0


def test_refuses_loop_indexed_write():
    """A loop-indexed write ``a[k] = a[k-1] + a[k]`` is a prefix scan; its write
    subset depends on the loop variable, so it is not an invariant-subset
    reduction. Must refuse."""
    sdfg = _loop_indexed_scan.to_sdfg(simplify=True)
    _specialize(sdfg, N=7)  # concrete: prove the refusal is the invariant-subset gate, not the size guard
    assert _apply(sdfg) == 0
    assert _num_wcr(sdfg) == 0


# --------------------------------------------------------------------------- #
# contour_integral pattern: an OUTER reduction loop whose body computes a
# per-iteration array increment (an inner element map) and accumulates it into a
# whole-array output whose index is INVARIANT over the loop variable. After the
# lift + LoopToMap the outer loop parallelizes and the accumulation becomes a WCR
# reduction. These run the FULL canonicalize pipeline (the wired pass) and check
# both full parallelization (0 residual loops) and numerical correctness -- the
# WCR nesting (inner element map WCR under a parallel outer map) must reduce
# correctly, including under real multithreading.
# --------------------------------------------------------------------------- #

KK = dace.symbol('KK')
NR = dace.symbol('NR')
NM = dace.symbol('NM')


def _specialize(sdfg, **subs):
    """Substitute concrete integer sizes for the kernel's size symbols.
    ``LiftLoopCarriedReduction`` refuses SYMBOLIC extents (the lift's cost is undecidable
    without knowing the inner-map size -- see the pass docstring and
    ``samples/optimization/maximal_parallelism.md``), so the lift-expected tests specialize."""
    for name, val in subs.items():
        sdfg.replace(name, str(val))
    return sdfg


def _canon_full(prog, **subs):
    from dace.transformation.passes.canonicalize import canonicalize
    sdfg = prog.to_sdfg(simplify=True)
    _specialize(sdfg, **subs)
    canonicalize(sdfg, validate=True, target='cpu')
    return sdfg


def _residual_loops(sdfg) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive()
               if isinstance(n, LoopRegion) and not getattr(n, 'pinned_sequential', False))


@dace.program
def _contour_sum(B: dace.float64[KK, NR, NM], P: dace.float64[NR, NM]):
    P[:] = 0.0
    for idx in range(KK):
        X = np.zeros((NR, NM), dtype=np.float64)
        for i, j in dace.map[0:NR, 0:NM]:
            X[i, j] = B[idx, i, j] * B[idx, i, j]   # per-iteration computed increment
        for i, j in dace.map[0:NR, 0:NM]:
            P[i, j] = P[i, j] + X[i, j]             # reduction over idx (invariant subset)


@dace.program
def _contour_two(B: dace.float64[KK, NR, NM], P0: dace.float64[NR, NM], P1: dace.float64[NR, NM]):
    P0[:] = 0.0
    P1[:] = 0.0
    for idx in range(KK):
        X = np.zeros((NR, NM), dtype=np.float64)
        for i, j in dace.map[0:NR, 0:NM]:
            X[i, j] = B[idx, i, j] + 1.0
        for i, j in dace.map[0:NR, 0:NM]:
            P0[i, j] = P0[i, j] + X[i, j]
            P1[i, j] = P1[i, j] + 2.0 * X[i, j]     # fused multi-output reduction (contour P0/P1)


@dace.program
def _contour_max(B: dace.float64[KK, NR, NM], M: dace.float64[NR, NM]):
    M[:] = -1.0e30
    for idx in range(KK):
        for i, j in dace.map[0:NR, 0:NM]:
            M[i, j] = max(M[i, j], B[idx, i, j])    # max reduction over idx


@dace.program
def _contour_prod(B: dace.float64[KK, NR, NM], P: dace.float64[NR, NM]):
    P[:] = 1.0
    for idx in range(KK):
        for i, j in dace.map[0:NR, 0:NM]:
            P[i, j] = P[i, j] * B[idx, i, j]        # product reduction over idx


@dace.program
def _contour_indexed_injective(B: dace.float64[KK, NM], Q: dace.float64[KK, NM]):
    for idx in range(KK):
        for j in dace.map[0:NM]:
            Q[idx, j] = Q[idx, j] + B[idx, j]       # write index USES idx -> injective, NOT a reduction


def test_contour_pattern_sum_reduction():
    """Single-accumulator whole-array reduction over the outer loop, with a
    per-iteration computed increment: fully parallelizes and sums correctly."""
    kk, nr, nm = 6, 5, 7
    sdfg = _canon_full(_contour_sum, KK=kk, NR=nr, NM=nm)
    assert _residual_loops(sdfg) == 0
    rng = np.random.default_rng(0)
    B = rng.standard_normal((kk, nr, nm))
    P = np.full((nr, nm), 123.0)
    ref = (B * B).sum(axis=0)
    sdfg(B=B.copy(), P=P)
    assert np.allclose(P, ref)


def test_contour_pattern_two_accumulators():
    """Fused multi-output reduction (contour P0/P1): both accumulators lift and
    parallelize; each reduces correctly."""
    kk, nr, nm = 6, 4, 5
    sdfg = _canon_full(_contour_two, KK=kk, NR=nr, NM=nm)
    assert _residual_loops(sdfg) == 0
    rng = np.random.default_rng(1)
    B = rng.standard_normal((kk, nr, nm))
    P0 = np.full((nr, nm), -7.0)
    P1 = np.full((nr, nm), 9.0)
    X = B + 1.0
    ref0 = X.sum(axis=0)
    ref1 = (2.0 * X).sum(axis=0)
    sdfg(B=B.copy(), P0=P0, P1=P1)
    assert np.allclose(P0, ref0) and np.allclose(P1, ref1)


def test_contour_pattern_max_reduction():
    """A max-reduction (WCR ``max``) over the outer loop parallelizes + is exact."""
    kk, nr, nm = 5, 4, 6
    sdfg = _canon_full(_contour_max, KK=kk, NR=nr, NM=nm)
    assert _residual_loops(sdfg) == 0
    rng = np.random.default_rng(2)
    B = rng.standard_normal((kk, nr, nm))
    M = np.zeros((nr, nm))
    ref = B.max(axis=0)
    sdfg(B=B.copy(), M=M)
    assert np.allclose(M, ref)


def test_contour_pattern_product_reduction():
    """A product-reduction (WCR ``*``) over the outer loop parallelizes + is exact."""
    kk, nr, nm = 5, 3, 4
    sdfg = _canon_full(_contour_prod, KK=kk, NR=nr, NM=nm)
    assert _residual_loops(sdfg) == 0
    rng = np.random.default_rng(3)
    B = 0.5 + rng.random((kk, nr, nm))  # away from 0 to keep the product well-conditioned
    P = np.zeros((nr, nm))
    ref = B.prod(axis=0)
    sdfg(B=B.copy(), P=P)
    assert np.allclose(P, ref)


def test_refuses_symbolic_sizes():
    """The lift's payoff depends on the inner-map size vs the machine -- undecidable for a
    SYMBOLIC extent, where lifting usually regresses (measured 3-4x slower past the crossover;
    see the pass docstring and ``samples/optimization/maximal_parallelism.md``). With symbolic
    ``KK/NR/NM`` the reduction axis must stay SEQUENTIAL (parallel inner map only). The concrete
    counterpart ``test_contour_pattern_sum_reduction`` -- same kernel, sizes substituted --
    confirms it fully parallelizes once the sizes are known, so the residual loop here is the
    size guard, not a matching failure."""
    sym = _canon_full(_contour_sum)  # KK, NR, NM left symbolic -- no substitution
    assert _residual_loops(sym) >= 1, 'symbolic-size reduction axis must stay sequential (lift refused)'


def test_contour_pattern_thread_safe_reduction():
    """The nested WCR (inner element map's WCR under the PARALLEL outer map) must
    reduce correctly under real multithreading -- an OMP=1-only check would miss a
    cross-iteration race. Runs the two-accumulator kernel several times with
    OMP_NUM_THREADS>1 and requires every run to match the reference."""
    import os
    kk, nr, nm = 8, 6, 5
    sdfg = _canon_full(_contour_two, KK=kk, NR=nr, NM=nm)
    assert _residual_loops(sdfg) == 0
    rng = np.random.default_rng(4)
    B = rng.standard_normal((kk, nr, nm))
    X = B + 1.0
    ref0 = X.sum(axis=0)
    ref1 = (2.0 * X).sum(axis=0)
    prev = os.environ.get('OMP_NUM_THREADS')
    os.environ['OMP_NUM_THREADS'] = '4'
    try:
        csdfg = sdfg.compile()
        for _ in range(6):
            P0 = np.full((nr, nm), 0.0)
            P1 = np.full((nr, nm), 0.0)
            csdfg(B=B.copy(), P0=P0, P1=P1)
            assert np.allclose(P0, ref0), f"P0 race: maxdiff {np.abs(P0 - ref0).max():.2e}"
            assert np.allclose(P1, ref1), f"P1 race: maxdiff {np.abs(P1 - ref1).max():.2e}"
    finally:
        if prev is None:
            os.environ.pop('OMP_NUM_THREADS', None)
        else:
            os.environ['OMP_NUM_THREADS'] = prev


def test_contour_pattern_indexed_write_is_injective_not_reduction():
    """When the write index USES the loop variable (``Q[idx, j]``), each iteration
    writes a DISTINCT element -- an injective parallel aug-assign, NOT a reduction.
    LiftLoopCarriedReduction must NOT add a WCR (the invariant-subset gate fails);
    LoopToMap parallelizes it directly. Value-preserving."""
    sdfg = _contour_indexed_injective.to_sdfg(simplify=True)
    assert _apply(sdfg) == 0          # not lifted
    assert _num_wcr(sdfg) == 0
    from dace.transformation.passes.canonicalize import canonicalize
    csdfg = _canon_full(_contour_indexed_injective)
    kk, nm = 5, 6
    rng = np.random.default_rng(5)
    B = rng.standard_normal((kk, nm))
    Q0 = rng.standard_normal((kk, nm))
    ref = Q0 + B
    Q = Q0.copy()
    csdfg(B=B.copy(), Q=Q, KK=kk, NM=nm)
    assert np.allclose(Q, ref)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
