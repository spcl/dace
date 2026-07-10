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
    ``k`` loop. Value-preserving (column sums)."""
    sdfg = _rowsum.to_sdfg(simplify=True)
    assert _apply(sdfg) >= 1
    assert _num_wcr(sdfg) >= 1, 'the accumulation should become a WCR write'
    sdfg.validate()
    # the k loop is now parallelizable
    assert sdfg.apply_transformations_repeated(LoopToMap) >= 1
    assert _num_loops(sdfg) == 0

    k, n = 5, 7
    rng = np.random.default_rng(0)
    B = rng.standard_normal((k, n))
    out = np.full(n, -123.0)
    ref = B.sum(axis=0)
    sdfg(B=B.copy(), out=out, K=k, N=n)
    assert np.allclose(out, ref)


def test_refuses_impure_accumulator():
    """The accumulator is also read to form the increment -- a recurrence, not a
    pure reduction -- so the WCR lift (which drops the read-back) would change the
    value. Must refuse."""
    sdfg = _impure_accumulator.to_sdfg(simplify=True)
    assert _apply(sdfg) == 0
    assert _num_wcr(sdfg) == 0


def test_refuses_loop_indexed_write():
    """A loop-indexed write ``a[k] = a[k-1] + a[k]`` is a prefix scan; its write
    subset depends on the loop variable, so it is not an invariant-subset
    reduction. Must refuse."""
    sdfg = _loop_indexed_scan.to_sdfg(simplify=True)
    assert _apply(sdfg) == 0
    assert _num_wcr(sdfg) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
