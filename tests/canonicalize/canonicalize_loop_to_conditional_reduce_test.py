# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.canonicalize.loop_to_conditional_reduce.LoopToConditionalReduce`.

Covers TSVC ``s3111`` (the conditional ``+=`` accumulator) and the refusal
contracts (non-accumulator conditional bodies, multi-write true-branches,
unsupported ops, etc.).
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.sdfg import nodes as nd
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.canonicalize.loop_to_conditional_reduce import LoopToConditionalReduce


N = dace.symbol('N')


def _num_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _num_maps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry))


# -----------------------------------------------------------------------------
# Positive: TSVC s3111-style conditional accumulators.
# -----------------------------------------------------------------------------

def test_tsvc_s3111_conditional_sum():
    """``if a[i] > 0: sum += a[i]`` -- the canonical conditional reduction.
    After the rewrite, the body is a single state with a mask tasklet
    (``__out = __addend if cond else 0.0``) whose output is a WCR ``+`` write
    to the accumulator scalar. ``LoopToMap`` then lifts the loop with no
    loop-carried RAW to worry about; the WCR makes the write atomic and
    codegen emits the standard ``#pragma omp parallel for reduction(+:s)``
    clause for CPU.
    """

    @dace.program
    def s3111(a: dace.float64[N], b: dace.float64[1]):
        sum_val = 0.0
        for i in range(N):
            if a[i] > 0.0:
                sum_val = sum_val + a[i]
        b[0] = sum_val

    sdfg = s3111.to_sdfg(simplify=True)
    assert _num_loops(sdfg) == 1
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res == 1
    sdfg.apply_transformations_repeated(LoopToMap, validate=False)
    sdfg.validate()
    assert _num_loops(sdfg) == 0
    assert _num_maps(sdfg) >= 1

    n = 32
    rng = np.random.default_rng(3111)
    a = rng.standard_normal(n)
    expected = float(np.sum(a[a > 0.0]))
    b = np.zeros(1)
    sdfg(a=a, b=b, N=n)
    assert np.isclose(b[0], expected), f"got {b[0]}, expected {expected}"


def test_no_positives_returns_zero():
    """Edge case: all ``a[i] <= 0`` -- the cond is false at every iteration,
    so every WCR contribution is the identity ``0.0``. The accumulator stays
    at its pre-loop initial value."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[1]):
        sum_val = 0.0
        for i in range(N):
            if a[i] > 0.0:
                sum_val = sum_val + a[i]
        b[0] = sum_val

    sdfg = kernel.to_sdfg(simplify=True)
    LoopToConditionalReduce().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(LoopToMap, validate=False)

    a = -np.abs(np.random.default_rng(0).standard_normal(8))  # all negative
    b = np.zeros(1)
    sdfg(a=a, b=b, N=8)
    assert np.isclose(b[0], 0.0)


def test_all_positives_acts_like_unconditional_sum():
    """Edge case: every ``a[i] > 0`` -- the cond is always true, so the
    rewrite behaves like an unconditional sum reduction. Numerics match
    ``np.sum(a)``."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[1]):
        sum_val = 0.0
        for i in range(N):
            if a[i] > 0.0:
                sum_val = sum_val + a[i]
        b[0] = sum_val

    sdfg = kernel.to_sdfg(simplify=True)
    LoopToConditionalReduce().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(LoopToMap, validate=False)

    a = np.abs(np.random.default_rng(1).standard_normal(16))  # all positive
    b = np.zeros(1)
    sdfg(a=a, b=b, N=16)
    assert np.isclose(b[0], float(np.sum(a)))


# -----------------------------------------------------------------------------
# Refusal contracts.
# -----------------------------------------------------------------------------

def test_refuses_unconditional_accumulator():
    """``sum = sum + a[i]`` (no ConditionalBlock) is NOT in this pass's
    scope -- it's the plain reduction shape that ``LoopToReduce`` already
    handles. Refuse."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[1]):
        sum_val = 0.0
        for i in range(N):
            sum_val = sum_val + a[i]
        b[0] = sum_val

    sdfg = kernel.to_sdfg(simplify=True)
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res is None, "unconditional accumulator must be left for LoopToReduce"


def test_refuses_true_branch_writes_extra_array():
    """The true-branch writes both the accumulator AND another non-transient
    array (``if cond: sum += a[i]; out[i] = sum`` -- a conditional prefix-scan
    plus output). The rewrite would silently drop the ``out[i] = sum`` write,
    so refuse."""

    @dace.program
    def kernel(a: dace.float64[N], out: dace.float64[N], b: dace.float64[1]):
        sum_val = 0.0
        for i in range(N):
            if a[i] > 0.0:
                sum_val = sum_val + a[i]
                out[i] = sum_val
        b[0] = sum_val

    sdfg = kernel.to_sdfg(simplify=True)
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res is None


def test_refuses_else_branch_with_content():
    """An else branch with side effects (``if cond: sum += a[i] else: ...``)
    isn't handled by the inline-ternary mask. Refuse."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N]):
        sum_val = 0.0
        for i in range(N):
            if a[i] > 0.0:
                sum_val = sum_val + a[i]
            else:
                b[i] = -1.0
        # Final consumer of sum_val so it isn't optimised away.
        b[0] = sum_val + b[0]

    sdfg = kernel.to_sdfg(simplify=True)
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res is None


# -----------------------------------------------------------------------------
# Cross-pass non-interference.
# -----------------------------------------------------------------------------

def test_doesnt_lift_an_argmax_loop():
    """Argmax conditional has a different shape (writes the array's value
    AND optionally an index, not an accumulator OP). LoopToConditionalReduce
    must refuse so ArgMaxLift handles it."""

    @dace.program
    def s314(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] > x:
                x = a[i]
        result[0] = x

    sdfg = s314.to_sdfg(simplify=True)
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res is None


def test_doesnt_lift_a_break_loop():
    """A break-loop has a BreakBlock in the conditional, not an accumulator
    update; the matcher refuses (the true-branch's terminal AN check
    eliminates this shape -- a BreakBlock is not an AccessNode)."""

    @dace.program
    def s481(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]

    sdfg = s481.to_sdfg(simplify=True)
    res = LoopToConditionalReduce().apply_pass(sdfg, {})
    assert res is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
