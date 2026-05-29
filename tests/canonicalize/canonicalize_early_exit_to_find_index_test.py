# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.canonicalize.early_exit_to_find_index.EarlyExitToFindIndex`.

Covers TSVC ``s481`` (body_post + cond), ``s482`` (body_pre + cond), and the
v1 refusal contract (``s332`` true-branch scalar rebind unsupported; cond-body
overlap; multiple breaks; etc.). Also cross-pass non-interference: the
break-loop pass must not fire on argmax / reduce / scan shapes.
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.libraries.standard.nodes import Reduce
from dace.sdfg import nodes as nd
from dace.transformation.passes.canonicalize.early_exit_to_find_index import EarlyExitToFindIndex


N = dace.symbol('N')


def _num_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _num_reduces(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce))


def _num_maps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry))


# -----------------------------------------------------------------------------
# Positive: TSVC s481 (body_post + cond) and s482 (body_pre + cond).
# -----------------------------------------------------------------------------

def test_tsvc_s481_break_then_body_post():
    """``for i: if d[i] < 0: break; a[i] += b[i] * c[i]``.
    Lifts to Phase-1 indicator+Reduce(Min) + Phase-2b body_post Map.
    No body_pre; no true-branch rebind. Output: 0 loops, 1 Reduce, 2 Maps
    (phi build + body_post).
    """

    @dace.program
    def s481(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]

    sdfg = s481.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _num_loops(sdfg) == 0
    assert _num_reduces(sdfg) == 1
    assert _num_maps(sdfg) == 2

    n = 16
    rng = np.random.default_rng(481)
    a = rng.standard_normal(n); b = rng.standard_normal(n)
    c = rng.standard_normal(n); d = rng.standard_normal(n)
    a_ref = a.copy()
    for i in range(n):
        if d[i] < 0:
            break
        a_ref[i] = a_ref[i] + b[i] * c[i]
    a_got = a.copy()
    sdfg(a=a_got, b=b.copy(), c=c.copy(), d=d.copy(), N=n)
    assert np.allclose(a_got, a_ref)


def test_tsvc_s482_body_pre_then_break():
    """``for i: a[i] += b[i] * c[i]; if c[i] > b[i]: break``.
    Lifts to Phase-1 + Phase-2a body_pre Map. The body_pre upper bound is
    ``min(exit_i + 1, N)`` so the last iteration before the break still runs
    its pre-check work.
    """

    @dace.program
    def s482(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(N):
            a[i] = a[i] + b[i] * c[i]
            if c[i] > b[i]:
                break

    sdfg = s482.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _num_loops(sdfg) == 0
    assert _num_reduces(sdfg) == 1
    assert _num_maps(sdfg) == 2

    n = 16
    rng = np.random.default_rng(482)
    a = rng.standard_normal(n); b = rng.standard_normal(n); c = rng.standard_normal(n)
    a_ref = a.copy()
    for i in range(n):
        a_ref[i] = a_ref[i] + b[i] * c[i]
        if c[i] > b[i]:
            break
    a_got = a.copy()
    sdfg(a=a_got, b=b.copy(), c=c.copy(), N=n)
    assert np.allclose(a_got, a_ref)


def test_no_fire_runs_full_range():
    """Corner: ``cond`` is never true → ``exit_i = N`` (sentinel via Reduce(Min)
    identity = N). All body iterations must run."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]

    sdfg = kernel.to_sdfg(simplify=True)
    EarlyExitToFindIndex().apply_pass(sdfg, {})
    sdfg.validate()

    n = 8
    # All d positive -> no break.
    a = np.ones(n); b = np.full(n, 2.0); c = np.full(n, 3.0); d = np.full(n, 1.0)
    a_ref = a.copy()
    for i in range(n):
        a_ref[i] = a_ref[i] + b[i] * c[i]
    sdfg(a=a, b=b.copy(), c=c.copy(), d=d.copy(), N=n)
    assert np.allclose(a, a_ref)


def test_fire_at_first_iteration_no_body_run():
    """Corner: cond fires at ``i = 0`` -> no body iteration should run."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]

    sdfg = kernel.to_sdfg(simplify=True)
    EarlyExitToFindIndex().apply_pass(sdfg, {})
    sdfg.validate()

    n = 8
    a = np.ones(n); b = np.full(n, 2.0); c = np.full(n, 3.0); d = np.full(n, -1.0)  # all negative
    a_orig = a.copy()
    sdfg(a=a, b=b.copy(), c=c.copy(), d=d.copy(), N=n)
    assert np.allclose(a, a_orig), "no body iteration should have run"


# -----------------------------------------------------------------------------
# Refusal contract (v1 scope).
# -----------------------------------------------------------------------------

def test_refuses_s332_true_branch_scalar_rebind():
    """TSVC s332: the true-branch writes scalars ``index = i; value = a[i]``
    before the break. The post-Phase-1 conditional rebind state is not yet
    emitted by v1 -- the pass refuses so the loop stays sequential rather
    than producing wrong numerics."""

    @dace.program
    def s332(a: dace.float64[N], result: dace.float64[1], threshold: dace.int64):
        index = -2
        value = -1.0
        for i in range(N):
            if a[i] > threshold:
                index = i
                value = a[i]
                break
        result[0] = value + float(index)

    sdfg = s332.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res is None, "s332 must be refused until the rebind state lands in v1.5"


def test_refuses_cond_reads_array_body_writes():
    """Soundness (S2-tight): cond reads ``a``, body writes ``a``. The
    Tier-Cheap whole-array disjointness check refuses (parallel evaluation
    of cond would see stale ``a`` while the sequential one would have seen
    the body's mid-loop updates)."""

    @dace.program
    def kernel(a: dace.float64[N]):
        for i in range(N):
            if a[i] < 0.0:
                break
            a[i] = a[i] * 2.0   # writes a -- cond's read-set intersects

    sdfg = kernel.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res is None, "cond/body overlap on the same array must be refused"


def test_refuses_no_break():
    """Plain loop with no break is not the target shape."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            a[i] = a[i] + b[i]

    sdfg = kernel.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res is None


def test_refuses_non_unit_stride():
    """``for i in range(0, N, 2): if ...: break`` -- non-unit stride is
    refused (the Phase-1 indicator Map's iteration range expects stride 1)."""

    @dace.program
    def kernel(a: dace.float64[N], d: dace.float64[N]):
        for i in range(0, N, 2):
            if d[i] < 0.0:
                break
            a[i] = a[i] * 2.0

    sdfg = kernel.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res is None


# -----------------------------------------------------------------------------
# Cross-pass non-interference: the break-loop pass must not fire on any of
# the other loop-lift shapes.
# -----------------------------------------------------------------------------

def test_doesnt_lift_plain_reduction():
    """``for i: s = s + a[i]`` -- LoopToReduce shape, no break, no
    ConditionalBlock. EarlyExitToFindIndex must refuse."""

    @dace.program
    def kernel(a: dace.float64[N], result: dace.float64[1]):
        s = 0.0
        for i in range(N):
            s = s + a[i]
        result[0] = s

    res = EarlyExitToFindIndex().apply_pass(kernel.to_sdfg(simplify=True), {})
    assert res is None


def test_doesnt_lift_scan():
    """``for i: out[i+1] = out[i] + a[i]`` -- LoopToScan shape, no break."""

    @dace.program
    def kernel(a: dace.float64[N], out: dace.float64[N + 1]):
        for i in range(N):
            out[i + 1] = out[i] + a[i]

    res = EarlyExitToFindIndex().apply_pass(kernel.to_sdfg(simplify=True), {})
    assert res is None


def test_doesnt_lift_argmax_loop():
    """TSVC s314 ``for i: if a[i] > x: x = a[i]`` -- ArgMaxLift shape. The
    conditional has no BreakBlock, so EarlyExitToFindIndex refuses."""

    @dace.program
    def s314(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] > x:
                x = a[i]
        result[0] = x

    res = EarlyExitToFindIndex().apply_pass(s314.to_sdfg(simplify=True), {})
    assert res is None


def test_loop_to_scan_doesnt_lift_break_loop():
    """And the reverse: LoopToScan must not pick up a break-loop."""
    from dace.transformation.passes.loop_to_scan import LoopToScan

    @dace.program
    def s481(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]

    res = LoopToScan().apply_pass(s481.to_sdfg(simplify=True), {})
    assert res is None


def test_loop_to_reduce_doesnt_lift_break_loop():
    """LoopToReduce must not pick up a break-loop either."""
    from dace.transformation.passes.loop_to_reduce import LoopToReduce

    @dace.program
    def s481(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]

    res = LoopToReduce().apply_pass(s481.to_sdfg(simplify=True), {})
    assert res is None


def test_arg_max_lift_doesnt_lift_break_loop():
    """ArgMaxLift must not pick up a break-loop either."""
    from dace.transformation.passes.canonicalize.arg_max_lift import ArgMaxLift

    @dace.program
    def s481(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]

    res = ArgMaxLift().apply_pass(s481.to_sdfg(simplify=True), {})
    assert res is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
