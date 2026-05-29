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

def test_tsvc_s332_true_branch_scalar_rebind():
    """TSVC s332: the true-branch writes scalars ``index = i; value = a[i]``
    before the break. v1.5 emits a Phase-3 ConditionalBlock guarded by
    ``exit_sym < N`` whose true branch is a deep-copy of the original break
    branch with the loop variable substituted by ``exit_sym``. When ``cond``
    never fires the guard is false and the pre-loop initial values
    ``index = -2``, ``value = -1.0`` are preserved.
    """

    @dace.program
    def s332(a: dace.float64[N], result: dace.float64[1], threshold: dace.float64):
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
    sdfg.validate()
    assert res == 1

    # Fire case: cond fires at index 3.
    a = np.array([1.0, 2.0, 3.0, 50.0, 5.0, 6.0, 7.0, 8.0])
    result = np.zeros(1)
    sdfg(a=a, result=result, threshold=10.0, N=8)
    # index = 3, value = 50.0 -> result = 50 + 3.0 = 53.0
    assert np.isclose(result[0], 53.0), f"fire: got {result[0]}, expected 53.0"

    # No-fire case: all a[i] <= threshold.
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = np.zeros(1)
    sdfg(a=a, result=result, threshold=100.0, N=4)
    # index = -2 (init), value = -1.0 (init) -> result = -1 + (-2) = -3.0
    assert np.isclose(result[0], -3.0), f"no-fire: got {result[0]}, expected -3.0"


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
# Soundness refusals: shapes where lifting the break to a find-first reduction
# + parallel map would NOT preserve sequential semantics. All must be refused.
# -----------------------------------------------------------------------------

def test_refuses_body_loop_carried_dep():
    """Body has a real read-after-write carry on ``a`` (``a[i] = a[i-1] + ...``).
    The find-first lift would parallelize the body and race on the carry.
    Soundness gate S4 (delegated to ``LoopToMap.can_be_applied`` on the
    splice-out variant of the loop) refuses."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], d: dace.float64[N]):
        for i in range(1, N):
            if d[i] < 0.0:
                break
            a[i] = a[i - 1] + b[i]   # genuine loop-carried dep on `a`

    sdfg = kernel.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res is None, "body with loop-carried dep on `a` must be refused (S4)"


def test_refuses_cond_reads_scalar_body_writes():
    """``acc = 0; for i: if acc > T: break; acc += a[i]`` -- the break
    condition reads a SCALAR (``acc``) that the body mutates each iteration.
    The sequential find-first depends on the cumulative ``acc``; the parallel
    rewrite using the initial ``acc`` would compute a different ``exit_i``.

    Cond's read of ``acc`` (a transient scalar) intersects with the body's
    write to ``acc`` -- the Tier-Cheap whole-array gate catches this. (Even
    if scalars were exempt, the dep is real: this is the prefix-scan-find-
    first family, out of scope for the simple find-first lift.)"""

    @dace.program
    def kernel(a: dace.float64[N], result: dace.float64[1], threshold: dace.float64):
        acc = 0.0
        for i in range(N):
            if acc > threshold:
                break
            acc = acc + a[i]
        result[0] = acc

    sdfg = kernel.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res is None, "cond reading iteration-carried scalar must be refused"


def test_refuses_cond_reads_array_body_modifies():
    """``for i: if d[i] < 0: break; d[i] = ...`` -- cond reads ``d`` and the
    body ALSO writes ``d``. Parallel cond evaluation sees the initial
    ``d[i]``; sequential cond at iteration ``i`` would see whatever
    body[0..i-1] wrote to ``d[i]``. Different exit indices -> different
    semantics. Tier-Cheap whole-array disjointness refuses."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            d[i] = d[i] * 2.0 + b[i]
            a[i] = d[i] + b[i]

    sdfg = kernel.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res is None, "cond/body overlap on the same array must be refused"


def test_refuses_multiple_breaks():
    """Two break conditionals in the same body. The find-first would have
    to track multiple exit points; the v1 matcher refuses any loop whose
    body contains more than one BreakBlock-bearing ConditionalBlock."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            if c[i] > 100.0:
                break
            a[i] = a[i] + b[i] * c[i]

    sdfg = kernel.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res is None, "multiple breaks must be refused"


def test_refuses_break_inside_nested_loop():
    """Break inside a NESTED loop only escapes the inner loop -- the outer
    loop's full iteration semantics are preserved. This is not the
    find-first shape; the matcher must refuse the outer (whose body
    contains a nested loop, not a top-level break-conditional)."""

    @dace.program
    def kernel(aa: dace.float64[N, N], d: dace.float64[N, N]):
        for i in range(N):
            for j in range(N):
                if d[i, j] < 0.0:
                    break  # breaks ONLY the j loop
                aa[i, j] = aa[i, j] * 2.0

    sdfg = kernel.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    # The OUTER ``i`` loop has no top-level break (the break is in the inner ``j``).
    # The INNER ``j`` loop has a break + non-trivial body; whether it lifts
    # depends on the inner body's parallelizability, but it must NOT lift the
    # OUTER. Verify at minimum the outer stays a LoopRegion.
    outer_loops = [r for r in sdfg.all_control_flow_regions()
                   if isinstance(r, LoopRegion) and r.loop_variable == 'i']
    assert len(outer_loops) >= 1, "outer i-loop must NOT be lifted"


def test_refuses_body_write_overlaps_cond_array_at_offset():
    """Body writes ``d[i+1]`` while cond reads ``d[i]``. Tier-Cheap
    whole-array disjointness can't tell the indices are disjoint
    (``j+1`` vs ``i`` at parameterised offsets) -- a Tier-Affine
    polyhedral check could admit some cases, but v1 stays conservative
    and refuses on any whole-array overlap."""

    @dace.program
    def kernel(a: dace.float64[N], d: dace.float64[N + 1]):
        for i in range(N):
            if d[i] < 0.0:
                break
            d[i + 1] = d[i + 1] + 1.0    # writes ``d`` at i+1; cond reads d[i]
            a[i] = a[i] + d[i + 1]

    sdfg = kernel.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res is None, ("Tier-Cheap must refuse any cond/body whole-array "
                         "overlap; affine index analysis is a follow-up")


def test_refuses_war_anti_dep_in_body():
    """Body reads ``a[i+1]`` and writes ``a[i]`` -- a write-after-read
    anti-dependence within the loop. ``LoopToMap.can_be_applied`` refuses
    this kind of body; the delegated S4 check propagates the refusal."""

    @dace.program
    def kernel(a: dace.float64[N + 1], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i] = a[i + 1] + 1.0   # WAR on `a`

    sdfg = kernel.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res is None, "body with WAR anti-dep must be refused (S4 via LoopToMap)"


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
