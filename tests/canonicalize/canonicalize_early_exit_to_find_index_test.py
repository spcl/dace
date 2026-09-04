# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.canonicalize.early_exit_to_find_index.EarlyExitToFindIndex`.

Covers TSVC ``s481`` (body_post + cond), ``s482`` (body_pre + cond), and the
v1 refusal contract (``s332`` true-branch scalar rebind unsupported; cond-body
overlap; multiple breaks; etc.). Also cross-pass non-interference: the
break-loop pass must not fire on argmax / reduce / scan shapes.

Also pins HPCAgent-Bench's ``ext_break_capture`` (s332 with an index + value capture): a naive
port that puts the capture inside a ``break``-less ``omp for`` lets every thread race to write the
shared answer, so a LATER (larger) firing index can overwrite the true first one -- exactly the
bug five independent agent submissions hit on this kernel. ``dace::find_first_index``'s
``reduction(min : best)`` (``dace/runtime/include/dace/detect.h:179``) is what keeps DaCe's own
lowering race-free; the tests below pin that this stays correct on every firing edge.
"""
import ast
import pathlib

import typing

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.libraries.standard.nodes import FindFirst, Reduce
from dace.sdfg import nodes as nd
from dace.transformation.passes.canonicalize.early_exit_to_find_index import EarlyExitToFindIndex
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.interstate import LoopToMap

N = dace.symbol('N')
M = dace.symbol('M')


def _num_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _num_reduces(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce))


def _find_first_nodes(sdfg):
    """The lifted searches: a ``FindFirst`` library node, or -- once expanded, which codegen does
    on its way out -- the tasklet whose body calls ``dace::find_first_index``. Matching the call
    and not merely a node type pins that the search still goes through the runtime primitive
    rather than a loop the pass grew back."""
    out = []
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, FindFirst):
            out.append(node)
        elif isinstance(node, nd.Tasklet) and 'dace::find_first_index' in node.code.as_string:
            out.append(node)
    return out


def _num_search_maps(sdfg):
    return len(_find_first_nodes(sdfg))


def _num_maps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry))


# -----------------------------------------------------------------------------
# Positive: TSVC s481 (body_post + cond) and s482 (body_pre + cond).
# -----------------------------------------------------------------------------


def test_tsvc_s481_break_then_body_post():
    """``for i: if d[i] < 0: break; a[i] += b[i] * c[i]``.
    Lifts to the Phase-1 chunked search Map + a Phase-2b body_post LoopRegion.
    The pass emits the body as a LoopRegion for a downstream ``LoopToMap`` to
    parallelize; after that lift the shape is 0 loops, 0 Reduces, 2 Maps
    (chunked search + body_post), the search Map being the WCR(Min) one.
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
    # The pass emits the body as a LoopRegion; the downstream LoopToMap lifts it
    # to the parallel body_post Map (part of the normal canonicalize pipeline).
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert _num_loops(sdfg) == 0
    assert _num_reduces(sdfg) == 0
    assert _num_maps(sdfg) == 1  # the body Map; the search is a library node, not a Map
    assert _num_search_maps(sdfg) == 1

    n = 16
    rng = np.random.default_rng(481)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    d = rng.standard_normal(n)
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
    Lifts to Phase-1 + a Phase-2a body_pre LoopRegion. The body_pre upper bound
    is ``min(exit_i + 1, N)`` so the last iteration before the break still runs
    its pre-check work. The downstream ``LoopToMap`` lifts the body_pre LoopRegion
    to a Map (0 loops, 0 Reduces, 2 Maps: chunked search + body_pre).
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
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert _num_loops(sdfg) == 0
    assert _num_reduces(sdfg) == 0
    assert _num_maps(sdfg) == 1  # the body Map; the search is a library node, not a Map
    assert _num_search_maps(sdfg) == 1

    n = 16
    rng = np.random.default_rng(482)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    a_ref = a.copy()
    for i in range(n):
        a_ref[i] = a_ref[i] + b[i] * c[i]
        if c[i] > b[i]:
            break
    a_got = a.copy()
    sdfg(a=a_got, b=b.copy(), c=c.copy(), N=n)
    assert np.allclose(a_got, a_ref)


def test_body_pre_and_body_post_combo_lifts():
    """Body has BOTH a body_pre (write ``e`` before the break check)
    AND a body_post (write ``a`` after the break check). The break
    condition reads ``d`` and the body never writes ``d``, so the
    disjointness check accepts. Both halves must lift: ``body_pre``
    over ``[0, min(exit_i+1, N))`` and ``body_post`` over ``[0, exit_i)``.
    The pass emits each as a LoopRegion; the downstream ``LoopToMap`` lifts both
    to Maps (0 loops, 0 Reduces, 3 Maps: chunked search + pre + post)."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], e: dace.float64[N]):
        for i in range(N):
            e[i] = a[i] + b[i]
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]

    sdfg = kernel.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1, 'body_pre + body_post combo must lift'
    # Two body LoopRegions (pre + post); the downstream LoopToMap lifts both.
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert _num_loops(sdfg) == 0
    assert _num_reduces(sdfg) == 0
    assert _num_search_maps(sdfg) == 1  # one FindFirst search node
    # phi/indicator + body_pre + body_post = 3 maps total
    assert _num_maps(sdfg) == 2, (f'expected 2 maps (body pre + post; the search is a library '
                                  f'node); got {_num_maps(sdfg)}')

    n = 16
    rng = np.random.default_rng(0xCAFE)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    d = np.linspace(1.0, -1.0, n)  # cond fires somewhere mid-range
    e = np.zeros(n)
    a_ref, e_ref = a.copy(), e.copy()
    for i in range(n):
        e_ref[i] = a_ref[i] + b[i]
        if d[i] < 0.0:
            break
        a_ref[i] = a_ref[i] + b[i] * c[i]
    sdfg(a=a, b=b.copy(), c=c.copy(), d=d.copy(), e=e, N=n)
    assert np.allclose(a, a_ref), f'body_post write diverged: max diff {np.abs(a - a_ref).max()}'
    assert np.allclose(e, e_ref), f'body_pre write diverged: max diff {np.abs(e - e_ref).max()}'


def test_body_pre_and_body_post_get_unique_block_names():
    """Regression: ``body_pre`` and ``body_post`` are both emitted as
    ``LoopRegion(label=f'{loop.label}_body')`` into the SAME parent CFG. They must be
    added with a unique name (``ensure_unique_name=True``); otherwise the second collides
    with the first and ``validate()`` raises ``Found multiple blocks with the same name``.
    Guards the whole class of "constructed CFG block added without the unique-name API"."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], e: dace.float64[N]):
        for i in range(N):
            e[i] = a[i] + b[i]  # body_pre  -> LoopRegion '<loop>_body'
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]  # body_post -> LoopRegion '<loop>_body' (must be uniquified)

    sdfg = kernel.to_sdfg(simplify=True)
    EarlyExitToFindIndex().apply_pass(sdfg, {})
    sdfg.validate()  # must NOT raise the duplicate-block-name error
    for cfg in sdfg.all_control_flow_regions():
        labels = [b.label for b in cfg.nodes()]
        assert len(labels) == len(set(labels)), f'{cfg.name}: duplicate block labels {labels}'


def test_refuses_break_cond_array_modified_pre_and_post():
    """The break check reads ``d[i]`` and the body writes ``d[i]`` both
    BEFORE and AFTER the check. The body's writes invalidate the
    Tier-Cheap whole-array disjointness, and parallel evaluation of
    the indicator map would race against the body's writes -- the
    pass must refuse."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            d[i] = a[i] + b[i] * c[i]
            if d[i] < 0.0:
                break
            d[i] = a[i] + b[i] * c[i]

    sdfg = kernel.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res is None, ('cond reads ``d`` and body writes ``d`` (pre and post): the '
                         'Tier-Cheap whole-array disjointness must refuse the lift')


def test_no_fire_runs_full_range():
    """Corner: ``cond`` is never true → ``exit_i = N`` (every chunk returns the
    sentinel and the WCR(Min) keeps it). All body iterations must run."""

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
    a = np.ones(n)
    b = np.full(n, 2.0)
    c = np.full(n, 3.0)
    d = np.full(n, 1.0)
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
    a = np.ones(n)
    b = np.full(n, 2.0)
    c = np.full(n, 3.0)
    d = np.full(n, -1.0)  # all negative
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
            a[i] = a[i] * 2.0  # writes a -- cond's read-set intersects

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
            a[i] = a[i - 1] + b[i]  # genuine loop-carried dep on `a`

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
    outer_loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable == 'i']
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
            d[i + 1] = d[i + 1] + 1.0  # writes ``d`` at i+1; cond reads d[i]
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
            a[i] = a[i + 1] + 1.0  # WAR on `a`

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
    """And the reverse: LoopToScan must not pick up a break-loop.

    Also pins the project rule -- a pass that does not apply must not mutate. The body
    here HAS a foldable copy tasklet, so a lifter that ran its own normalization
    preprocess would edit the graph and report ``0`` instead of ``None``."""
    from dace.transformation.passes.loop_to_scan import LoopToScan

    @dace.program
    def s481(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]

    sdfg = s481.to_sdfg(simplify=True)
    before = sdfg.to_json()
    res = LoopToScan().apply_pass(sdfg, {})
    assert res is None
    assert sdfg.to_json() == before, 'LoopToScan refused the loop but still mutated the SDFG'


def test_loop_to_reduce_doesnt_lift_break_loop():
    """LoopToReduce must not pick up a break-loop either -- in either emit mode, and
    without mutating (same rule as the LoopToScan case above)."""
    from dace.transformation.passes.loop_to_reduce import LoopToReduce

    @dace.program
    def s481(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]

    for prefer in ('reduce-libnode', 'wcr-scalar'):
        sdfg = s481.to_sdfg(simplify=True)
        before = sdfg.to_json()
        res = LoopToReduce(prefer=prefer).apply_pass(sdfg, {})
        assert res is None
        assert sdfg.to_json() == before, f'LoopToReduce({prefer}) refused the loop but still mutated the SDFG'


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


# -----------------------------------------------------------------------------
# simplify=False: the break predicate is a body-local transient scalar
# (``if __tmp0:`` where ``__tmp0 = a[i] > K`` is written by a body tasklet) and
# the arithmetic body spans several slice/binop states. The corpus gate runs
# simplify=True which collapses both shapes and hides this; these tests exercise
# the raw frontend shape directly.
# -----------------------------------------------------------------------------


def test_simplify_false_transient_predicate_resolves():
    """The resolver must inline the body tasklet ``__tmp0 = d_index < 0`` and
    the gather copy ``d_index = d[i]`` so a bare ``__tmp0`` predicate resolves
    to the array subscript ``d[i]`` the phi Map can read directly."""

    @dace.program
    def s481(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i] = a[i] + b[i] * c[i]

    sdfg = s481.to_sdfg(simplify=False)
    p = EarlyExitToFindIndex()
    loop = next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)
    cb = p._find_unique_break_conditional(loop)
    assert cb is not None
    bp, bpo = p._partition_body(loop, cb)
    _bindings, expr = p._resolve_cond_expression(loop, cb, bp + bpo, sdfg)
    # Predicate names ``d`` as a subscript and no bare transient survives.
    assert p._read_arrays_from_expr(expr, sdfg) == {'d'}
    assert p._cond_is_fully_resolved(expr, sdfg), f'predicate still unresolved: {expr!r}'


def assert_explicit_dataflow(sdfg):
    """HARD INVARIANT: every Tasklet reads each ``sdfg.arrays`` container through
    an in-connector -- no data container (array or scalar) is referenced by bare
    name in a tasklet body. A bare, unconnected ``__tmp0`` / ``threshold`` was
    the original miscompile."""
    for node, state in sdfg.all_nodes_recursive():
        if not isinstance(node, nd.Tasklet):
            continue
        try:
            tree = ast.parse(node.code.as_string)
        except SyntaxError:
            continue  # non-Python tasklet body -- not emitted by this pass
        owner = state.sdfg
        subscript_bases = {
            id(n.value)
            for n in ast.walk(tree) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
        }
        conns = set(node.in_connectors) | set(node.out_connectors)
        for name_node in ast.walk(tree):
            if not isinstance(name_node, ast.Name) or id(name_node) in subscript_bases:
                continue
            assert not (name_node.id in owner.arrays and name_node.id not in conns), (
                f'{node.label}: reads data container {name_node.id!r} by bare name without an '
                f'in-connector -- explicit-dataflow invariant violated: {node.code.as_string!r}')


# Matrix kernels: each is a break loop that MUST lift under BOTH simplify modes
# (s481 guard-before-body, s482 body-before-guard, s332 find-first + scalar
# capture). These are the ``ext_break_find_first`` / ``ext_break_post_body`` /
# ``ext_break_capture`` corpus shapes.


@dace.program
def ee_s481(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(N):
        if d[i] < 0.0:
            break
        a[i] = a[i] + b[i] * c[i]


@dace.program
def ee_s482(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N):
        a[i] = a[i] + b[i] * c[i]
        if c[i] > b[i]:
            break


@dace.program
def ee_s332(a: dace.float64[N], result: dace.float64[1], threshold: dace.float64):
    index = -2
    value = -1.0
    for i in range(N):
        if a[i] > threshold:
            index = i
            value = a[i]
            break
    result[0] = value + float(index)


def drive_s481(sdfg):
    n = 12
    for tag, d in (('fire', np.linspace(1.0, -1.0, n)), ('nofire', np.ones(n))):
        rng = np.random.default_rng(481)
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        c = rng.standard_normal(n)
        ref = a.copy()
        for i in range(n):
            if d[i] < 0.0:
                break
            ref[i] = ref[i] + b[i] * c[i]
        got = a.copy()
        sdfg(a=got, b=b.copy(), c=c.copy(), d=d.copy(), N=n)
        assert np.allclose(got, ref), f's481 {tag}: max diff {np.abs(got - ref).max()}'


def drive_s482(sdfg):
    n = 12
    rng = np.random.default_rng(482)
    b = rng.standard_normal(n)
    for tag, c in (('fire', b + np.linspace(-1.0, 1.0, n)), ('nofire', b - 1.0)):
        a = rng.standard_normal(n)
        ref = a.copy()
        for i in range(n):
            ref[i] = ref[i] + b[i] * c[i]
            if c[i] > b[i]:
                break
        got = a.copy()
        sdfg(a=got, b=b.copy(), c=c.copy(), N=n)
        assert np.allclose(got, ref), f's482 {tag}: max diff {np.abs(got - ref).max()}'


def drive_s332(sdfg):
    a = np.array([1.0, 2.0, 3.0, 50.0, 5.0, 6.0, 7.0, 8.0])
    result = np.zeros(1)
    sdfg(a=a, result=result, threshold=10.0, N=8)
    assert np.isclose(result[0], 53.0), f's332 fire: got {result[0]}, expected 53.0'  # index 3, value 50
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = np.zeros(1)
    sdfg(a=a, result=result, threshold=100.0, N=4)
    assert np.isclose(result[0], -3.0), f's332 no-fire: got {result[0]}, expected -3.0'  # index -2, value -1


_EE_MATRIX = {'s481': (ee_s481, drive_s481), 's482': (ee_s482, drive_s482), 's332': (ee_s332, drive_s332)}


@pytest.mark.parametrize('simplify', [True, False], ids=['sTrue', 'sFalse'])
@pytest.mark.parametrize('kernel', ['s481', 's482', 's332'])
def test_early_exit_lifts_both_simplify_modes(kernel, simplify):
    """Each early-exit kernel MUST lift (accelerate) under BOTH simplify=True and
    simplify=False -- the multi-state arithmetic body and the transient-scalar
    predicate are now emitted/inlined, not refused. Asserts: the pass lifts
    (res==1), the explicit-dataflow invariant holds on the lifted output, the
    residual body parallelizes to 0 loops, and the result is bit-exact (fire +
    no-fire)."""
    program, driver = _EE_MATRIX[kernel]
    sdfg = program.to_sdfg(simplify=simplify)

    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res == 1, f'{kernel} (simplify={simplify}) must lift, not refuse'
    sdfg.validate()
    # HARD INVARIANT on the lifted output: no dangling bare data read in any tasklet.
    assert_explicit_dataflow(sdfg)

    # Finish parallelization (the pipeline's SimplifyPass + LoopToMap): the
    # find-first body loop must collapse and lift to a Map -- 0 residual loops.
    SimplifyPass().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert _num_loops(sdfg) == 0, f'{kernel} (simplify={simplify}): break-loop + body must fully parallelize'
    assert _num_reduces(sdfg) == 0, f'{kernel} (simplify={simplify}): no separate find-first Reduce'
    assert _num_search_maps(sdfg) == 1, (f'{kernel} (simplify={simplify}): exactly one FindFirst '
                                         'search node')
    assert_explicit_dataflow(sdfg)  # invariant still holds after the full lift

    driver(sdfg)


def test_search_stays_parallel_and_cancels_through_canonicalize():
    """DESIGN RULING: the canonical early-exit form stays PARALLEL -- no sequential fallback, no
    profitability gate -- and the search itself is a library node over a runtime primitive, not a
    loop the pass writes by hand.

    Runs the FULL canonicalize pipeline (where ``MapToForLoop`` lowers every other Map to a
    LoopRegion) and pins the resulting shape and the emitted C++:

    * exactly one search survives, with 0 residual sequential loops;
    * it is a ``FindFirst`` library node, defaulting to the ``OpenMP`` implementation, so the
      chunked cancelling search runs threaded (``parallel=true`` at the call site);
    * the expanded body is ONE ``dace::find_first_index`` call and contains no loop of its own --
      the chunking, the shared cancellation hint and the blocked ``simd`` scan live in
      ``dace/runtime/include/dace/detect.h``, where there is one implementation to tune;
    * canonicalize never emits ``omp critical``.
    """
    from dace.transformation.passes.canonicalize.pipeline import canonicalize

    sdfg = ee_s481.to_sdfg(simplify=True)
    canonicalize(sdfg)
    sdfg.validate()

    assert _num_loops(sdfg) == 0, 'the lifted form must not fall back to a sequential loop'
    searches = _find_first_nodes(sdfg)
    assert len(searches) == 1, f'expected exactly one search, got {len(searches)}'
    search = searches[0]
    assert isinstance(search, FindFirst), f'the search must stay a library node, got {type(search).__name__}'
    assert search.implementation in (None, 'OpenMP'), (
        f'the search must keep the parallel implementation, got {search.implementation!r}')
    assert FindFirst.default_implementation == 'OpenMP'

    code = '\n'.join(obj.clean_code for obj in sdfg.generate_code())
    call_lines = [ln for ln in code.splitlines() if 'dace::find_first_index' in ln]
    assert len(call_lines) == 1, f'expected exactly one find_first_index call, got {call_lines}'
    assert call_lines[0].rstrip().endswith('true);'), (
        f'the search must be lowered threaded (parallel=true), got {call_lines[0].strip()!r}')
    assert 'omp critical' not in code

    # The pass emits no scan of its own: the only loop in the search is the runtime function's.
    expanded = ee_s481.to_sdfg(simplify=True)
    canonicalize(expanded)
    expanded.expand_library_nodes()
    bodies = [n.code.as_string for n in _find_first_nodes(expanded)]
    assert len(bodies) == 1, f'expected one expanded search body, got {len(bodies)}'
    assert 'for (' not in bodies[0], f'the expansion must not carry a hand-written loop: {bodies[0]!r}'


def _detect_header_text():
    """The runtime header the search lowers to, located from the installed package."""
    return (pathlib.Path(dace.__file__).parent / 'runtime' / 'include' / 'dace' / 'detect.h').read_text()


def test_find_first_answer_is_a_reduction_not_the_shared_hint():
    """The answer and the cancellation hint must be two different variables.

    Folding them into one shared word makes the update a read-compare-write, which is not atomic
    as a whole: a thread that found a SMALLER index can have its write overwritten by a thread
    that found a larger one, and the search then answers a firing index that is not the first.
    The hint may lose updates (it only prunes); the answer may not, so it is a reduction. This is
    a source assertion because the numeric symptom needs a loaded machine to appear."""
    text = _detect_header_text()
    body = text.split('inline long long find_first_index(')[1]
    chunk_pragma = next(ln for ln in body.splitlines() if 'omp parallel for schedule' in ln)
    assert 'reduction(min : best)' in chunk_pragma, (f'the chunk loop must reduce the answer, got {chunk_pragma!r}')
    # Measured: guided and block static hand a contiguous prefix to one thread, so a hit inside
    # that prefix is found by a serial scan -- 5x slower on a hit 10% into a 1e7 range.
    assert 'schedule(dynamic, 1)' in chunk_pragma, f'the chunk schedule is a measured choice, got {chunk_pragma!r}'
    assert 'return best;' in body, 'find_first_index must return the reduction, never the shared hint'
    assert 'return hint;' not in body, 'returning the raced hint loses updates under load'


def test_find_first_is_exact_when_most_indices_fire():
    """Numeric guard for the same bug: with a dense firing tail every chunk finds something and
    they all race to publish, so a lost update answers too large an index and the body loop then
    runs too far. Repeated, because a race needs room to show."""
    sdfg = ee_s481.to_sdfg(simplify=True)
    assert EarlyExitToFindIndex().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    n = 4096
    rng = np.random.default_rng(4811)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    for trial in range(16):
        first = 1 + 7 * trial
        d = np.ones(n)
        d[first:] = -1.0  # every index from ``first`` on fires
        a = rng.standard_normal(n)
        ref = a.copy()
        ref[:first] = ref[:first] + b[:first] * c[:first]
        got = a.copy()
        sdfg(a=got, b=b.copy(), c=c.copy(), d=d.copy(), N=n)
        assert np.allclose(got, ref), (f'dense-firing trial {trial} (first={first}): the search answered a firing '
                                       f'index that is not the first; max diff {np.abs(got - ref).max()}')


def test_body_ordering_memlet_survives_the_body_clone():
    """``for i: if d[i] < 0: break; a[i, :] = 1.0`` -- the body clone must keep the map entry's
    ordering memlet, or the constant tasklet falls out of its scope."""

    @dace.program
    def s481m(a: dace.float64[N, M], d: dace.float64[N]):
        for i in range(N):
            if d[i] < 0.0:
                break
            a[i, :] = 1.0

    sdfg = s481m.to_sdfg(simplify=True)
    before = sum(1 for st in sdfg.states() for e in st.edges() if e.data.is_empty())
    assert before == 1, f'the fixture must carry exactly one ordering memlet, got {before}'

    assert EarlyExitToFindIndex().apply_pass(sdfg, {}) == 1
    after = sum(1 for st in sdfg.states() for e in st.edges() if e.data.is_empty())
    assert after == before, f'the body clone dropped the ordering memlet: {before} -> {after}'
    sdfg.validate()

    n, m = 8, 4
    rng = np.random.default_rng(4811)
    a = rng.random((n, m))
    d = rng.random(n) - 0.3
    expected = a.copy()
    for i in range(n):
        if d[i] < 0.0:
            break
        expected[i, :] = 1.0

    got = a.copy()
    sdfg(a=got, d=d, N=n, M=m)
    assert np.allclose(got, expected), f'max diff {np.abs(got - expected).max()}'


# -----------------------------------------------------------------------------
# ext_break_capture (HPCAgent-Bench s332: find-first PLUS index+value capture).
#
# Structurally this is NOT s481/s482: the whole loop body is the break-conditional
# itself, so body_pre and body_post are both empty and ``_emit_body_loop`` returns
# None for each (early_exit_to_find_index.py, the "no live body work to emit" guard).
# The only per-iteration work IS the search; the capture is a single indexed read
# at the winning index, done once in Phase 3 -- there is nothing left for a Map to
# run in parallel. A Map appearing here later is a regression, not a missed
# optimization.
#
# The captured value, not the found index, is the part a hand-parallelized port
# gets wrong: ``break`` cannot appear inside ``#pragma omp for``, so a naive port
# drops it and has every thread write ``out_index``/``out_value`` under the bare
# guard ``if (a[i] > K)``, with no ordering between threads. A thread that finds a
# LATER (larger) index can finish last and overwrite a thread that already wrote
# the true first-firing one -- the sibling kernels have no such capture, so this
# failure mode is unique to ext_break_capture. ``find_first_index``'s
# ``reduction(min : best)`` at ``detect.h:179`` is the fix a naive port is missing.
# -----------------------------------------------------------------------------

_EXT_BREAK_CAPTURE_THRESHOLD = 1.0


@dace.program
def ext_break_capture(a: dace.float64[N], out_index: dace.int64[1], out_value: dace.float64[1]):
    out_index[0] = -1
    out_value[0] = -1.0
    for i in range(N):
        if a[i] > _EXT_BREAK_CAPTURE_THRESHOLD:
            out_index[0] = i
            out_value[0] = a[i]
            break


def ext_break_capture_reference(a: np.ndarray, n: int, threshold: float) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Sequential numpy reference matching ``ext_break_capture`` statement for statement."""
    out_index = np.array([-1], dtype=np.int64)
    out_value = np.array([-1.0], dtype=np.float64)
    for i in range(n):
        if a[i] > threshold:
            out_index[0] = i
            out_value[0] = a[i]
            break
    return out_index, out_value


@pytest.fixture(scope='module')
def ext_break_capture_pipeline():
    """Run the production recipe once: canonicalize picks the parallel search form,
    finalize_for_target resolves its implementation, then compile. Shared across the
    structural test and every parametrized numeric case below -- rebuilding per case
    would only pay compiler time, not change what is asserted."""
    from dace.transformation.passes.canonicalize.pipeline import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    sdfg = ext_break_capture.to_sdfg(simplify=False)
    canonicalize(sdfg, target='cpu')
    finalize_for_target(sdfg, 'cpu')
    csdfg = sdfg.compile()
    return sdfg, csdfg


def test_ext_break_capture_canonicalizes_to_findfirst_only(ext_break_capture_pipeline):
    """Deliberately NOT xfail: this refusal is correct today and must stay correct (same
    contract as ``smt_required_parallel_test.test_colliding_scatter_stays_sequential``). A
    numeric check alone would not catch a regression here -- a future pass could grow a
    same-answer Map around the capture and still pass every numeric case while reintroducing
    exactly the unsynchronized-write race described above."""
    sdfg, _csdfg = ext_break_capture_pipeline

    assert _num_maps(sdfg) == 0, ('ext_break_capture has no per-iteration body independent of the '
                                  'search -- a Map here is a regression, not progress')
    searches = _find_first_nodes(sdfg)
    assert len(searches) == 1, f'expected exactly one FindFirst search, got {len(searches)}'
    search = searches[0]
    assert isinstance(search, FindFirst), f'the search must stay a library node, got {type(search).__name__}'
    assert search.implementation == 'OpenMP', (
        f'the search must resolve to the parallel implementation, got {search.implementation!r}')

    code = '\n'.join(obj.clean_code for obj in sdfg.generate_code())
    call_lines = [ln for ln in code.splitlines() if 'dace::find_first_index' in ln]
    assert len(call_lines) == 1, f'expected exactly one find_first_index call, got {call_lines}'
    assert call_lines[0].rstrip().endswith('true);'), (
        f'the search must be lowered threaded (parallel=true), not a serial fallback: {call_lines[0].strip()!r}')


@pytest.mark.parametrize('case', ['fires_at_first', 'fires_in_middle', 'fires_at_last', 'never_fires'])
def test_ext_break_capture_matches_oracle_on_every_edge(ext_break_capture_pipeline, case):
    """The four shapes a racing parallel port breaks differently: firing at the FIRST index
    (a body that ran ahead of the search would already have clobbered it), firing in the MIDDLE
    (the textbook race -- a later chunk finishing first must not win), firing at the LAST index
    (every earlier non-match must not stick), and NEVER firing (the -1 / -1.0 sentinels must
    survive -- a shared 'hint' used as the answer instead of ``reduction(min : ...)`` tends to
    drift off the true no-fire value under load). A single-case numeric check would only catch
    one of these failure modes."""
    n = 32
    rng = np.random.default_rng(332)
    a = rng.uniform(-1000.0, _EXT_BREAK_CAPTURE_THRESHOLD - 1e-3, n)
    if case == 'fires_at_first':
        a[0] = _EXT_BREAK_CAPTURE_THRESHOLD + 500.0
    elif case == 'fires_in_middle':
        a[n // 2] = _EXT_BREAK_CAPTURE_THRESHOLD + 500.0
    elif case == 'fires_at_last':
        a[-1] = _EXT_BREAK_CAPTURE_THRESHOLD + 500.0
    elif case == 'never_fires':
        pass  # every a[i] stays below the threshold
    else:
        raise AssertionError(case)

    ref_index, ref_value = ext_break_capture_reference(a, n, _EXT_BREAK_CAPTURE_THRESHOLD)

    _sdfg, csdfg = ext_break_capture_pipeline
    got_index = np.zeros(1, dtype=np.int64)
    got_value = np.zeros(1, dtype=np.float64)
    csdfg(a=a.copy(), out_index=got_index, out_value=got_value, N=n)

    assert got_index[0] == ref_index[0], f'{case}: index {got_index[0]} != oracle {ref_index[0]}'
    assert np.isclose(got_value[0], ref_value[0]), f'{case}: value {got_value[0]} != oracle {ref_value[0]}'

    if case == 'never_fires':
        # Explicit, not just oracle-equal: the sentinels are the value a racing implementation
        # drifts away from, so pin them by their literal value too.
        assert got_index[0] == -1
        assert got_value[0] == -1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
