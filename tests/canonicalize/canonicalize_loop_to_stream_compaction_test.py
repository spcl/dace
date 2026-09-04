# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.canonicalize.loop_to_stream_compaction.LoopToStreamCompaction`.

Covers the TSVC conditional-append kernels ``s341`` / ``s342`` / ``s343``, two synthetic
compactions (cursor bumped AFTER the writes, and a stride-2 cursor), the whole-nest claim
(``s343``'s cursor is carried across both levels, so the phases leave the sequential outer
loop) and its refusal on a triangular nest, the numerical contract against an executable
numpy oracle -- including the live-out cursor value and the zero-trip loop -- and one test
per refusal in the pass's contract.

The pass runs inside the canonicalize pipeline's ``loop_to_x`` stage, so every test drives the
full pipeline: a refusal that only holds when the pass is run in isolation is not a refusal.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.cpu_specialization import cpu_specialize
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import (LoopToStreamCompaction, IDX_PREFIX,
                                                                               MASK_PREFIX, TOTAL_PREFIX)

N = dace.symbol('N', dtype=dace.int64)
M = dace.symbol('M', dtype=dace.int64)


def lifted(sdfg: dace.SDFG) -> bool:
    """True iff the compaction lift fired -- it is the only source of these transients."""
    return any(name.startswith(MASK_PREFIX) for name in sdfg.arrays)


def num_maps(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry))


def num_loops(sdfg: dace.SDFG) -> int:
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def omp_parallel_for(sdfg: dace.SDFG) -> int:
    return sum(part.clean_code.count('#pragma omp parallel for') for part in sdfg.generate_code())


def nest_claimed(sdfg: dace.SDFG, levels: int) -> bool:
    """True iff ``mask`` carries one dimension per nest level -- the whole-nest claim."""
    return any(len(sdfg.arrays[name].shape) == levels for name in sdfg.arrays if name.startswith(MASK_PREFIX))


def phases_under_a_loop(sdfg: dace.SDFG) -> bool:
    """True if a loop re-enters a phase -- one fork/join pair per phase per outer iteration."""
    buffers = [name for name in sdfg.arrays if name.startswith((MASK_PREFIX, IDX_PREFIX))]
    for state in sdfg.all_states():
        if not any(node.data in buffers for node in state.data_nodes()):
            continue
        region = state.parent_graph
        while region is not None and region is not sdfg:
            if isinstance(region, LoopRegion) and region.loop_variable:
                return True
            region = region.parent_graph
    return False


def build(program) -> dace.SDFG:
    """Canonicalize, then specialize for the CPU -- the assertions below read emitted schedules.

    The two are separate stages: ``canonicalize`` takes parallel wherever the choice is open and
    ``cpu_specialize`` is what gives parallelism back for a scope that cannot pay for a region. A
    test that counts ``#pragma omp parallel for`` is asking the second question, so it has to run
    the second stage.
    """
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    cpu_specialize(sdfg)
    sdfg.validate()
    return sdfg


# -----------------------------------------------------------------------------
# Executable numpy oracles -- the reference is the sequential loop itself.
# -----------------------------------------------------------------------------


def oracle_s341(a, b):
    j = -1
    for i in range(b.shape[0]):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i]
    return j


def oracle_s342(a, b):
    j = -1
    for i in range(a.shape[0]):
        if a[i] > 0.0:
            j = j + 1
            a[i] = b[j]
    return j


def oracle_s343(aa, bb, flat):
    k = -1
    for i in range(aa.shape[0]):
        for j in range(aa.shape[0]):
            if bb[j, i] > 0.0:
                k = k + 1
                flat[k] = aa[j, i]
    return k


# -----------------------------------------------------------------------------
# Positive: the lift fires, parallelizes, and preserves values.
# -----------------------------------------------------------------------------


@dace.program
def s341_kernel(a: dace.float64[N], b: dace.float64[N], jout: dace.int64[1]):
    j = -1
    for i in range(N):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i]
    jout[0] = j


@dace.program
def s342_kernel(a: dace.float64[N], b: dace.float64[N]):
    j = -1
    for i in range(N):
        if a[i] > 0.0:
            j = j + 1
            a[i] = b[j]


@dace.program
def s343_kernel(aa: dace.float64[M, M], bb: dace.float64[M, M], flat: dace.float64[M * M]):
    k = -1
    for i in range(M):
        for j in range(M):
            if bb[j, i] > 0.0:
                k = k + 1
                flat[k] = aa[j, i]


def test_s341_lifts_and_parallelizes():
    """The textbook compaction: every loop becomes a Map, both phases emit an OMP parallel for."""
    sdfg = build(s341_kernel)
    assert lifted(sdfg)
    assert any(name.startswith(TOTAL_PREFIX) for name in sdfg.arrays)  # live-out published from a Reduce
    assert num_loops(sdfg) == 0
    assert num_maps(sdfg) >= 2  # phase 1 (mask) and phase 3 (scatter)
    assert omp_parallel_for(sdfg) >= 2


def test_s341_numerics_and_live_out():
    sdfg = build(s341_kernel)
    rng = np.random.default_rng(1341)
    for n in (1, 2, 31, 255):
        b = rng.standard_normal(n)
        want = np.zeros(n)
        want_j = oracle_s341(want, b)
        got = np.zeros(n)
        got_j = np.zeros(1, dtype=np.int64)
        sdfg(a=got, b=b.copy(), jout=got_j, N=n)
        assert np.array_equal(got, want)
        assert int(got_j[0]) == want_j


def test_s341_zero_trip_leaves_cursor_untouched():
    """``total`` comes from a Reduce over an empty range, so a zero-trip loop publishes ``c_in``."""
    sdfg = build(s341_kernel)
    got = np.zeros(1)
    got_j = np.zeros(1, dtype=np.int64)
    sdfg(a=got, b=np.zeros(0), jout=got_j, N=0)
    assert int(got_j[0]) == -1


def test_s342_cursor_on_the_read_side():
    """``a[i] = b[j]`` -- the cursor indexes the SOURCE; the guard read of ``a`` is hoisted safely."""
    sdfg = build(s342_kernel)
    assert lifted(sdfg)
    assert num_loops(sdfg) == 0
    assert omp_parallel_for(sdfg) >= 2

    rng = np.random.default_rng(1342)
    for n in (1, 17, 128):
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        want = a.copy()
        oracle_s342(want, b)
        got = a.copy()
        sdfg(a=got, b=b.copy(), N=n)
        assert np.array_equal(got, want)


def test_s343_lifts_the_whole_nest():
    """The cursor is carried across BOTH levels, so the nest is claimed whole.

    ``mask`` / ``rank`` carry one dimension per level and nothing re-enters a phase, so the
    kernel costs three passes instead of three per outer iteration -- and the extents stay
    SYMBOLIC, which the per-level shape is what buys: a linearized ``mask[M*i + j]`` would
    leave ``LoopToMap`` unable to prove ``|M| >= 1`` and refuse both the probe and phase 1.
    """
    sdfg = build(s343_kernel)
    assert lifted(sdfg)
    assert nest_claimed(sdfg, 2)
    assert not phases_under_a_loop(sdfg)
    assert num_loops(sdfg) == 0  # no sequential residue: both phases are Maps
    assert num_maps(sdfg) >= 2
    assert omp_parallel_for(sdfg) >= 2

    rng = np.random.default_rng(1343)
    for m in (1, 5, 24):
        aa = rng.standard_normal((m, m))
        bb = rng.standard_normal((m, m))
        want = np.zeros(m * m)
        oracle_s343(aa, bb, want)
        got = np.zeros(m * m)
        sdfg(aa=aa.copy(), bb=bb.copy(), flat=got, M=m)
        assert np.array_equal(got, want)


MM = 24


@dace.program
def s343_static_kernel(aa: dace.float64[MM, MM], bb: dace.float64[MM, MM], flat: dace.float64[MM * MM],
                       kout: dace.int64[1]):
    k = -1
    for i in range(MM):
        for j in range(MM):
            if bb[j, i] > 0.0:
                k = k + 1
                flat[k] = aa[j, i]
    kout[0] = k


def test_s343_static_extents_lift_the_whole_nest():
    """Literal extents take the same whole-nest path, and publish the live-out cursor from it.

    ``total`` is a ``Reduce`` over the shaped mask, so the cursor the nest leaves behind counts
    every taken iteration of BOTH levels.

    A literal extent is also a size the CPU cost model can read, and the two phases land on
    OPPOSITE sides of it: the mask phase is the 24x24 nest (576 iterations, above the 256-iteration
    break-even) and the scatter phase is the 24-iteration cursor sweep (below it). So the terminal
    ``cpu_specialize`` band must pin exactly one of them -- which is a sharper statement than a size
    where both fall the same way, since it fails if the band stops running, if it runs on the wrong
    phase, or if it stops reading iteration counts at all. That is a target decision and not a
    property of the lift, so the lift's own parallelism is read separately with the cost model
    switched off -- threshold 0, its documented A/B lever, which leaves the maps exactly as
    canonicalization built them.
    """
    sdfg = build(s343_static_kernel)
    assert lifted(sdfg)
    assert nest_claimed(sdfg, 2)
    assert not phases_under_a_loop(sdfg)
    assert num_maps(sdfg) >= 2
    # Exactly the 576-iteration mask phase; the 24-iteration scatter phase cannot pay for a fork.
    # The break-even is PINNED, not inherited: CalibrateCpuThresholds scales it by the host's core
    # count on purpose (``scale_for_team``), so on a 64-core node it becomes 2048 -- above BOTH
    # phases, and the test then reads 0 and blames the band for a decision the calibration made.
    # 512 rather than the 256 named above because calibration only leaves a key alone when it
    # DIFFERS from the schema default, and 256 IS that default: pinning to it is indistinguishable
    # from not pinning at all. Any value with 24 < v < 576 states the same thing about the band.
    with dace.config.set_temporary('compiler', 'cpu', 'parallel_min_work_per_region', value=512):
        assert omp_parallel_for(build(s343_static_kernel)) == 1
    with dace.config.set_temporary('compiler', 'cpu', 'parallel_min_work_per_region', value=0):
        assert omp_parallel_for(build(s343_static_kernel)) >= 2

    rng = np.random.default_rng(3431)
    for scale in (1.0, -1.0, 0.25):
        aa = rng.standard_normal((MM, MM))
        bb = rng.standard_normal((MM, MM)) * scale
        want = np.zeros(MM * MM)
        want_k = oracle_s343(aa, bb, want)
        got = np.zeros(MM * MM)
        got_k = np.zeros(1, dtype=np.int64)
        sdfg(aa=aa.copy(), bb=bb.copy(), flat=got, kout=got_k)
        assert np.array_equal(got, want)
        assert int(got_k[0]) == want_k


@dace.program
def triangular_nest(bb: dace.float64[MM, MM], flat: dace.float64[MM * MM], kout: dace.int64[1]):
    k = -1
    for i in range(MM):
        for j in range(i + 1):
            if bb[j, i] > 0.0:
                k = k + 1
                flat[k] = bb[j, i]
    kout[0] = k


def oracle_triangular(bb, flat):
    k = -1
    for i in range(bb.shape[0]):
        for j in range(i + 1):
            if bb[j, i] > 0.0:
                k = k + 1
                flat[k] = bb[j, i]
    return k


def test_triangular_nest_is_not_claimed_whole():
    """A level whose extent moves with an outer iterator cannot be a ``mask`` dimension."""
    sdfg = build(triangular_nest)
    assert not nest_claimed(sdfg, 2)

    rng = np.random.default_rng(3432)
    bb = rng.standard_normal((MM, MM))
    want = np.zeros(MM * MM)
    want_k = oracle_triangular(bb, want)
    got = np.zeros(MM * MM)
    got_k = np.zeros(1, dtype=np.int64)
    sdfg(bb=bb.copy(), flat=got, kout=got_k)
    assert np.array_equal(got, want)
    assert int(got_k[0]) == want_k


@dace.program
def bump_after_writes(b: dace.float64[N], c: dace.float64[N], out: dace.float64[N]):
    cnt = 0
    for i in range(N):
        if b[i] > c[i]:
            out[cnt] = b[i] * 2.0 + c[i]
            cnt = cnt + 1


def test_bump_after_writes():
    """Rebinding the cursor at guard entry keeps the body verbatim, so bump position is irrelevant."""
    sdfg = build(bump_after_writes)
    assert lifted(sdfg)
    assert num_loops(sdfg) == 0

    rng = np.random.default_rng(7)
    n = 200
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    want = np.zeros(n)
    cnt = 0
    for i in range(n):
        if b[i] > c[i]:
            want[cnt] = b[i] * 2.0 + c[i]
            cnt = cnt + 1
    got = np.zeros(n)
    sdfg(b=b.copy(), c=c.copy(), out=got, N=n)
    assert np.array_equal(got, want)


@dace.program
def stride_two_cursor(b: dace.float64[N], out: dace.float64[2 * N]):
    cnt = 0
    for i in range(N):
        if b[i] > 0.0:
            out[cnt] = b[i]
            out[cnt + 1] = -b[i]
            cnt = cnt + 2


def test_loop_invariant_step_of_two():
    """``cnt += 2`` is still a closed form ``c_in + 2 * rank[i]``; only a VARYING step breaks it."""
    sdfg = build(stride_two_cursor)
    assert lifted(sdfg)
    assert num_loops(sdfg) == 0

    rng = np.random.default_rng(9)
    n = 128
    b = rng.standard_normal(n)
    want = np.zeros(2 * n)
    cnt = 0
    for i in range(n):
        if b[i] > 0.0:
            want[cnt] = b[i]
            want[cnt + 1] = -b[i]
            cnt = cnt + 2
    got = np.zeros(2 * n)
    sdfg(b=b.copy(), out=got, N=n)
    assert np.array_equal(got, want)


FLAG = dace.symbol('FLAG', dtype=dace.int64)


@dace.program
def guard_without_preamble(a: dace.float64[N], b: dace.float64[N]):
    j = -1
    for i in range(N):
        if FLAG > 0:
            j = j + 1
            a[j] = b[i]


def test_guard_as_body_start_block():
    """A symbol-only guard leaves no preamble state, so there is no in-edge to rebind the cursor on.

    The rewrite has to add one. Without it the phase-3 copy keeps the carried cursor, ``LoopToMap``
    refuses it, and the scatter stays a sequential loop -- correct, but the lift bought nothing.
    """
    from dace.transformation.passes.canonicalize.pipeline import CANONICALIZE_STAGES

    sdfg = guard_without_preamble.to_sdfg(simplify=True)
    for label, factory in CANONICALIZE_STAGES:
        if label == 'loop_to_x':
            break
        for unit in factory():
            unit.apply_pass(sdfg, {})
    loop = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable][0]
    assert isinstance(loop.start_block, dace.sdfg.state.ConditionalBlock)  # no preamble to bind on

    assert LoopToStreamCompaction().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert num_loops(sdfg) == 0
    assert num_maps(sdfg) >= 2

    rng = np.random.default_rng(4242)
    n = 40
    b = rng.standard_normal(n)
    for flag in (1, 0):
        want = np.zeros(n)
        j = -1
        for i in range(n):
            if flag > 0:
                j = j + 1
                want[j] = b[i]
        got = np.zeros(n)
        sdfg(a=got, b=b.copy(), N=n, FLAG=flag)
        assert np.array_equal(got, want)


@pytest.mark.parametrize('pattern', ['all-taken', 'none-taken', 'alternating'])
def test_degenerate_masks(pattern):
    """An all-true, all-false or alternating guard must still match the sequential loop exactly."""
    sdfg = build(s341_kernel)
    n = 64
    b = {'all-taken': np.ones(n), 'none-taken': -np.ones(n), 'alternating': np.array([1.0, -1.0] * (n // 2))}[pattern]
    want = np.zeros(n)
    want_j = oracle_s341(want, b)
    got = np.zeros(n)
    got_j = np.zeros(1, dtype=np.int64)
    sdfg(a=got, b=b.copy(), jout=got_j, N=n)
    assert np.array_equal(got, want)
    assert int(got_j[0]) == want_j


# -----------------------------------------------------------------------------
# Refusals -- firing on any of these is a silent miscompile.
# -----------------------------------------------------------------------------


@dace.program
def bump_outside_guard(a: dace.float64[N], b: dace.float64[N]):
    j = -1
    for i in range(N):
        j = j + 1
        if b[i] > 0.0:
            a[j] = b[i]


def test_refuse_bump_outside_guard():
    """An unconditional bump advances on non-taken iterations, so ``rank`` is not the cursor."""
    assert not lifted(build(bump_outside_guard))


@dace.program
def data_dependent_step(a: dace.float64[N], b: dace.float64[N], c: dace.int64[N]):
    j = -1
    for i in range(N):
        if b[i] > 0.0:
            j = j + c[i]
            a[j] = b[i]


def test_refuse_data_dependent_step():
    """``c_in + K * rank[i]`` needs a loop-invariant ``K``; a per-iteration step is not a scan."""
    assert not lifted(build(data_dependent_step))


@dace.program
def cond_uses_cursor(a: dace.float64[N], b: dace.float64[N]):
    j = -1
    for i in range(N):
        if b[i] > 0.0 and j < 10:
            j = j + 1
            a[j] = b[i]


def test_refuse_cond_depends_on_cursor():
    """Phase 1 computes the mask before any cursor exists; a cursor-dependent guard is unavailable."""
    assert not lifted(build(cond_uses_cursor))


@dace.program
def inplace_compaction(Z: dace.float64[N], I: dace.int64[N]):
    c = 0
    for i in range(N):
        if I[i] > 0:
            Z[c] = Z[i]
            c = c + 1


def test_refuse_in_place_compaction():
    """The npbench ``mandelbrot2`` shape: the target is also the source, so the scatter would race it.

    Sequentially ``c <= i`` makes the overwrite safe; in parallel the iteration writing ``Z[4]``
    runs concurrently with the one reading ``Z[4]``.
    """
    assert not lifted(build(inplace_compaction))


@dace.program
def inplace_via_symbol(Z: dace.float64[N], I: dace.int64[N]):
    c = 0
    for i in range(N):
        if I[i] > 0:
            t = Z[i] * 1.0
            Z[c] = t
            c = c + 1


def test_refuse_in_place_compaction_through_a_scalar():
    """Same alias, routed through a temporary -- the read must be found wherever it is expressed."""
    assert not lifted(build(inplace_via_symbol))


@dace.program
def two_cursors(a: dace.float64[N], b: dace.float64[N], d: dace.float64[N], e: dace.float64[N]):
    j = -1
    k = -1
    for i in range(N):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i]
        if e[i] > 0.0:
            k = k + 1
            d[k] = e[i]


def test_refuse_multiple_cursors():
    """One mask counts one guard; two appends in one loop need two ranks."""
    assert not lifted(build(two_cursors))


@dace.program
def cond_reads_target(a: dace.float64[N], b: dace.float64[N]):
    j = -1
    for i in range(N):
        if a[i] > 0.0:
            j = j + 1
            a[j] = b[i]


def test_refuse_cond_depends_on_earlier_writes_to_target():
    """The guard reads the array the append writes, so hoisting the guard reads changes them."""
    assert not lifted(build(cond_reads_target))


@dace.program
def guard_read_offset_write(a: dace.float64[N], b: dace.float64[N]):
    j = -1
    for i in range(N - 1):
        if a[i] > 0.0:
            j = j + 1
            a[i + 1] = b[j]


def test_refuse_guard_read_against_offset_body_write():
    """Guard reads ``a[i]``, body writes ``a[i+1]``: iteration i+1's guard sees iteration i's write."""
    assert not lifted(build(guard_read_offset_write))


@dace.program
def body_carried_dependence(a: dace.float64[N], b: dace.float64[N], d: dace.float64[N]):
    j = -1
    for i in range(1, N):
        if b[i] > 0.0:
            j = j + 1
            a[j] = d[i - 1]
            d[i] = b[i]


def test_refuse_body_carried_dependence():
    """A carry the cursor knows nothing about: delegated to ``LoopToMap`` on the affine-cursor model."""
    assert not lifted(build(body_carried_dependence))


@dace.program
def has_else_arm(a: dace.float64[N], b: dace.float64[N]):
    j = -1
    for i in range(N):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i]
        else:
            a[0] = 0.0


def test_refuse_else_arm():
    """A two-armed guard appends on a path the single mask does not model."""
    assert not lifted(build(has_else_arm))


@dace.program
def guard_read_disjoint_write(a: dace.float64[N], b: dace.float64[N], X: dace.float64[2 * N]):
    j = -1
    for i in range(N):
        if X[i] > 0.0:
            j = j + 1
            a[j] = b[i]
            X[i + N] = 1.0


def test_refuse_guard_read_array_written_anywhere_by_the_body():
    """Deliberately conservative: the halves are disjoint, so the loop itself is parallelizable.

    The phase split is what this refuses. Phase 1 evaluates every guard before phase 3 runs any
    body write, and this pass licenses that only for a guard-read array the body touches at the
    identical injective point. A disjoint write happens to be harmless; proving that in general
    is a dependence test this pass does not do, so it refuses instead of guessing.
    """
    assert not lifted(build(guard_read_disjoint_write))


@dace.program
def continue_in_guard(a: dace.float64[N], b: dace.float64[N]):
    j = -1
    for i in range(N):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i]
            continue


def test_refuse_continue_in_guard():
    """Phase 1 re-executes the preamble and phase 3 the body; a control-flow escape breaks both."""
    assert not lifted(build(continue_in_guard))


@dace.program
def has_break(a: dace.float64[N], b: dace.float64[N]):
    j = -1
    for i in range(N):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i]
        if b[i] > 100.0:
            break


def test_refuse_break():
    """Phase 1 runs the whole range; an early exit makes the mask cover iterations that never ran."""
    assert not lifted(build(has_break))


def test_refuse_non_unit_stride():
    """``rank`` is indexed by ``i - start`` in unit steps, so a strided loop mis-indexes the mask."""

    @dace.program
    def strided(a: dace.float64[N], b: dace.float64[N]):
        j = -1
        for i in range(0, N, 2):
            if b[i] > 0.0:
                j = j + 1
                a[j] = b[i]

    sdfg = strided.to_sdfg(simplify=True)
    loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
    assert loops
    assert LoopToStreamCompaction().loop_extent(loops[0]) == (None, None)


def test_refuse_side_effecting_body():
    """Phase 1 re-executes the preamble, so a side effect in the loop would run twice.

    ``LoopToMap`` cannot rule on this -- it judges whether the loop's iterations are independent,
    not whether the loop may be evaluated twice -- so the purity gate is the only check. Same
    SDFG, same pipeline prefix, ``side_effects`` the only difference.
    """
    from dace.transformation.passes.canonicalize.pipeline import CANONICALIZE_STAGES

    def prefix_up_to_loop_to_x() -> dace.SDFG:
        sdfg = bump_after_writes.to_sdfg(simplify=True)
        for label, factory in CANONICALIZE_STAGES:
            if label == 'loop_to_x':
                return sdfg
            for unit in factory():
                unit.apply_pass(sdfg, {})
        return sdfg

    clean = prefix_up_to_loop_to_x()
    assert LoopToStreamCompaction().apply_pass(clean, {}) == 1

    marked = prefix_up_to_loop_to_x()
    loop = [r for r in marked.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable][0]
    tasklets = [n for state in loop.all_states() for n in state.nodes() if isinstance(n, nd.Tasklet)]
    assert tasklets
    for tasklet in tasklets:
        tasklet.side_effects = True
    assert LoopToStreamCompaction().apply_pass(marked, {}) is None


def test_refuse_pinned_sequential_loop():
    """``pinned_sequential`` is a directive from an earlier pass, not an obstacle to work around."""
    sdfg = s341_kernel.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert lifted(sdfg)

    pinned = s341_kernel.to_sdfg(simplify=True)
    for region in pinned.all_control_flow_regions():
        if isinstance(region, LoopRegion) and region.loop_variable:
            region.pinned_sequential = True
    assert LoopToStreamCompaction().apply_pass(pinned, {}) is None


def test_pass_is_a_no_op_when_it_refuses():
    """A pass that does not apply must not mutate: refusal leaves the SDFG byte-identical."""
    sdfg = build(inplace_compaction)
    before = sdfg.to_json()
    assert LoopToStreamCompaction().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
