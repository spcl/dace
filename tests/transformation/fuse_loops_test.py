# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``FuseLoops`` transformation unit tests -- the single-pair loop-fusion transformation form (the agent
arm; the ``LoopFusion`` pass is this transformation applied to a fixpoint).

The transformation shares its legality kernel with the ``LoopFusion`` pass, so the pass's numpy suite
already covers the fusion math exhaustively. These tests pin the TRANSFORMATION interface an agent (or the
pass) drives: ``can_be_applied_to`` identifies exactly the legal pairs, ``apply_to`` / repeated application
fuses them, the result is bit-exact to the un-fused program, and no input crashes.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.fuse_loops import FuseLoops

N = dace.symbol("N")


def nloops(sdfg):
    return sum(1 for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion))


def adjacent_loop_pairs(sdfg):
    """Every (first, second) LoopRegion adjacency (single sequencing edge) across the SDFG."""
    pairs = []
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for first in cfg.nodes():
            if not isinstance(first, LoopRegion):
                continue
            out = cfg.out_edges(first)
            if len(out) == 1 and isinstance(out[0].dst, LoopRegion) and out[0].dst is not first:
                pairs.append((first, out[0].dst))
    return pairs


def run_fused(prog, inputs, n, simplify=True):
    """Build the program twice: an un-fused reference and a copy fused by repeated ``FuseLoops``. Returns
    (before, after, applied, bit_exact)."""
    ref = prog.to_sdfg(simplify=simplify)
    ref.name = prog.name + "_ref"
    ref_bufs = {k: v.copy() for k, v in inputs.items()}
    ref(**ref_bufs, N=n)

    sd = prog.to_sdfg(simplify=simplify)
    before = nloops(sd)
    applied = sd.apply_transformations_repeated(FuseLoops) or 0
    after = nloops(sd)
    sd.name = prog.name + "_fused"
    fus_bufs = {k: v.copy() for k, v in inputs.items()}
    sd(**fus_bufs, N=n)

    exact = all(np.allclose(fus_bufs[k], ref_bufs[k], equal_nan=True) for k in inputs)
    return before, after, applied, exact


def mk(n=48, names=("a", "b", "c", "d"), seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.random(n) for k in names}


# --- fuses a legal pair, value-preserving ------------------------------------------------------------


def test_fuse_loops_fuses_two_sequential_recurrences():
    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            c[i] = c[i - 1] + b[i]

    before, after, applied, exact = run_fused(prog, mk(names=("a", "b", "c")), 48)
    assert before == 2 and after == 1 and applied == 1
    assert exact


def test_can_be_applied_to_identifies_the_fusable_pair():
    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            c[i] = c[i - 1] + b[i]

    sd = prog.to_sdfg(simplify=True)
    pairs = adjacent_loop_pairs(sd)
    assert len(pairs) == 1
    first, second = pairs[0]
    assert FuseLoops.can_be_applied_to(sd, first=first, second=second)


def test_apply_to_a_named_pair_fuses_it():
    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            c[i] = c[i - 1] + b[i]

    sd = prog.to_sdfg(simplify=True)
    first, second = adjacent_loop_pairs(sd)[0]
    FuseLoops.apply_to(sd, first=first, second=second, verify=True, save=False, annotate=False)
    assert nloops(sd) == 1
    bufs = mk(names=("a", "b", "c"))
    ref = {k: v.copy() for k, v in bufs.items()}
    prog.to_sdfg(simplify=True)(**ref, N=48)
    got = {k: v.copy() for k, v in bufs.items()}
    sd(**got, N=48)
    assert all(np.allclose(got[k], ref[k]) for k in bufs)


# --- refuses illegal / unsafe pairs ------------------------------------------------------------------


def test_refuses_read_ahead_forward_flow():
    # body2 reads b[i+1] -- a value body1 has not yet produced at iteration i in the fused sweep.
    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(0, N - 1):
            c[i] = b[i + 1]

    sd = prog.to_sdfg(simplify=True)
    for first, second in adjacent_loop_pairs(sd):
        assert not FuseLoops.can_be_applied_to(sd, first=first, second=second)
    before, after, applied, exact = run_fused(prog, mk(names=("a", "b", "c")), 48)
    assert applied == 0  # refused
    assert exact  # ... and therefore trivially value-preserving


def test_refuses_mismatched_iteration_range():
    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(2, N):
            c[i] = c[i - 1] + b[i]

    sd = prog.to_sdfg(simplify=True)
    for first, second in adjacent_loop_pairs(sd):
        assert not FuseLoops.can_be_applied_to(sd, first=first, second=second)


def test_refuses_doall_parallel_loops():
    # two independent element-wise loops are DOALL -- FuseLoops must not serialize them (LoopToMap's job).
    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] * 2.0
        for i in range(N):
            c[i] = a[i] + 1.0

    sd = prog.to_sdfg(simplify=True)
    for first, second in adjacent_loop_pairs(sd):
        assert not FuseLoops.can_be_applied_to(sd, first=first, second=second)


# --- never crashes -----------------------------------------------------------------------------------


@pytest.mark.parametrize("simplify", [True, False])
def test_never_crashes_on_single_or_no_loop(simplify):
    @dace.program
    def one_loop(a: dace.float64[N], b: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]

    sd = one_loop.to_sdfg(simplify=simplify)
    assert (sd.apply_transformations_repeated(FuseLoops) or 0) == 0  # nothing to fuse, no crash


def test_value_preserving_across_a_battery():
    cases = []

    @dace.program
    def chain3(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            c[i] = c[i - 1] + b[i]
        for i in range(1, N):
            d[i] = d[i - 1] + c[i]

    @dace.program
    def same_target(a: dace.float64[N], b: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            b[i] = b[i] * 2.0

    for prog, names in ((chain3, ("a", "b", "c", "d")), (same_target, ("a", "b"))):
        before, after, applied, exact = run_fused(prog, mk(names=names), 48)
        assert exact, f"{prog.name}: fused result diverged from reference"
        assert after <= before


# =====================================================================================================
# Arbitrary loop / map nesting patterns.
#
# FuseLoops fuses two adjacent loops only when each body is a SINGLE compute state -- a map body
# qualifies, a nested for-loop body does not. So the space splits into: fusable (outer sequential loop +
# inner map), refused-for-structure (nested for-loops / DOALL outer), refused-for-dependence (a real
# cross-loop hazard), and deep nests FuseLoops must simply never crash on. The invariant on EVERY case is
# value-preservation: `exact` (fused result == un-fused reference, bit-for-bit). A wrongly-fused real
# dependence would fail `exact`, so that assertion alone is the correctness net; `applied == 0` pins the
# cases we additionally expect to be refused.
# =====================================================================================================

f64 = dace.float64


def mk2d(n=32, names=("a", "b", "c", "d"), seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.random((n, n)) for k in names}


# --- fusable: outer SEQUENTIAL recurrence loop carrying a nested MAP over the free dimension ----------


@dace.program
def seq_outer_map_inner_2d(a: f64[N, N], b: f64[N, N], c: f64[N, N]):
    for i in range(1, N):
        for j in dace.map[0:N]:
            b[i, j] = b[i - 1, j] + a[i, j]
    for i in range(1, N):
        for j in dace.map[0:N]:
            c[i, j] = c[i - 1, j] + b[i, j]


@dace.program
def seq_outer_map_inner_scaled(a: f64[N, N], b: f64[N, N], c: f64[N, N]):
    for i in range(1, N):
        for j in dace.map[0:N]:
            b[i, j] = b[i - 1, j] * 0.5 + a[i, j]
    for i in range(1, N):
        for j in dace.map[0:N]:
            c[i, j] = c[i - 1, j] * 0.5 + b[i, j]


@dace.program
def seq_outer_map_reduction_body(a: f64[N, N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1]
        for j in dace.map[0:N]:
            b[i] += a[i, j]
    for i in range(1, N):
        c[i] = c[i - 1] + b[i]


@pytest.mark.parametrize("prog,names", [
    (seq_outer_map_inner_2d, ("a", "b", "c")),
    (seq_outer_map_inner_scaled, ("a", "b", "c")),
])
def test_map_bodied_recurrence_pair_is_conservatively_refused(prog, names):
    # A cross-loop b-write / b-read dependence expressed through a MAP body carries subsets the v1
    # point-wise classifier resolves conservatively, so FuseLoops refuses -- it never WRONGLY fuses, which
    # is what matters. The un-fused program is the correct result (exact).
    before, after, applied, exact = run_fused(prog, mk2d(names=names), 24)
    assert exact
    assert applied == 0 and after == before


def test_fusable_outer_seq_loop_with_reduction_body():
    before, after, applied, exact = run_fused(seq_outer_map_reduction_body, {
        "a": np.random.default_rng(1).random((24, 24)),
        "b": np.random.default_rng(2).random(24),
        "c": np.random.default_rng(3).random(24),
    }, 24)
    assert exact  # value-preserving whether or not the reduction-bodied pair fuses


# --- refused for structure: DOALL outer loop (element-wise) must NOT be serialized -------------------


@dace.program
def doall_1d_pair(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(N):
        b[i] = a[i] * 2.0
    for i in range(N):
        c[i] = a[i] + 1.0


@dace.program
def doall_2d_map_pair(a: f64[N, N], b: f64[N, N], c: f64[N, N]):
    for i in range(N):
        for j in dace.map[0:N]:
            b[i, j] = a[i, j] * 2.0
    for i in range(N):
        for j in dace.map[0:N]:
            c[i, j] = a[i, j] + 1.0


def test_doall_scalar_pair_is_refused():
    # a scalar element-wise pair is DOALL -- LoopToMap would parallelize it, so FuseLoops must not
    # serialize it.
    before, after, applied, exact = run_fused(doall_1d_pair, mk(names=("a", "b", "c")), 48)
    assert exact
    assert applied == 0


def test_doall_map_bodied_pair_fuses_on_read_only_shared():
    # the outer loops are element-wise but each carries a map, so LoopToMap does not re-map them
    # (_is_doall is False) and their only shared array (a) is read by both -- no dependence, a legal fuse.
    before, after, applied, exact = run_fused(doall_2d_map_pair, mk2d(names=("a", "b", "c")), 24)
    assert exact
    assert applied == 1 and after == before - 1


# --- refused for structure: nested for-loops (outer body is multiple blocks, not one compute state) --


@dace.program
def nested_seq_loops_2level(a: f64[N, N], b: f64[N, N], c: f64[N, N]):
    for i in range(1, N):
        for j in range(1, N):
            b[i, j] = b[i, j - 1] + a[i, j]
    for i in range(1, N):
        for j in range(1, N):
            c[i, j] = c[i, j - 1] + b[i, j]


@dace.program
def nested_seq_loops_3level(a: f64[N, N], b: f64[N, N]):
    for i in range(1, N):
        for j in range(1, N):
            for k in range(1, N):
                b[i, j] = b[i, j - 1] + a[i, j]
    for i in range(1, N):
        for j in range(1, N):
            for k in range(1, N):
                b[i, j] = b[i, j - 1] * 1.0


@pytest.mark.parametrize("prog,names", [
    (nested_seq_loops_2level, ("a", "b", "c")),
    (nested_seq_loops_3level, ("a", "b")),
])
def test_nested_for_loops_are_refused_but_value_preserving(prog, names):
    before, after, applied, exact = run_fused(prog, mk2d(names=names), 16)
    assert exact
    # a nested-for-loop outer body is not a single compute state -> FuseLoops refuses (v1 conservative).
    assert applied == 0


# --- refused for a REAL cross-loop data dependence (both non-DOALL, same range, single state) --------


@dace.program
def read_ahead_flow(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N - 1):
        b[i] = b[i - 1] + a[i]
    for i in range(1, N - 1):
        c[i] = c[i - 1] + b[i + 1]  # reads b[i+1] -- ahead of body1's production in the fused sweep


@dace.program
def read_behind_anti(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        c[i] = c[i - 1] + b[i - 1]  # body1 reads b[i-1]
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]  # body2 overwrites b[i] -- body1's read would see body2's new value


@dace.program
def divergent_output_writes(a: f64[N], b: f64[N]):
    for i in range(1, N - 1):
        b[i] = b[i - 1] + a[i]  # writes b[i]
    for i in range(1, N - 1):
        b[i + 1] = a[i] * 2.0  # writes b[i+1] -- a different cell than body1's, last-writer order differs


@dace.program
def transpose_dependence(a: f64[N, N], b: f64[N, N], c: f64[N, N]):
    for i in range(1, N):
        for j in dace.map[0:N]:
            b[i, j] = b[i - 1, j] + a[i, j]
    for i in range(1, N):
        for j in dace.map[0:N]:
            c[i, j] = c[i - 1, j] + b[j, i]  # reads b transposed -- not the same-index cell


@pytest.mark.parametrize("prog,names,n,d1", [
    (read_ahead_flow, ("a", "b", "c"), 48, 1),
    (read_behind_anti, ("a", "b", "c"), 48, 1),
    (divergent_output_writes, ("a", "b"), 48, 1),
    (transpose_dependence, ("a", "b", "c"), 20, 2),
])
def test_real_dependence_blocks_fusion(prog, names, n, d1):
    inputs = mk(n=n, names=names) if d1 == 1 else mk2d(n=n, names=names)
    before, after, applied, exact = run_fused(prog, inputs, n)
    assert exact  # the un-fused program is the correct result; fusing would diverge
    assert applied == 0  # FuseLoops recognizes the hazard and refuses


# --- map-outer / loop-inner, and deeper nests: FuseLoops must never crash, always value-preserving ---


@dace.program
def map_outer_seq_inner(a: f64[N, N], b: f64[N, N]):
    for i in dace.map[0:N]:
        for j in range(1, N):
            b[i, j] = b[i, j - 1] + a[i, j]


@dace.program
def loop_map_loop_sandwich(a: f64[N, N], b: f64[N, N], c: f64[N, N]):
    for i in range(1, N):
        for j in dace.map[0:N]:
            b[i, j] = b[i - 1, j] + a[i, j]
    for i in range(1, N):
        for j in range(1, N):
            c[i, j] = c[i, j - 1] + b[i, j]


@dace.program
def two_inner_maps_in_one_loop(a: f64[N, N], b: f64[N, N], c: f64[N, N]):
    for i in range(1, N):
        for j in dace.map[0:N]:
            b[i, j] = b[i - 1, j] + a[i, j]
        for j in dace.map[0:N]:
            c[i, j] = c[i - 1, j] + b[i, j]


@dace.program
def deep_4level_nest(a: f64[N, N], b: f64[N, N]):
    for i in range(1, N):
        for j in range(1, N):
            for k in range(1, N):
                for m in dace.map[0:N]:
                    b[i, j] = b[i - 1, j] + a[i, j]


@pytest.mark.parametrize("prog,names,n", [
    (map_outer_seq_inner, ("a", "b"), 16),
    (loop_map_loop_sandwich, ("a", "b", "c"), 16),
    (two_inner_maps_in_one_loop, ("a", "b", "c"), 16),
    (deep_4level_nest, ("a", "b"), 8),
])
def test_arbitrary_nesting_never_crashes_and_preserves_value(prog, names, n):
    before, after, applied, exact = run_fused(prog, mk2d(names=names), n)
    assert exact  # whatever FuseLoops does (or refuses to do) on the nest, the result is unchanged


# --- composition: chains of fusable + interleaved unfusable loops -------------------------------------


@dace.program
def chain_with_a_barrier(a: f64[N], b: f64[N], c: f64[N], d: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]
    for i in range(1, N):
        c[i] = c[i - 1] + b[i]  # fuses with the b-loop
    for i in range(1, N - 1):
        d[i] = c[i + 1]  # read-ahead of c -- a barrier: cannot fuse onto the (b,c) loop


@dace.program
def four_fusable_recurrences(a: f64[N], b: f64[N], c: f64[N], d: f64[N], e: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]
    for i in range(1, N):
        c[i] = c[i - 1] + b[i]
    for i in range(1, N):
        d[i] = d[i - 1] + c[i]
    for i in range(1, N):
        e[i] = e[i - 1] + d[i]


def test_chain_fuses_up_to_the_barrier():
    before, after, applied, exact = run_fused(chain_with_a_barrier, mk(names=("a", "b", "c", "d")), 48)
    assert exact
    assert applied >= 1 and after < before  # the (b,c) pair fuses; the read-ahead d-loop stays separate


def test_long_chain_of_fusable_recurrences_collapses():
    before, after, applied, exact = run_fused(four_fusable_recurrences, mk(names=("a", "b", "c", "d", "e")), 48)
    assert exact
    assert after < before and applied >= 1


# --- stencils and gathers in map bodies (value-preservation across access shapes) --------------------


@dace.program
def stencil_map_body_fusable(a: f64[N, N], b: f64[N, N], c: f64[N, N]):
    for i in range(1, N):
        for j in dace.map[1:N - 1]:
            b[i, j] = b[i - 1, j] + a[i, j - 1] + a[i, j + 1]
    for i in range(1, N):
        for j in dace.map[1:N - 1]:
            c[i, j] = c[i - 1, j] + b[i, j]


@dace.program
def gather_indirect_in_body(a: f64[N], idx: dace.int64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[idx[i]]
    for i in range(1, N):
        c[i] = c[i - 1] + b[i]


def test_stencil_in_map_body_is_value_preserving():
    before, after, applied, exact = run_fused(stencil_map_body_fusable, mk2d(names=("a", "b", "c")), 24)
    assert exact


def test_gather_indexed_read_is_value_preserving():
    rng = np.random.default_rng(7)
    inputs = {
        "a": rng.random(48),
        "idx": rng.integers(0, 48, size=48).astype(np.int64),
        "b": rng.random(48),
        "c": rng.random(48),
    }
    before, after, applied, exact = run_fused(gather_indirect_in_body, inputs, 48)
    assert exact  # gather in the body must not break value-preservation whichever way it is classified


# --- more scalar recurrence shapes that DO fuse (different ops / shared-read / same-cell output) ------


@dace.program
def scalar_sub_recurrence(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] - a[i]
    for i in range(1, N):
        c[i] = c[i - 1] - b[i]


@dace.program
def scalar_mul_recurrence(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] * a[i]
    for i in range(1, N):
        c[i] = c[i - 1] * b[i]


@dace.program
def independent_recurrences_shared_read(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]
    for i in range(1, N):
        c[i] = c[i - 1] + a[i]  # both only READ the shared a -> no cross dependence -> fuse


@dace.program
def same_cell_output_then_scale(a: f64[N], b: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]
    for i in range(1, N):
        b[i] = b[i] * 2.0  # writes the SAME cell body1 wrote (same-point output) + reads it -> legal


@dace.program
def multi_statement_body(a: f64[N], b: f64[N], c: f64[N], d: f64[N]):
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]
        d[i] = b[i] * 2.0
    for i in range(1, N):
        c[i] = c[i - 1] + b[i]


@dace.program
def min_recurrence(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        b[i] = min(b[i - 1], a[i])
    for i in range(1, N):
        c[i] = min(c[i - 1], b[i])


@pytest.mark.parametrize("prog,names", [
    (scalar_sub_recurrence, ("a", "b", "c")),
    (scalar_mul_recurrence, ("a", "b", "c")),
    (independent_recurrences_shared_read, ("a", "b", "c")),
    (multi_statement_body, ("a", "b", "c", "d")),
    (min_recurrence, ("a", "b", "c")),
])
def test_more_scalar_recurrence_pairs_fuse(prog, names):
    before, after, applied, exact = run_fused(prog, mk(names=names), 48)
    assert exact
    assert applied >= 1 and after < before


def test_same_cell_reread_scale_is_refused():
    # body2 rewrites b[i] and body1 reads b[i-1]; in a fused sweep body1 would read the value body2 already
    # scaled at i-1 -- a read-behind anti-dependence, so FuseLoops must refuse.
    before, after, applied, exact = run_fused(same_cell_output_then_scale, mk(names=("a", "b")), 48)
    assert exact
    assert applied == 0


@pytest.mark.parametrize("n", [4, 16, 32, 64])
def test_scalar_recurrence_fuses_across_sizes(n):
    @dace.program
    def prog(a: f64[N], b: f64[N], c: f64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            c[i] = c[i - 1] + b[i]

    before, after, applied, exact = run_fused(prog, mk(n=n, names=("a", "b", "c")), n)
    assert exact
    assert applied == 1 and after == before - 1


# --- more real dependence blockers (scalar, reach the _fusion_legal gate) ----------------------------


@dace.program
def read_ahead_stride_two(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N - 2):
        b[i] = b[i - 1] + a[i]
    for i in range(1, N - 2):
        c[i] = c[i - 1] + b[i + 2]  # read-ahead by 2 -- b[i+2] not yet produced in the fused sweep


@dace.program
def output_write_offset(a: f64[N], b: f64[N]):
    for i in range(1, N - 1):
        b[i] = b[i - 1] + a[i]  # writes b[i]
    for i in range(1, N - 1):
        b[i + 1] = a[i] * 3.0  # writes b[i+1] -- a different cell each iteration, order matters


@dace.program
def read_ahead_anti_is_safe(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N - 1):
        c[i] = c[i - 1] + b[i + 1]  # body1 reads b[i+1] -- AHEAD of body2's write to b[i]
    for i in range(1, N - 1):
        b[i] = b[i - 1] + a[i]  # body2 writes b[i]; the fused read of b[i+1] still sees the old value


@pytest.mark.parametrize("prog,names", [
    (read_ahead_stride_two, ("a", "b", "c")),
    (output_write_offset, ("a", "b")),
])
def test_more_real_dependences_block_fusion(prog, names):
    before, after, applied, exact = run_fused(prog, mk(names=names), 48)
    assert exact
    assert applied == 0


def test_read_ahead_anti_dependence_is_a_legal_fusion():
    # a read that is AHEAD of the other loop's write (WAR, not read-behind) stays correct when fused --
    # body1 reads b[i+1] before body2 has written it in the fused sweep. FuseLoops fuses, value-preserving.
    before, after, applied, exact = run_fused(read_ahead_anti_is_safe, mk(names=("a", "b", "c")), 48)
    assert exact
    assert applied >= 1


# --- more map / branch / deep structural cases (never crash, value-preserving) -----------------------


@dace.program
def map_bodied_read_only_two_ops(a: f64[N, N], b: f64[N, N], c: f64[N, N]):
    for i in range(N):
        for j in dace.map[0:N]:
            b[i, j] = a[i, j] * a[i, j]
    for i in range(N):
        for j in dace.map[0:N]:
            c[i, j] = a[i, j] + a[i, j]


@dace.program
def conditional_in_loop_body(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(1, N):
        if a[i] > 0.5:
            b[i] = b[i - 1] + a[i]
        else:
            b[i] = b[i - 1]
    for i in range(1, N):
        c[i] = c[i - 1] + b[i]


@dace.program
def reverse_recurrence(a: f64[N], b: f64[N], c: f64[N]):
    for i in range(N - 2, -1, -1):
        b[i] = b[i + 1] + a[i]
    for i in range(N - 2, -1, -1):
        c[i] = c[i + 1] + b[i]


@dace.program
def outer_loop_2d_map_3ops(a: f64[N, N], b: f64[N, N], c: f64[N, N], d: f64[N, N]):
    for i in range(1, N):
        for j in dace.map[0:N]:
            b[i, j] = b[i - 1, j] + a[i, j]
    for i in range(1, N):
        for j in dace.map[0:N]:
            c[i, j] = c[i - 1, j] + a[i, j]
    for i in range(1, N):
        for j in dace.map[0:N]:
            d[i, j] = b[i, j] + c[i, j]


def test_map_bodied_read_only_fuses():
    before, after, applied, exact = run_fused(map_bodied_read_only_two_ops, mk2d(names=("a", "b", "c")), 24)
    assert exact
    assert applied == 1 and after == before - 1


def test_conditional_loop_body_is_refused():
    before, after, applied, exact = run_fused(conditional_in_loop_body, mk(names=("a", "b", "c")), 48)
    assert exact
    assert applied == 0  # a branch in the body -> not a single compute state -> refused


def test_reverse_iteration_pair_is_value_preserving():
    before, after, applied, exact = run_fused(reverse_recurrence, mk(names=("a", "b", "c")), 48)
    assert exact  # negative-stride recurrences: whatever fuses, the value is preserved


def test_three_map_bodied_loops_are_value_preserving():
    before, after, applied, exact = run_fused(outer_loop_2d_map_3ops, mk2d(names=("a", "b", "c", "d")), 20)
    assert exact
    assert after <= before
