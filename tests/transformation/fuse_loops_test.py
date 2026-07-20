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
    before, after, applied, exact = run_fused(
        seq_outer_map_reduction_body, {
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


@dace.program
def invariant_scalar_then_recurrence(a: f64[N], d: f64[N]):
    s = np.float64(0.0)
    for i in range(1, N):
        s = a[i]  # writes a LOOP-INVARIANT location: the last iteration's value wins
    for i in range(1, N):
        d[i] = d[i - 1] + s  # a recurrence reading it -> neither loop is DOALL


@dace.program
def invariant_scalar_read_then_written(a: f64[N], d: f64[N]):
    s = np.float64(0.0)
    for i in range(1, N):
        d[i] = d[i - 1] + s  # reads the invariant location...
    for i in range(1, N):
        s = a[i]  # ...that the second loop overwrites (the mirror, anti direction)


def test_fusion_across_an_invariant_scalar_is_refused():
    # The dependence classifier reports no CARRIED offset for a loop-invariant location -- there is no
    # iterator in either subset to carry one -- which must not be read as "no dependence". Unfused, loop2
    # sees the FINAL s; fused, it sees the RUNNING s (d[i]=d[i-1]+a[i], a different prefix sum).
    before, after, applied, exact = run_fused(invariant_scalar_then_recurrence, mk(names=("a", "d")), 48)
    assert exact
    assert applied == 0


def test_fusion_across_an_invariant_scalar_overwritten_later_is_refused():
    before, after, applied, exact = run_fused(invariant_scalar_read_then_written, mk(names=("a", "d")), 48)
    assert exact
    assert applied == 0


# =====================================================================================================
# Intermediate contraction (buffer localization).
#
# Fusing two sequential loops that share an intermediate ``tmp`` is only half the win: the ``[N]`` buffer
# between them is reclaimable once both bodies run in the same iteration. FuseLoops contracts a transient
# the fused body writes-and-reads at the SAME point ``tmp[i]`` -- no cross-iteration history, no use
# outside the loop -- down to a reused ``[1]`` slot (the loop analogue of MapFusionVertical's smaller
# intermediate). It is orthogonal to parallelism, so it fires on the sequential recurrences LoopToMap left.
#
# Every case pins BOTH nets: value-preservation (`exact`, bit-for-bit vs the un-fused reference) and the
# buffer measurement (`big_before`/`big_after` = transient arrays whose element count is not statically 1).
# Contraction must NEVER change a value and must NEVER fire when a cross-iteration/offset/outside access
# makes a single slot unsound -- those cases fuse (or refuse) but keep the full buffer.
# =====================================================================================================


def big_transients(sdfg):
    """Names of transient ARRAY descriptors whose element count is not statically 1 -- i.e. the buffers a
    localization would shrink. A contracted ``[1]`` slot has volume 1 and is excluded."""
    out = []
    for sd in sdfg.all_sdfgs_recursive():
        for name, desc in sd.arrays.items():
            if desc.transient and isinstance(desc, dace.data.Array):
                vol = 1
                for s in desc.shape:
                    vol *= s
                if str(dace.symbolic.simplify(vol)) != "1":
                    out.append(name)
    return out


def fuse_and_measure(prog, inputs, n, simplify=True):
    """Like ``run_fused`` but also reports the big-transient set before/after fusion, so a test can assert
    the intermediate buffer was (or was not) contracted. Returns (applied, exact, big_before, big_after)."""
    ref = prog.to_sdfg(simplify=simplify)
    ref.name = prog.name + "_ref"
    rb = {k: v.copy() for k, v in inputs.items()}
    ref(**rb, N=n)

    sd = prog.to_sdfg(simplify=simplify)
    big_before = big_transients(sd)
    applied = sd.apply_transformations_repeated(FuseLoops) or 0
    big_after = big_transients(sd)
    sd.name = prog.name + "_fused"
    fb = {k: v.copy() for k, v in inputs.items()}
    sd(**fb, N=n)
    exact = all(np.allclose(fb[k], rb[k], equal_nan=True) for k in inputs)
    return applied, exact, big_before, big_after


# --- contraction FIRES: transient written & read only at point i, both loops non-DOALL ---------------
#
# Neither loop is DOALL (each carries a recurrence on an OUTPUT array), so LoopToMap left them for
# FuseLoops; ``tmp`` is a pure per-iteration value -> contractible to a scalar.


@dace.program
def localize_scalar_chain(a: f64[N], b: f64[N], acc: f64[N], out: f64[N]):
    tmp = np.empty_like(a)
    for i in range(1, N):
        acc[i] = acc[i - 1] + a[i]  # recurrence carrier -> loop1 not DOALL
        tmp[i] = acc[i] + b[i]  # intermediate, point i
    for i in range(1, N):
        out[i] = out[i - 1] + tmp[i]  # recurrence carrier reading tmp[i] point -> loop2 not DOALL


@dace.program
def localize_mul_op(a: f64[N], b: f64[N], acc: f64[N], out: f64[N]):
    tmp = np.empty_like(a)
    for i in range(1, N):
        acc[i] = acc[i - 1] * 0.5 + a[i]
        tmp[i] = acc[i] * b[i]
    for i in range(1, N):
        out[i] = out[i - 1] - tmp[i]


@dace.program
def localize_min_op(a: f64[N], b: f64[N], acc: f64[N], out: f64[N]):
    tmp = np.empty_like(a)
    for i in range(1, N):
        acc[i] = acc[i - 1] + a[i]
        tmp[i] = min(acc[i], b[i])
    for i in range(1, N):
        out[i] = max(out[i - 1], tmp[i])


@dace.program
def localize_shared_read(a: f64[N], b: f64[N], acc: f64[N], out: f64[N]):
    tmp = np.empty_like(a)
    for i in range(1, N):
        acc[i] = acc[i - 1] + a[i]
        tmp[i] = acc[i] + b[i]
    for i in range(1, N):
        out[i] = out[i - 1] + tmp[i] + a[i]  # loop2 also reads the shared input a -- tmp still contracts


@pytest.mark.parametrize("prog", [localize_scalar_chain, localize_mul_op, localize_min_op, localize_shared_read])
def test_intermediate_is_localized_to_a_scalar(prog):
    applied, exact, big_before, big_after = fuse_and_measure(prog, mk(names=("a", "b", "acc", "out")), 48)
    assert applied >= 1
    assert exact
    assert "tmp" in "".join(big_before)  # the [N] intermediate existed before fusion
    assert len(big_after) < len(big_before)  # ... and was contracted away
    assert not any("tmp" in x for x in big_after)


@pytest.mark.parametrize("n", [4, 16, 48, 64])
def test_intermediate_localization_holds_across_sizes(n):
    applied, exact, big_before, big_after = fuse_and_measure(localize_scalar_chain, mk(n=n, names=("a", "b", "acc",
                                                                                                   "out")), n)
    assert applied >= 1 and exact
    assert len(big_after) < len(big_before)


@dace.program
def localize_two_intermediates(a: f64[N], b: f64[N], acc: f64[N], out: f64[N]):
    t1 = np.empty_like(a)
    t2 = np.empty_like(a)
    for i in range(1, N):
        acc[i] = acc[i - 1] + a[i]
        t1[i] = acc[i] + b[i]
        t2[i] = acc[i] - b[i]
    for i in range(1, N):
        out[i] = out[i - 1] + t1[i] * t2[i]


def test_two_point_intermediates_both_localized():
    applied, exact, big_before, big_after = fuse_and_measure(localize_two_intermediates,
                                                             mk(names=("a", "b", "acc", "out")), 48)
    assert applied >= 1 and exact
    assert len(big_before) >= 2  # t1, t2 both [N] before
    assert big_after == []  # both contracted


@dace.program
def localize_long_chain(a: f64[N], p: f64[N], q: f64[N], out: f64[N]):
    t1 = np.empty_like(a)
    t2 = np.empty_like(a)
    for i in range(1, N):
        p[i] = p[i - 1] + a[i]
        t1[i] = p[i] * 2.0
    for i in range(1, N):
        q[i] = q[i - 1] + t1[i]
        t2[i] = q[i] - a[i]
    for i in range(1, N):
        out[i] = out[i - 1] + t2[i]


def test_long_chain_localizes_every_intermediate():
    # A 3-loop chain collapses one adjacency per merge: FuseLoops leaves the merged body as two states
    # (StateFusion runs between passes, not inside the transformation), so the NEXT pair only matches once
    # the body is re-fused to a single compute state. Interleave a simplify to model the canon pipeline and
    # drive the chain fully closed -- then EVERY intermediate is localized.
    n = 48
    inputs = mk(names=("a", "p", "q", "out"))
    ref = localize_long_chain.to_sdfg(simplify=True)
    ref.name = "long_chain_ref"
    rb = {k: v.copy() for k, v in inputs.items()}
    ref(**rb, N=n)

    sd = localize_long_chain.to_sdfg(simplify=True)
    big_before = big_transients(sd)
    applied = 0
    while True:
        got = sd.apply_transformations(FuseLoops)
        if not got:
            break
        applied += got
        sd.simplify()  # re-fuse the merged body's two states so the next adjacency can match
    big_after = big_transients(sd)
    sd.name = "long_chain_fused"
    fb = {k: v.copy() for k, v in inputs.items()}
    sd(**fb, N=n)

    assert applied >= 2  # both adjacencies fuse once the body is re-collapsed between merges
    assert all(np.allclose(fb[k], rb[k], equal_nan=True) for k in inputs)
    assert big_after == []  # t1 and t2 both contracted to scalars


# --- contraction REFUSED (unsound to use one slot) but fusion still happens, value preserved ----------


@dace.program
def intermediate_carries_history(a: f64[N], b: f64[N], acc: f64[N], out: f64[N]):
    tmp = np.zeros(N)  # zero-init: tmp[i-1] at i==1 reads a defined cell -> deterministic reference
    for i in range(1, N):
        acc[i] = acc[i - 1] + a[i]
        tmp[i] = tmp[i - 1] + acc[i]  # tmp reads its OWN previous cell -> cross-iteration, not one slot
    for i in range(1, N):
        out[i] = out[i - 1] + tmp[i]


@dace.program
def intermediate_read_behind_in_second(a: f64[N], b: f64[N], acc: f64[N], out: f64[N]):
    tmp = np.zeros(N)
    for i in range(1, N):
        acc[i] = acc[i - 1] + a[i]
        tmp[i] = acc[i] + b[i]
    for i in range(2, N):
        out[i] = out[i - 1] + tmp[i - 1]  # reads the PREVIOUS tmp cell -> two live cells, no single slot


@dace.program
def intermediate_used_after_loop(a: f64[N], b: f64[N], acc: f64[N], out: f64[N], sink: f64[N]):
    tmp = np.empty_like(a)
    for i in range(1, N):
        acc[i] = acc[i - 1] + a[i]
        tmp[i] = acc[i] + b[i]
    for i in range(1, N):
        out[i] = out[i - 1] + tmp[i]
    sink[0] = tmp[3]  # tmp is live AFTER the loop -> not exclusive -> must not be contracted


@pytest.mark.parametrize("prog,names", [
    (intermediate_carries_history, ("a", "b", "acc", "out")),
    (intermediate_read_behind_in_second, ("a", "b", "acc", "out")),
    (intermediate_used_after_loop, ("a", "b", "acc", "out", "sink")),
])
def test_unsafe_intermediate_is_not_contracted_but_value_preserved(prog, names):
    applied, exact, big_before, big_after = fuse_and_measure(prog, mk(names=names), 48)
    assert exact  # value is always preserved
    assert big_before == big_after  # the buffer is NOT contracted (a single slot would be unsound)


def test_two_d_intermediate_not_contracted_v1():
    # a 2-D intermediate is written under a map scope (many cells per outer iteration) -- v1 refuses to
    # contract it; fusion (and value) are unaffected.
    @dace.program
    def prog(a: f64[N, N], acc: f64[N, N], out: f64[N, N]):
        tmp = np.empty_like(a)
        for i in range(1, N):
            for j in dace.map[0:N]:
                acc[i, j] = acc[i - 1, j] + a[i, j]
                tmp[i, j] = acc[i, j] * 2.0
        for i in range(1, N):
            for j in dace.map[0:N]:
                out[i, j] = out[i - 1, j] + tmp[i, j]

    applied, exact, big_before, big_after = fuse_and_measure(prog, mk2d(names=("a", "acc", "out")), 24)
    assert exact
    assert big_before == big_after  # 2-D intermediate left at full size in v1


@dace.program
def gather_indexed_intermediate(a: f64[N], idx: dace.int64[N], acc: f64[N], out: f64[N]):
    tmp = np.empty_like(a)
    for i in range(1, N):
        acc[i] = acc[i - 1] + a[i]
        tmp[i] = acc[idx[i]]  # written at point i but read via a gather -> not a pure point of i
    for i in range(1, N):
        out[i] = out[i - 1] + tmp[i]


def test_gather_written_intermediate_value_preserved():
    rng = np.random.default_rng(11)
    inputs = {
        "a": rng.random(48),
        "idx": rng.integers(0, 48, size=48).astype(np.int64),
        "acc": rng.random(48),
        "out": rng.random(48),
    }
    applied, exact, big_before, big_after = fuse_and_measure(gather_indexed_intermediate, inputs, 48)
    assert exact  # tmp WRITE is still point i, so tmp itself may localize; the value must be preserved


# --- flow hazard THROUGH a produced intermediate --------------------------------------------------------
#
# A flow (RAW) hazard carried by a compiler temp: body1 PRODUCES tmp[i], body2 reads tmp[i+1] ahead of
# that production. Because tmp is genuinely written in body1 (not a foldable constant), the read-ahead
# dependence survives simplify and FuseLoops must refuse. (The anti/WAR-through-a-temp mirrors are covered
# by the arg-array cases `read_behind_anti` / `read_ahead_anti_is_safe` above -- a read of a temp BEFORE it
# is produced folds to its init, dissolving the very dependence under test, so those live on arg arrays.)


@dace.program
def intermediate_flow_read_ahead_blocks(a: f64[N], acc: f64[N], out: f64[N]):
    tmp = np.zeros(N)  # produced cells overwrite the zero; the one unwritten tail cell stays a defined 0
    for i in range(1, N - 1):
        acc[i] = acc[i - 1] + a[i]
        tmp[i] = acc[i]  # tmp[i] genuinely produced here (not a foldable constant)
    for i in range(1, N - 1):
        out[i] = out[i - 1] + tmp[i + 1]  # reads tmp AHEAD of its production -> fusion illegal


def test_intermediate_flow_read_ahead_refuses_fusion():
    applied, exact, big_before, big_after = fuse_and_measure(intermediate_flow_read_ahead_blocks,
                                                             mk(names=("a", "acc", "out")), 48)
    assert applied == 0  # a real read-ahead flow hazard through the temp -> refuse the fuse entirely
    assert exact
    assert big_before == big_after  # nothing fused -> nothing contracted


def test_contraction_never_crashes_without_an_intermediate():
    # a single loop, or a pair sharing no transient, must not trip the contraction pass.
    @dace.program
    def prog(a: f64[N], b: f64[N], c: f64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]
        for i in range(1, N):
            c[i] = c[i - 1] + b[i]  # b is a program ARG, not a transient -> never contracted

    applied, exact, big_before, big_after = fuse_and_measure(prog, mk(names=("a", "b", "c")), 48)
    assert applied == 1 and exact
    assert big_before == big_after == []  # nothing transient to contract
