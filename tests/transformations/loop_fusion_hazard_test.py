# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop-form parity corpus for the MAP-fusion hazard corpus in `map_fusion_hazard_test.py`.

A pattern that is a genuine hazard between two Maps is equally a hazard between the two sequential
LoopRegions those Maps came from: if `FuseLoops` (the transformation the `LoopFusion` pass applies to a
fixpoint, `fuse_loops.py`) fuses something `MapFusionVertical` / `MapFusionHorizontal` refuses, that is
a miscompile. This file builds the LOOP-form equivalent of every pattern in `map_fusion_hazard_test.py`
and checks `FuseLoops` reaches the same verdict.

Construction: each Map-based fixture reuses the builders from `map_fusion_hazard_test.py` and is
converted with `MapToForLoop`, then `sdfg.simplify()` collapses `MapToForLoop`'s own
wrapping-NestedSDFG / empty-state scaffolding so the two loops end up as direct CFG siblings
(`simplify()` only runs InlineSDFGs / InlineControlFlowRegions / FuseStates / dead-code elimination --
none of it is LoopFusion or MapFusion, so it cannot itself change the fuse verdict).
`MapToForLoop.can_be_applied` refuses a Map with a surviving WCR output (a genuine parallel
reduction), so the two WCR shapes are hand-written as an explicit scalar read-modify-write loop -- the
loop-form a WCR reduction lowers to when it is NOT parallelizable, which is exactly what those two
shapes are testing.

Parity table (map verdict from `map_fusion_hazard_test.py`, loop verdict from `FuseLoops` here):

    pattern                                           map verdict   loop verdict   agree?
    copy-memlet hides the write                       REFUSE        REFUSE         yes
    view aliases the written array (WAR)              REFUSE        REFUSE         yes
    unknown boundary subset fails closed               REFUSE        REFUSE         yes
    InOut split / cross-state NestedSDFG read          REFUSE        REFUSE         yes
    WCR consumed by the second Map/loop                REFUSE        REFUSE         yes
    WCR independent, rides along                       FUSE          REFUSE         NO (benign)

No pattern was found where the loop form fuses something the Map form refuses (the dangerous
direction). The one divergence is `FuseLoops` being MORE conservative, not less -- see
`test_reduction_independent_of_the_second_loop_is_refused_by_a_doall_guard` for why, and the
forced-fuse check inside it for the numeric proof that fusing would in fact be safe.

Each test runs the UNFUSED loop-form SDFG as its own oracle, then fuses and runs again. Refusing to
fuse is always acceptable; changing the numbers never is.
"""
import numpy as np
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import dataflow as dftrans
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.interstate.fuse_loops import FuseLoops
from dace.transformation.passes.loop_fission import _single_compute_state

from .map_fusion_hazard_test import (
    N,
    _inner_write_then_read_sdfg,
    _make_inout_split_sdfg,
    _reader_map,
    _run_wcr,
    _wcr_shape,
    _writer_map,
)
from .map_fusion_vertical_test import unique_name

# ----- shared helpers -----------------------------------------------------------------------------


def map_entries(sdfg: dace.SDFG) -> list:
    """Every MapEntry anywhere in `sdfg`, in no particular order."""
    return [n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.MapEntry)]


def loop_regions(sdfg: dace.SDFG) -> list:
    """Every top-level LoopRegion in `sdfg`."""
    return [n for n in sdfg.nodes() if isinstance(n, LoopRegion)]


def convert_maps_to_loops(sdfg: dace.SDFG) -> dace.SDFG:
    """Turn every Map in `sdfg` into a LoopRegion, then simplify away `MapToForLoop`'s own scaffolding
    so adjacent loops end up as direct CFG siblings (what `FuseLoops`'s pattern requires)."""
    while map_entries(sdfg):
        MapToForLoop.apply_to(sdfg, map_entry=map_entries(sdfg)[0])
    sdfg.simplify()
    sdfg.validate()
    return sdfg


def run_ab(sdfg: dace.SDFG) -> dict:
    args = {"A": np.arange(N, dtype=np.float64) + 1.0, "B": np.zeros(N, dtype=np.float64)}
    sdfg(**args)
    return args


def assert_loop_fusion_preserves_meaning(build, what: str) -> int:
    """Fuse or refuse, but never change the numbers. Returns the number of pairs FuseLoops fused."""
    oracle = run_ab(build())
    fused = build()
    applied = fused.apply_transformations_repeated(FuseLoops, validate_all=True) or 0
    if applied:
        got = run_ab(fused)
        for name, expected in oracle.items():
            assert np.array_equal(got[name], expected), (f"fusing the loop-form changed {name} "
                                                         f"({what}): {got[name]} != {expected}")
    return applied


def force_fuse(sdfg: dace.SDFG, first: LoopRegion, second: LoopRegion) -> None:
    """Apply `FuseLoops` to `(first, second)` bypassing the DOALL-deference guard, the same way
    `allow_doall_fuse=True` does for `ReconstructWavefrontNest`. `apply_transformations_repeated` /
    `apply_to` cannot forward it (it is a plain keyword of `can_be_applied`, not a `properties.Property`
    -- `_can_be_applied_and_apply` only forwards entries that are in `cls.__properties__`), so this
    drives the transformation instance directly."""
    instance = FuseLoops()
    instance.setup_match(sdfg, sdfg.cfg_id, -1, {
        FuseLoops.first: sdfg.node_id(first),
        FuseLoops.second: sdfg.node_id(second)
    }, 0)
    assert instance.can_be_applied(sdfg, 0, sdfg, allow_doall_fuse=True), \
        "expected the dependence logic alone (DOALL guard aside) to judge this pair fusable"
    instance.apply(sdfg, sdfg)
    sdfg.validate()


# --- copy-memlet naming the inner container hides the write (map verdict: REFUSE) ----------------------


def build_copy_memlet_hides_write_map_form() -> dace.SDFG:
    """The `map_fusion_hazard_test.test_copy_memlet_naming_the_inner_container_hides_the_write` fixture,
    rebuilt here so this file can independently confirm the map-fusion verdict it compares against."""
    sdfg = dace.SDFG(unique_name("copy_memlet_hides_write_map"))
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    _, first_out = _writer_map(state, sdfg, "m1", "1", "A", "i - 1", 1.0, name_inner_container=True)
    second_entry, _ = _writer_map(state, sdfg, "m2", "2", "A", "i", 2.0)
    state.add_nedge(first_out, second_entry, dace.Memlet())
    sdfg.validate()
    return sdfg


def build_copy_memlet_hides_write_loop_form() -> dace.SDFG:
    """Loop-form of the pattern above: two sequential LoopRegions (instead of two Maps ordered by an
    empty Memlet) writing `A`. `A[i - 1]` is still written through a copy that names the inner scalar
    `buf1`, so a scanner keyed on `Memlet.data` alone would miss it. `FuseLoops._accesses` keys on the
    ACCESS NODE's own `.data` instead of the Memlet's, so this particular blind spot cannot recur once
    the Map scope (and its connectors) is gone -- there is no more "inner container name" to hide behind.
    """
    sdfg = dace.SDFG(unique_name("copy_memlet_hides_write_loop"))
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    state1 = sdfg.add_state("s1", is_start_block=True)
    _writer_map(state1, sdfg, "m1", "1", "A", "i - 1", 1.0, name_inner_container=True)
    state2 = sdfg.add_state_after(state1, "s2")
    _writer_map(state2, sdfg, "m2", "2", "A", "i", 2.0)
    sdfg.validate()
    return convert_maps_to_loops(sdfg)


def test_copy_memlet_naming_the_inner_container_hides_the_write_loop_form():
    map_applied = build_copy_memlet_hides_write_map_form().apply_transformations_repeated(dftrans.MapFusionHorizontal,
                                                                                          validate_all=True)
    assert map_applied == 0, "map-fusion baseline changed -- re-check this loop-form parity claim"

    loop_applied = assert_loop_fusion_preserves_meaning(build_copy_memlet_hides_write_loop_form,
                                                        "copy-Memlet naming the inner container")
    assert loop_applied == 0, "expected FuseLoops to refuse this WAW hazard, matching MapFusionHorizontal"


# --- a View of the written array is not a different array (map verdict: REFUSE) -----------------------


def build_view_aliases_array_map_form() -> dace.SDFG:
    """The `map_fusion_hazard_test.test_view_of_the_same_array_is_not_a_different_array` fixture."""
    sdfg = dace.SDFG(unique_name("view_aliases_array_map"))
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_view("V", [N - 1], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    source = state.add_access("A")
    view = state.add_access("V")
    state.add_edge(source, None, view, "views", dace.Memlet(data="A", subset=f"0:{N - 1}"))
    _, first_out = _reader_map(state, sdfg, "m1", "1", view, "V", "i - 1", "B", "i")
    second_entry, _ = _writer_map(state, sdfg, "m2", "2", "A", "i", 99.0)
    state.add_nedge(first_out, second_entry, dace.Memlet())
    sdfg.validate()
    return sdfg


def build_view_aliases_array_loop_form() -> dace.SDFG:
    """Loop-form: a WAR through a View -- loop1 reads `V[i - 1]` (a View of `A`) into `B[i]`, loop2
    writes `A[i]`. Reads/writes are keyed by the real AccessNode's `.data`, and a View resolves to what
    it views before `FuseLoops` ever looks at it (`MapToForLoop._copy_boundary_views` keeps the View's
    real edge across the scope boundary instead of leaving a dangling View descriptor), so the alias is
    visible to `_accesses` the same way it is visible to the (already-fixed) map-fusion scanner.
    """
    sdfg = dace.SDFG(unique_name("view_aliases_array_loop"))
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_view("V", [N - 1], dace.float64)
    state1 = sdfg.add_state("s1", is_start_block=True)
    source = state1.add_access("A")
    view = state1.add_access("V")
    state1.add_edge(source, None, view, "views", dace.Memlet(data="A", subset=f"0:{N - 1}"))
    _reader_map(state1, sdfg, "m1", "1", view, "V", "i - 1", "B", "i")
    state2 = sdfg.add_state_after(state1, "s2")
    _writer_map(state2, sdfg, "m2", "2", "A", "i", 99.0)
    sdfg.validate()
    return convert_maps_to_loops(sdfg)


def test_view_of_the_same_array_is_not_a_different_array_loop_form():
    map_applied = build_view_aliases_array_map_form().apply_transformations_repeated(dftrans.MapFusionHorizontal,
                                                                                     validate_all=True)
    assert map_applied == 0, "map-fusion baseline changed -- re-check this loop-form parity claim"

    loop_applied = assert_loop_fusion_preserves_meaning(build_view_aliases_array_loop_form,
                                                        "View aliasing the written array")
    assert loop_applied == 0, "expected FuseLoops to refuse this WAR-through-a-View hazard"


# --- an unknown boundary subset must not erase the access sets (map verdict: fails closed) --------------


def test_unknown_boundary_subset_must_not_erase_the_access_sets_loop_form():
    """Loop-form analogue of `test_an_unknown_boundary_subset_must_not_erase_the_access_sets`.

    `FuseLoops._accesses` does not special-case an unresolvable subset the way
    `map_fusion_helper._boundary_access` does (returning `None`) -- it goes through
    `Memlet.get_dst_subset`, whose `try_initialize` fills a blanked subset in with the FULL ARRAY range.
    That is still fail-CLOSED: a full-array range fails `_dep_class`'s single-point-access check, so
    `_fusion_legal` refuses exactly like the explicit-`None` path does. Different mechanism, same
    contract -- an access the scan cannot resolve must never look like "this loop touches nothing".
    """
    sdfg = dace.SDFG(unique_name("unknown_boundary_subset_loop"))
    sdfg.add_array("A", [N], dace.float64)
    state1 = sdfg.add_state("s1", is_start_block=True)
    _writer_map(state1, sdfg, "m1", "1", "A", "i - 1", 1.0)
    state2 = sdfg.add_state_after(state1, "s2")
    _writer_map(state2, sdfg, "m2", "2", "A", "i", 2.0)
    sdfg.validate()
    sdfg = convert_maps_to_loops(sdfg)

    loops = loop_regions(sdfg)
    first = next(loop for loop in loops if "m1" in loop.label)
    second = next(loop for loop in loops if "m2" in loop.label)
    body1 = _single_compute_state(first)

    # Blank the write-to-A edge's subset in body1, exactly as an uninitialised Memlet would present it.
    blanked = next(e for n in body1.nodes() if isinstance(n, nodes.AccessNode) and n.data == "A"
                   for e in body1.in_edges(n))
    blanked.data.subset = None

    reads, writes = FuseLoops._accesses(body1)
    assert writes.get("A"), "an unknown subset erased the write set instead of being reported as unknown"

    body2 = _single_compute_state(second)
    instance = FuseLoops()
    assert instance._fusion_legal(first, second, (reads, writes), FuseLoops._accesses(body2)) is False, \
        "an unresolvable write subset must refuse fusion (fail closed), not silently allow it"


# --- InOut split must not redirect a cross-state read (map verdict: REFUSE) -----------------------------


def build_inout_split_loop_form_unified_names() -> dace.SDFG:
    """Loop-form of `test_inout_split_must_not_redirect_a_cross_state_read`.

    `_make_inout_split_sdfg` names the two Maps' parameters `i` / `j`; a straight `MapToForLoop`
    conversion would carry that mismatch into the two LoopRegions' `loop_variable`, and
    `FuseLoops._fusion_legal` does not rename iterators before running the dependence classifier (only
    `_merge` does, and only AFTER legality is decided) -- an unrelated gap that would refuse this pair
    for the WRONG reason. Both Maps are rebuilt here using the SAME iterator name `i` so the test
    isolates the actual question: does `FuseLoops` reach the same verdict as `MapFusionVertical` on the
    InOut-split hazard itself.

    `MapFusionVertical`'s bug here is mechanism-specific: fusing SPLITS the NestedSDFG's `T` InOut
    connector, and the split misreads a cross-STATE read inside the NSDFG's OWN body (`T` written in
    one inner state, read back in the next) as an external read, redirecting it to the wrong source.
    `FuseLoops` has no analogous mechanism -- it never touches or splits a NestedSDFG's connectors, it
    just runs body1 then body2 per iteration -- so this hazard CLASS cannot arise in `FuseLoops` by
    construction, not by luck (see the forced-fuse check in the test below).
    """
    sdfg = dace.SDFG(unique_name("inout_split_loop_unified"))
    for name in ("T", "X", "B"):
        sdfg.add_array(name, [N], dace.float64)

    state1 = sdfg.add_state("s1", is_start_block=True)
    first_entry, first_exit = state1.add_map("m1", dict(i=f"0:{N}"))
    nsdfg = state1.add_nested_sdfg(_inner_write_then_read_sdfg(), {"T"}, {"T", "X"})
    first_entry.add_in_connector("IN_T")
    first_entry.add_out_connector("OUT_T")
    state1.add_edge(state1.add_access("T"), None, first_entry, "IN_T", dace.Memlet(data="T", subset=f"0:{N}"))
    state1.add_edge(first_entry, "OUT_T", nsdfg, "T", dace.Memlet(data="T", subset="i"))
    for name in ("T", "X"):
        first_exit.add_in_connector(f"IN_{name}")
        first_exit.add_out_connector(f"OUT_{name}")
        state1.add_edge(nsdfg, name, first_exit, f"IN_{name}", dace.Memlet(data=name, subset="i"))
    intermediate = state1.add_access("T")
    state1.add_edge(first_exit, "OUT_T", intermediate, None, dace.Memlet(data="T", subset=f"0:{N}"))
    state1.add_edge(first_exit, "OUT_X", state1.add_access("X"), None, dace.Memlet(data="X", subset=f"0:{N}"))

    state2 = sdfg.add_state_after(state1, "s2")
    second_entry, second_exit = state2.add_map("m2", dict(i=f"0:{N}"))  # same name as m1, on purpose
    second_entry.add_in_connector("IN_T")
    second_entry.add_out_connector("OUT_T")
    state2.add_edge(state2.add_access("T"), None, second_entry, "IN_T", dace.Memlet(data="T", subset=f"0:{N}"))
    consumer = state2.add_tasklet("cons", {"a"}, {"o"}, "o = a")
    state2.add_edge(second_entry, "OUT_T", consumer, "a", dace.Memlet(data="T", subset="i"))
    second_exit.add_in_connector("IN_B")
    second_exit.add_out_connector("OUT_B")
    state2.add_edge(consumer, "o", second_exit, "IN_B", dace.Memlet(data="B", subset="i"))
    state2.add_edge(second_exit, "OUT_B", state2.add_access("B"), None, dace.Memlet(data="B", subset=f"0:{N}"))
    sdfg.validate()
    return convert_maps_to_loops(sdfg)


def run_txb(sdfg: dace.SDFG) -> dict:
    args = {"T": np.full(N, 1000.0), "X": np.zeros(N), "B": np.zeros(N)}
    sdfg(**args)
    return args


def test_inout_split_must_not_redirect_a_cross_state_read_loop_form():
    """Map verdict: REFUSE (`MapFusionVertical`). Loop verdict: REFUSE, but for an UNRELATED reason:
    the consumer loop is independently DOALL, and `FuseLoops` defers DOALL loops to `LoopToMap` by
    policy (`_is_doall`). Forcing past that policy guard shows the dependence logic ALONE also fuses
    this correctly -- the InOut-split hazard class genuinely does not exist for `FuseLoops`, it is not
    merely refused for the right reason by accident.
    """
    map_applied = _make_inout_split_sdfg().apply_transformations_repeated(dftrans.MapFusionVertical, validate_all=True)
    assert map_applied == 0, "map-fusion baseline changed -- re-check this loop-form parity claim"

    oracle = run_txb(_make_inout_split_sdfg())
    sdfg = build_inout_split_loop_form_unified_names()
    got = run_txb(sdfg)
    for name in ("T", "X", "B"):
        assert np.array_equal(got[name], oracle[name]), f"loop-form conversion alone changed {name}"

    applied = sdfg.apply_transformations_repeated(FuseLoops, validate_all=True) or 0
    assert applied == 0, "expected FuseLoops to refuse (its DOALL-deference guard), matching MapFusionVertical"

    forced = build_inout_split_loop_form_unified_names()
    loops = loop_regions(forced)
    first = next(loop for loop in loops
                 if forced.out_edges(loop) and isinstance(forced.out_edges(loop)[0].dst, LoopRegion))
    second = forced.out_edges(first)[0].dst
    force_fuse(forced, first, second)
    got_forced = run_txb(forced)
    for name in ("T", "X", "B"):
        assert np.array_equal(got_forced[name], oracle[name]), \
            f"forced fuse past the DOALL guard changed {name} -- would be the InOut-split miscompile class"


# --- WCR shapes: reduction consumed vs. independent -----------------------------------------------------


def build_wcr_loop_form(reduction_is_consumed: bool) -> dace.SDFG:
    """Loop-form of `_wcr_shape`. `MapToForLoop.can_be_applied` refuses a Map with a surviving WCR
    output (it would serialize a genuine parallel reduction), so the reduction loop is hand-written as
    an explicit scalar read-modify-write accumulation into `out` -- the loop-form a WCR reduction lowers
    to when it is NOT parallelizable, which is exactly what this shape is testing.
    """
    sdfg = dace.SDFG(unique_name(f"wcr_loop_form_{reduction_is_consumed}"))
    for name in ("A", "B", "D"):
        sdfg.add_array(name, [N], dace.float64)
    sdfg.add_array("C", [N], dace.float64, transient=True)
    sdfg.add_array("out", [1], dace.float64)

    loop1 = LoopRegion("reduce_loop", f"i < {N}", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop1, is_start_block=True)
    body1 = loop1.add_state("body1", is_start_block=True)
    reduce_tasklet = body1.add_tasklet("body1_t", {"a", "b"}, {"wcr_o", "c_o"}, "wcr_o = a * b\nc_o = a + b")
    body1.add_edge(body1.add_access("A"), None, reduce_tasklet, "a", dace.Memlet("A[i]"))
    body1.add_edge(body1.add_access("B"), None, reduce_tasklet, "b", dace.Memlet("B[i]"))
    body1.add_edge(reduce_tasklet, "c_o", body1.add_access("C"), None, dace.Memlet("C[i]"))
    accum = body1.add_tasklet("accum", {"cur", "delta"}, {"nxt"}, "nxt = cur + delta")
    body1.add_edge(body1.add_access("out"), None, accum, "cur", dace.Memlet("out[0]"))
    body1.add_edge(reduce_tasklet, "wcr_o", accum, "delta", dace.Memlet())
    body1.add_edge(accum, "nxt", body1.add_access("out"), None, dace.Memlet("out[0]"))

    scale_state = sdfg.add_state("scale")
    scale_tasklet = scale_state.add_tasklet("scale_t", {"iv"}, {"ov"}, "ov = iv * 2.0")
    scale_state.add_edge(scale_state.add_access("out"), None, scale_tasklet, "iv", dace.Memlet("out[0]"))
    scale_state.add_edge(scale_tasklet, "ov", scale_state.add_access("out"), None, dace.Memlet("out[0]"))

    loop2 = LoopRegion("consume_loop", f"i < {N}", "i", "i = 0", "i = i + 1")
    body2 = loop2.add_state("body2", is_start_block=True)
    if reduction_is_consumed:
        consume_tasklet = body2.add_tasklet("body2_t", {"ov2"}, {"d"}, "d = ov2")
        body2.add_edge(body2.add_access("out"), None, consume_tasklet, "ov2", dace.Memlet("out[0]"))
        body2.add_edge(consume_tasklet, "d", body2.add_access("D"), None, dace.Memlet("D[i]"))
        sdfg.add_node(loop2)
        # loop2 needs the FULLY-SCALED value -- the scale tasklet is a genuine barrier between the
        # reduce and its consumer, so loop2 can never be a direct CFG sibling of loop1.
        sdfg.add_edge(loop1, scale_state, dace.InterstateEdge())
        sdfg.add_edge(scale_state, loop2, dace.InterstateEdge())
    else:
        consume_tasklet = body2.add_tasklet("body2_t", {"cv", "av", "bv"}, {"d"}, "d = cv + av + bv")
        body2.add_edge(body2.add_access("C"), None, consume_tasklet, "cv", dace.Memlet("C[i]"))
        body2.add_edge(body2.add_access("A"), None, consume_tasklet, "av", dace.Memlet("A[i]"))
        body2.add_edge(body2.add_access("B"), None, consume_tasklet, "bv", dace.Memlet("B[i]"))
        body2.add_edge(consume_tasklet, "d", body2.add_access("D"), None, dace.Memlet("D[i]"))
        # out is untouched by loop2 in this variant, so nothing forces the scale between the two loops
        # -- put it after both, making loop1/loop2 direct CFG siblings (the fair loop-form of "rides
        # along independently").
        sdfg.add_node(loop2)
        sdfg.add_edge(loop1, loop2, dace.InterstateEdge())
        sdfg.add_edge(loop2, scale_state, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def test_reduction_consumed_by_the_second_loop_is_not_fused():
    """Map verdict: REFUSE (neither `MapFusionVertical` nor `MapFusionHorizontal` fuse this pair --
    `MapFusionVertical` matches only `MapExit -> (AccessNode) -> MapEntry`, and the scale tasklet
    between the reduce and its consumer breaks that direct adjacency, so the pair is never even a
    fusion candidate). Loop verdict: REFUSE, for the SAME structural reason: `FuseLoops` only matches
    `LoopRegion -> LoopRegion` (`loop1 -> scale state -> loop2` here), so it never considers this pair
    either. The barrier is inherent to the computation (loop2 needs the fully-reduced, scaled value),
    not an artifact of either representation -- AGREE.
    """
    map_applied = _wcr_shape(reduction_is_consumed=True).apply_transformations_repeated(dftrans.MapFusionVertical,
                                                                                        validate_all=True)
    assert map_applied == 0, "map-fusion baseline changed -- re-check this loop-form parity claim"

    oracle = _run_wcr(build_wcr_loop_form(reduction_is_consumed=True))
    sdfg = build_wcr_loop_form(reduction_is_consumed=True)
    applied = sdfg.apply_transformations_repeated(FuseLoops, validate_all=True) or 0
    assert applied == 0, "expected no adjacent LoopRegion pair to match -- the scale state is a genuine barrier"
    got = _run_wcr(sdfg)
    for name, expected in oracle.items():
        assert np.array_equal(got[name], expected), f"unfused loop-form changed {name}"


def test_reduction_independent_of_the_second_loop_is_refused_by_a_doall_guard():
    """Map verdict: FUSE (`MapFusionVertical` fuses the reduction Map with the elementwise consumer;
    the reduction rides along untouched). Loop verdict: REFUSE -- the ONE divergence in this corpus,
    and the BENIGN direction (a missed optimization, not a miscompile).

    `FuseLoops._fusion_legal` independently computes `True` for this pair (`C` is a clean
    per-iteration producer/consumer, `out` is untouched by the second loop) -- the refusal is entirely
    the `_is_doall` guard: `consume_loop`, read alone, is exactly the kind of embarrassingly-parallel
    loop `LoopToMap` would turn into a Map, and `FuseLoops`'s policy is to never serialize a loop that
    could parallelize on its own, deferring to `LoopToMap` instead. In a canon pipeline that runs
    `LoopToMap` before `LoopFusion` (the documented intended order), `consume_loop` would already be a
    Map by the time `FuseLoops` runs, and this exact LoopRegion pair would not arise --
    `MapFusionVertical` would handle the fusion instead, matching `_wcr_shape`'s own structure. The gap
    only bites a caller that runs `FuseLoops` out of that order.

    The forced-fuse check (bypassing the guard the same way `allow_doall_fuse=True` does for
    `ReconstructWavefrontNest`) confirms `_fusion_legal`'s claim: fusing is value-preserving here, it is
    simply left undone by policy.
    """
    map_applied = _wcr_shape(reduction_is_consumed=False).apply_transformations_repeated(dftrans.MapFusionVertical,
                                                                                         validate_all=True)
    assert map_applied == 1, "map-fusion baseline changed -- re-check this loop-form parity claim"

    oracle = _run_wcr(build_wcr_loop_form(reduction_is_consumed=False))

    sdfg = build_wcr_loop_form(reduction_is_consumed=False)
    applied = sdfg.apply_transformations_repeated(FuseLoops, validate_all=True) or 0
    assert applied == 0, "current FuseLoops policy refuses this pair -- documents the divergence from MapFusion"

    forced = build_wcr_loop_form(reduction_is_consumed=False)
    loops = loop_regions(forced)
    first = next(loop for loop in loops if loop.label == "reduce_loop")
    second = next(loop for loop in loops if loop.label == "consume_loop")
    force_fuse(forced, first, second)
    got = _run_wcr(forced)
    for name, expected in oracle.items():
        assert np.array_equal(got[name], expected), f"forced fuse changed {name} -- would contradict _fusion_legal"


# --- shape-gate relaxation: a ConditionalBlock / multi-state body is judged on legality, not shape -----
#
# `FuseLoops.can_be_applied` used to gate on `_single_compute_state` (`loop_fission.py`): any body that
# was not "one plain SDFGState, optionally with empty companion states" was refused outright, before
# `_fusion_legal` ever ran. Two real corpus shapes never even reached the legality kernel because of this:
# npbench `mandelbrot2`'s per-pixel masked update body (`[empty bridge state, ConditionalBlock]`) and
# polybench `ludcmp`'s multi-state forward/back-substitution bodies. The two tests below are the minimal
# synthetic form of each shape: a masked running-count "scan" (`ConditionalBlock` + index-defining bridge,
# mirroring mandelbrot2's own shape) that has no cross-loop dependence and must now be ACCEPTED, and a
# multi-state body pair with a genuine read-ahead flow hazard (mirroring ludcmp's back-substitution) that
# must stay REFUSED -- proving the relaxation only widened what shape `_fusion_legal` gets to judge, not
# what it accepts.


def _masked_running_count_loop(sdfg: dace.SDFG, label: str, mask_name: str, acc_name: str, out_name: str) -> LoopRegion:
    """A loop whose body is `[empty bridge state, ConditionalBlock]`: the bridge defines `mask_name` from
    `M[i]` (an interstate-edge assignment, exactly mandelbrot2's own shape), and the `ConditionalBlock`'s
    branch condition reads that symbol. The true branch increments a private per-loop accumulator
    `acc_name` and copies it to `out_name[i]`; the false branch copies `acc_name` through unchanged -- a
    running count of how many `M[i]` were nonzero so far, a genuine sequential scan (`LoopToMap` cannot
    parallelize it: `acc_name` carries a self-dependence across iterations), so acceptance below exercises
    the shape-gate relaxation on its own, with no help from the DOALL-deference guard.
    """
    loop = LoopRegion(label, f"i < {N}", "i", "i = 0", "i = i + 1")
    bridge = loop.add_state(f"{label}_bridge", is_start_block=True)
    cond = ConditionalBlock(f"{label}_cond", sdfg, loop)
    loop.add_node(cond)

    branch_true = ControlFlowRegion(f"{label}_true", sdfg)
    st_true = branch_true.add_state(f"{label}_true_s", is_start_block=True)
    inc = st_true.add_tasklet(f"{label}_inc", {"cur"}, {"nxt"}, "nxt = cur + 1.0")
    st_true.add_edge(st_true.add_access(acc_name), None, inc, "cur", dace.Memlet(f"{acc_name}[0]"))
    st_true.add_edge(inc, "nxt", st_true.add_access(acc_name), None, dace.Memlet(f"{acc_name}[0]"))
    copy_true = st_true.add_tasklet(f"{label}_copy_true", {"a"}, {"o"}, "o = a")
    st_true.add_edge(st_true.add_access(acc_name), None, copy_true, "a", dace.Memlet(f"{acc_name}[0]"))
    st_true.add_edge(copy_true, "o", st_true.add_access(out_name), None, dace.Memlet(f"{out_name}[i]"))
    cond.add_branch(CodeBlock(f"{mask_name} != 0"), branch_true)

    branch_false = ControlFlowRegion(f"{label}_false", sdfg)
    st_false = branch_false.add_state(f"{label}_false_s", is_start_block=True)
    copy_false = st_false.add_tasklet(f"{label}_copy_false", {"a"}, {"o"}, "o = a")
    st_false.add_edge(st_false.add_access(acc_name), None, copy_false, "a", dace.Memlet(f"{acc_name}[0]"))
    st_false.add_edge(copy_false, "o", st_false.add_access(out_name), None, dace.Memlet(f"{out_name}[i]"))
    cond.add_branch(None, branch_false)

    loop.add_edge(bridge, cond, dace.InterstateEdge(assignments={mask_name: "M[i]"}))
    return loop


def build_conditional_running_count_pair() -> dace.SDFG:
    """Two adjacent same-range loops, each `_masked_running_count_loop`, writing INDEPENDENT outputs `B`
    / `C` from the SAME mask `M` -- no array is touched by both bodies, so `_fusion_legal` has nothing to
    refuse once it is reached. `_single_compute_state` (the OLD gate) returns `None` for both bodies (a
    `ConditionalBlock` is not an `SDFGState`), so this fixture is the direct witness of the shape relaxation.
    """
    sdfg = dace.SDFG(unique_name("conditional_running_count_pair"))
    sdfg.add_array("M", [N], dace.int32)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_array("C", [N], dace.float64)
    sdfg.add_array("acc1", [1], dace.float64, transient=True)
    sdfg.add_array("acc2", [1], dace.float64, transient=True)

    loop1 = _masked_running_count_loop(sdfg, "loop1", "mask1", "acc1", "B")
    loop2 = _masked_running_count_loop(sdfg, "loop2", "mask2", "acc2", "C")

    init = sdfg.add_state("init", is_start_block=True)
    zero1 = init.add_tasklet("zero1", {}, {"o"}, "o = 0.0")
    init.add_edge(zero1, "o", init.add_access("acc1"), None, dace.Memlet("acc1[0]"))
    zero2 = init.add_tasklet("zero2", {}, {"o"}, "o = 0.0")
    init.add_edge(zero2, "o", init.add_access("acc2"), None, dace.Memlet("acc2[0]"))

    sdfg.add_node(loop1)
    sdfg.add_node(loop2)
    end = sdfg.add_state("end")
    sdfg.add_edge(init, loop1, dace.InterstateEdge())
    sdfg.add_edge(loop1, loop2, dace.InterstateEdge())
    sdfg.add_edge(loop2, end, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def run_mc(sdfg: dace.SDFG) -> dict:
    rng = np.random.default_rng(3)
    args = {
        "M": (rng.random(N) > 0.5).astype(np.int32),
        "B": np.zeros(N, dtype=np.float64),
        "C": np.zeros(N, dtype=np.float64),
    }
    sdfg(**args)
    return args


def test_conditional_block_body_is_accepted_once_the_shape_gate_is_relaxed():
    """The mandelbrot2-shaped body (`ConditionalBlock` + index-defining bridge) must no longer be refused
    just for not being a plain `SDFGState` -- with no cross-loop dependence, fusion is now ACCEPTED.
    First proves the OLD gate (`_single_compute_state`) would have refused this shape outright, then
    proves the current code accepts and fuses it without changing the numbers.
    """
    sdfg = build_conditional_running_count_pair()
    first = next(loop for loop in loop_regions(sdfg) if loop.label == "loop1")
    second = next(loop for loop in loop_regions(sdfg) if loop.label == "loop2")
    assert _single_compute_state(first) is None and _single_compute_state(second) is None, \
        "fixture must actually exercise the OLD gate's blind spot (a ConditionalBlock body)"

    oracle = run_mc(build_conditional_running_count_pair())
    fused = build_conditional_running_count_pair()
    applied = fused.apply_transformations_repeated(FuseLoops, validate_all=True) or 0
    assert applied >= 1, "expected the relaxed shape gate to let this ConditionalBlock-bodied pair fuse"
    got = run_mc(fused)
    for name in ("B", "C"):
        assert np.array_equal(got[name], oracle[name]), f"fusing the ConditionalBlock bodies changed {name}"


# --- multi-state body pair with a genuine read-ahead flow hazard: relaxation must stay sound ------------


def build_read_ahead_multi_state_pair() -> dace.SDFG:
    """Loop-form of `ludcmp`'s back-substitution hazard: `loop1`'s TWO-state body (`_linear_blocks`'s
    plain multi-block chain, unreachable through the OLD single-compute-state gate) writes `y[i]`; `loop2`
    reads `y[i + 1]` -- a read-ahead of a value `loop1` has not produced yet at iteration `i`. Fusing would
    make the fused loop read `y[i + 1]` before `loop1`'s own iteration `i + 1` has run, changing the value.
    """
    sdfg = dace.SDFG(unique_name("read_ahead_multi_state_pair"))
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("y", [N], dace.float64)
    sdfg.add_array("z", [N], dace.float64)
    sdfg.add_array("tmp", [1], dace.float64, transient=True)

    bound = N - 1  # i < bound -> max i = N - 2, so y[i + 1] stays in bounds
    loop1 = LoopRegion("loop1", f"i < {bound}", "i", "i = 0", "i = i + 1")
    sa = loop1.add_state("sa", is_start_block=True)
    t_a = sa.add_tasklet("t_a", {"a"}, {"o"}, "o = a * 2.0")
    sa.add_edge(sa.add_access("A"), None, t_a, "a", dace.Memlet("A[i]"))
    sa.add_edge(t_a, "o", sa.add_access("tmp"), None, dace.Memlet("tmp[0]"))
    sb = loop1.add_state_after(sa, "sb")
    t_b = sb.add_tasklet("t_b", {"a"}, {"o"}, "o = a + 1.0")
    sb.add_edge(sb.add_access("tmp"), None, t_b, "a", dace.Memlet("tmp[0]"))
    sb.add_edge(t_b, "o", sb.add_access("y"), None, dace.Memlet("y[i]"))

    loop2 = LoopRegion("loop2", f"i < {bound}", "i", "i = 0", "i = i + 1")
    s2 = loop2.add_state("s2", is_start_block=True)
    t2 = s2.add_tasklet("t2", {"a"}, {"o"}, "o = a * 3.0")
    s2.add_edge(s2.add_access("y"), None, t2, "a", dace.Memlet("y[i + 1]"))  # read-ahead
    s2.add_edge(t2, "o", s2.add_access("z"), None, dace.Memlet("z[i]"))

    start = sdfg.add_state("start", is_start_block=True)
    sdfg.add_node(loop1)
    sdfg.add_node(loop2)
    end = sdfg.add_state("end")
    sdfg.add_edge(start, loop1, dace.InterstateEdge())
    sdfg.add_edge(loop1, loop2, dace.InterstateEdge())
    sdfg.add_edge(loop2, end, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def run_yz(sdfg: dace.SDFG) -> dict:
    rng = np.random.default_rng(11)
    args = {"A": rng.random(N), "y": np.zeros(N), "z": np.zeros(N)}
    sdfg(**args)
    return args


def test_read_ahead_flow_hazard_across_multi_state_bodies_still_refused():
    """The shape gate now lets a multi-state body (`_linear_blocks`) reach `_fusion_legal` -- this test is
    the guard that the relaxation did not also relax legality: a genuine read-ahead flow hazard on `y`
    must still refuse, on the module docstring's own flow rule (`read in body2 ahead of its production in
    body1`), which `BreakAntiDependence._dep_class` reports as ``'WAR'`` for the constant +1 offset here.
    """
    sdfg = build_read_ahead_multi_state_pair()
    first = next(loop for loop in loop_regions(sdfg) if loop.label == "loop1")
    second = next(loop for loop in loop_regions(sdfg) if loop.label == "loop2")
    acc1 = FuseLoops._body_accesses(first)
    acc2 = FuseLoops._body_accesses(second)
    assert acc1 is not None and acc2 is not None, "the multi-state body must now reach the legality kernel"
    assert FuseLoops()._fusion_legal(first, second, acc1, acc2) is False, \
        "a read-ahead flow hazard on y must refuse fusion, not be silently accepted"

    oracle = run_yz(build_read_ahead_multi_state_pair())
    sdfg2 = build_read_ahead_multi_state_pair()
    applied = sdfg2.apply_transformations_repeated(FuseLoops, validate_all=True) or 0
    assert applied == 0, "expected the read-ahead hazard on y to refuse fusion"
    got = run_yz(sdfg2)
    for name in ("y", "z"):
        assert np.array_equal(got[name], oracle[name]), f"unfused loop-form changed {name}"


if __name__ == "__main__":
    pytest.main([__file__])
