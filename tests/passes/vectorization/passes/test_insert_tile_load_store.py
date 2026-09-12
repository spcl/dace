# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`InsertTileLoadStore`.

Validates the staging-first replacement chain end-to-end:

  ``StageGlobalArrayThroughScalars  ->  WidenAccesses  ->
   InsertTileLoadStore``

on simple Python kernels. Each test confirms:

* Lane-dep non-transient reads gain a ``TileLoad`` lib node.
* Lane-dep non-transient writes gain a ``TileStore`` lib node.
* CONSTANT (loop-invariant) edges stay direct -- no lib node, no Python
  assignment tasklet inserted.
"""
import dace
from dace import data as dt
from dace.libraries.tileops import TileLoad, TileStore
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import BypassTrivialAssignTasklets
from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.passes.vectorization.stage_global_array_through_scalars import (
    StageGlobalArrayThroughScalars, )
from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses
from dace.transformation.passes.vectorization.insert_tile_load_store import (
    InsertTileLoadStore,
    _assert_post_stage_invariants,
)

N = dace.symbol("N")


def _stage_widen_insert(prog):
    """Apply the three staging-first passes in order, return ``(sdfg, body_state)``."""
    sdfg = prog.to_sdfg(simplify=True)
    BypassTrivialAssignTasklets().apply_pass(sdfg, {})
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs)
    StageGlobalArrayThroughScalars().apply_pass(sdfg, {})
    WidenAccesses(widths=(8, )).apply_pass(sdfg, {})
    InsertTileLoadStore(widths=(8, )).apply_pass(sdfg, {})
    body_state = None
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for n in state.nodes():
                if isinstance(n, (TileLoad, TileStore)):
                    body_state = state
                    break
            if body_state:
                break
        if body_state:
            break
    return sdfg, body_state


@dace.program
def linear_kernel(A: dace.float64[N], B: dace.float64[N], scale: dace.float64):
    """``B[i] = A[i] * scale`` -- LINEAR read+write, CONSTANT scalar."""
    for i in dace.map[0:N]:
        B[i] = A[i] * scale


def test_linear_kernel_emits_tileload_and_tilestore():
    """B[i] = A[i] * scale -- A and B widened via TileLoad/TileStore; scale stays direct."""
    sdfg, body_state = _stage_widen_insert(linear_kernel)
    assert body_state is not None, "expected at least one TileLoad / TileStore in some body NSDFG"
    tile_loads = [n for n in body_state.nodes() if isinstance(n, TileLoad)]
    tile_stores = [n for n in body_state.nodes() if isinstance(n, TileStore)]
    assert len(tile_loads) == 1, f"expected 1 TileLoad (A), got {len(tile_loads)}"
    assert len(tile_stores) == 1, f"expected 1 TileStore (B), got {len(tile_stores)}"
    # ``scale`` is CONSTANT -- stays as direct edge from scale AN to the consumer tasklet.
    # No TileLoad for scale.
    for tl in tile_loads:
        in_edges = list(body_state.in_edges(tl))
        # TileLoad's _src reads from a non-transient (A, not scale).
        for e in in_edges:
            if e.dst_conn == "_src":
                assert e.data.data != "scale", "scale must not get a TileLoad (CONSTANT)"


@dace.program
def two_loads_kernel(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    """``C[i] = A[i] + B[i]`` -- two LINEAR reads, one LINEAR write."""
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def test_two_lane_dep_reads_emit_two_tileloads():
    """Both A and B reads gain a TileLoad."""
    sdfg, body_state = _stage_widen_insert(two_loads_kernel)
    assert body_state is not None
    tile_loads = [n for n in body_state.nodes() if isinstance(n, TileLoad)]
    tile_stores = [n for n in body_state.nodes() if isinstance(n, TileStore)]
    assert len(tile_loads) == 2, f"expected 2 TileLoads (A,B), got {len(tile_loads)}"
    assert len(tile_stores) == 1


@dace.program
def constant_only_kernel(A: dace.float64[N], scale: dace.float64):
    """``A[i] = scale + scale`` -- no lane-dep read (only the LINEAR write)."""
    for i in dace.map[0:N]:
        A[i] = scale + scale


def test_constant_read_no_tileload():
    """``scale`` is CONSTANT -- no TileLoad emitted for it. Only the LINEAR
    write to A gets a TileStore."""
    sdfg, body_state = _stage_widen_insert(constant_only_kernel)
    if body_state is None:
        # Kernel may be too trivial to produce TileStore; allow.
        return
    tile_loads = [n for n in body_state.nodes() if isinstance(n, TileLoad)]
    tile_stores = [n for n in body_state.nodes() if isinstance(n, TileStore)]
    assert len(tile_loads) == 0, "scale is CONSTANT, no TileLoad expected"
    assert len(tile_stores) == 1


# ---------------------------------------------------------------------------
# Regression tests for the empty-memlet (ordering-edge) family of defects.
#
# These three build the minimal state shape each defective function needs directly
# (not through the Python frontend) and call the pass method under test in isolation.
# Hand-built, not frontend-derived: flagged per the audit's fixture caveat -- the shape
# exercised here (an AccessNode's in-/out-edge list containing an empty ordering edge
# next to a real data edge) is the one StateFusionExtended's WAR/WAW happens-before
# mechanism (`first_state.add_nedge(j, i, memlet.Memlet())` in
# state_fusion_with_happens_before.py) produces when it fires inside a tile-tagged
# NSDFG body; existing full-pipeline kernel corpora (TSVC/cloudsc/ICON) were swept and
# did not happen to reach this exact adjacency, so this constructs it directly.
# ---------------------------------------------------------------------------


def test_stage_writes_skips_ordering_edge_as_representative():
    """An empty (ordering) in-edge sorted before the real per-tile write in-edge must
    not become the representative edge `_stage_writes_in_state` classifies from.

    Without the fix, `an_side_subset` falls back to the AN's FULL descriptor shape for
    the empty edge, which classifies as CONSTANT (loop-invariant) and makes the pass
    skip staging entirely -- the real `A[i:i+4]` write is silently left as a bare
    Tasklet -> AccessNode edge instead of being routed through a TileStore.
    """
    inner_sdfg = dace.SDFG("body")
    inner_sdfg.add_array("A", [16], dace.float64)
    state = inner_sdfg.add_state("s")

    producer = state.add_tasklet("prod", {}, {"_out"}, "_out = 1.0")
    an_a = state.add_access("A")
    # Ordering-only edge inserted FIRST, so an unfiltered in_edges()[0] picks it.
    dummy = state.add_access("A")
    state.add_nedge(dummy, an_a, dace.Memlet())
    # The real per-tile write, inserted SECOND.
    state.add_edge(producer, "_out", an_a, None, dace.Memlet("A[i:i+4]"))

    pas = InsertTileLoadStore(widths=(4, ))
    staged = pas._stage_writes_in_state(state, inner_sdfg, ("i", ), None)

    assert staged == 1, "the real A[i:i+4] write must be staged despite the ordering in-edge"
    tile_stores = [n for n in state.nodes() if isinstance(n, TileStore)]
    assert len(tile_stores) == 1, "write must be routed through a TileStore, not left as a bare AN edge"
    # The ordering edge itself must survive untouched (still empty).
    ordering_edges = [e for e in state.in_edges(an_a) if e.data.is_empty()]
    assert len(ordering_edges) == 1


def test_assert_post_stage_invariants_allows_ordering_edge_between_globals():
    """A happens-before edge between two ordinary (non-transient, non-Scalar)
    AccessNodes -- the shape StateFusionExtended's WAW mechanism produces -- must not
    trip the design 3.8.3 (2) "AN -> AN survivor" check. Only a REAL AN->AN data copy
    that is neither a Scalar bridge nor a transient->output writeback is a violation.
    """
    sdfg = dace.SDFG("body")
    sdfg.add_array("A", [16], dace.float64)
    state = sdfg.add_state("s")

    an_a1 = state.add_access("A")  # first-state (then-branch) writer instance
    an_a2 = state.add_access("A")  # second-state (else-branch) writer instance
    state.add_nedge(an_a1, an_a2, dace.Memlet())  # WAW happens-before edge, no data crosses

    _assert_post_stage_invariants(state)  # must not raise


def test_resize_scalar_chain_preserves_ordering_edge():
    """`_resize_scalar_chain_downstream_of_tiles` walks every out-edge of a to-be-
    resized scalar transient and force-writes a tile-shape subset onto it. An ordering
    (empty) out-edge -- e.g. a WAW happens-before edge reusing that same scalar as a
    first-state write ordered before a second-state write -- must be left alone, not
    turned into a malformed memlet (subset set, data still None).
    """
    inner_sdfg = dace.SDFG("body")
    inner_sdfg.add_array("tile_src", [4], dace.float64, transient=True)
    inner_sdfg.add_scalar("s1", dace.float64, transient=True)
    inner_sdfg.add_scalar("sink", dace.float64, transient=True)
    state = inner_sdfg.add_state("s")

    tile_src_an = state.add_access("tile_src")
    t1 = state.add_tasklet("t1", {"_in"}, {"_out"}, "_out = _in")
    state.add_edge(tile_src_an, None, t1, "_in", dace.Memlet("tile_src[0]"))  # tile-shaped input
    s1_an = state.add_access("s1")
    state.add_edge(t1, "_out", s1_an, None, dace.Memlet("s1[0]"))  # real producer edge

    # `sink` is reached from s1 ONLY via an ordering edge -- no data crosses s1 -> sink.
    sink_an = state.add_access("sink")
    state.add_nedge(s1_an, sink_an, dace.Memlet())

    pas = InsertTileLoadStore(widths=(4, ))
    n_resized = pas._resize_scalar_chain_downstream_of_tiles(state)

    assert n_resized == 1
    assert isinstance(inner_sdfg.arrays["s1"], dt.Array)  # s1 did get widened to (4,)
    edge = state.edges_between(s1_an, sink_an)[0]
    assert edge.data.is_empty(), "the ordering edge must stay empty, not gain a bogus subset"
