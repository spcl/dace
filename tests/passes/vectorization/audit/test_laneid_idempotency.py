# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
End-to-end tripwire: ``expand_interstate_assignments_to_lanes`` and
``resolve_missing_laneid_symbols`` must be idempotent — running the same pass
twice on the same SDFG must not double-suffix names into
``foo_laneid_3_laneid_0``-style garbage.

The bug was previously masked by an in-pass regex hack
(``re.search(r"_(\\d+)$", k)``) that worked for the LHS but did not catch the
free-symbol substitution on the RHS, so a second invocation could still produce
doubly-encoded names. The current implementation uses ``LaneIdScheme.parse``
plus an explicit side-set check on free symbols, so the second call is a no-op.
"""
import dace
from dace.transformation.passes.vectorization.vectorization_utils import (
    LaneIdScheme,
    expand_interstate_assignments_to_lanes,
)


def _build_minimal_sdfg_for_idempotency():
    """Build a parent SDFG whose innermost map contains a NestedSDFG with one
    interstate edge carrying ``k = arr[i] + 1`` so the lane-expansion has actual
    work to do.
    """
    parent = dace.SDFG("parent")
    parent.add_array("arr", shape=(64, ), dtype=dace.float64)
    parent.add_array("out", shape=(64, ), dtype=dace.float64)
    parent.add_symbol("k", dace.int64)

    pstate = parent.add_state("p", is_start_block=True)

    # Inner SDFG: two states with an interstate edge that assigns k = arr[i] + 1.
    inner = dace.SDFG("inner")
    inner.add_array("arr", shape=(64, ), dtype=dace.float64)
    inner.add_array("out", shape=(64, ), dtype=dace.float64)
    inner.add_symbol("i", dace.int64)
    inner.add_symbol("k", dace.int64)
    s0 = inner.add_state("s0", is_start_block=True)
    s1 = inner.add_state("s1")
    inner.add_edge(s0, s1, dace.InterstateEdge(assignments={"k": "arr[i] + 1"}))
    # s1 has a tasklet that uses k somewhere — kept tiny.
    an = s1.add_access("out")
    t = s1.add_tasklet("write_k", set(), {"_o"}, "_o = k")
    s1.add_edge(t, "_o", an, None, dace.memlet.Memlet("out[i]"))

    # Build the parent map + nsdfg.
    me, mx = pstate.add_map("vmap", {"i": "0:64:8"})
    arr_an = pstate.add_access("arr")
    out_an = pstate.add_access("out")
    nsdfg_node = pstate.add_nested_sdfg(inner, {"arr"}, {"out"})
    pstate.add_edge(arr_an, None, me, "IN_arr", dace.memlet.Memlet.from_array("arr", parent.arrays["arr"]))
    me.add_in_connector("IN_arr")
    pstate.add_edge(me, "OUT_arr", nsdfg_node, "arr", dace.memlet.Memlet.from_array("arr", parent.arrays["arr"]))
    me.add_out_connector("OUT_arr")
    pstate.add_edge(nsdfg_node, "out", mx, "IN_out", dace.memlet.Memlet.from_array("out", parent.arrays["out"]))
    mx.add_in_connector("IN_out")
    pstate.add_edge(mx, "OUT_out", out_an, None, dace.memlet.Memlet.from_array("out", parent.arrays["out"]))
    mx.add_out_connector("OUT_out")
    return parent, pstate, inner, nsdfg_node


def _interstate_assignments(sdfg):
    """Return a dict of {(src.label, dst.label): {assign_var: rhs}} so two SDFGs can be
    structurally compared independent of edge identity."""
    return {(e.src.label, e.dst.label): dict(e.data.assignments) for e in sdfg.all_interstate_edges()}


def test_expand_interstate_assignments_to_lanes_is_idempotent():
    parent, pstate, inner, nsdfg_node = _build_minimal_sdfg_for_idempotency()

    # First call — produce per-lane keys.
    expand_interstate_assignments_to_lanes(inner,
                                           nsdfg_node,
                                           pstate,
                                           vector_width=8,
                                           invariant_data=set(),
                                           vector_map_param="i")
    after_first = _interstate_assignments(inner)
    syms_after_first = set(inner.symbols.keys())

    # Sanity: at least one per-lane key was produced.
    assert any(LaneIdScheme.is_laneid(k) for assigns in after_first.values()
               for k in assigns), f"Expected lane-encoded keys after first call, got {after_first}"

    # Second call — must be a no-op (no NEW lane keys, no doubly-suffixed names).
    expand_interstate_assignments_to_lanes(inner,
                                           nsdfg_node,
                                           pstate,
                                           vector_width=8,
                                           invariant_data=set(),
                                           vector_map_param="i")
    after_second = _interstate_assignments(inner)
    syms_after_second = set(inner.symbols.keys())

    # No double-suffixed names anywhere.
    for assigns in after_second.values():
        for k in assigns:
            parsed = LaneIdScheme.parse(k)
            if parsed is not None:
                base, _ = parsed
                assert not LaneIdScheme.is_laneid(base), (f"Doubly-encoded lane id on the second call: {k!r}")

    # Assignments stable across the second call.
    assert after_second == after_first, (f"Second call mutated assignments — not idempotent.\n"
                                         f"  before: {after_first}\n"
                                         f"  after:  {after_second}")

    # No new symbols added on the second call.
    assert syms_after_second == syms_after_first, (
        f"Second call added new symbols: {syms_after_second - syms_after_first}")
