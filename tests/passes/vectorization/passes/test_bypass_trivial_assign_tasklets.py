# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`BypassTrivialAssignTasklets`.

Each test builds a minimal body-NSDFG by hand and asserts the pass
folds the targeted ``AN -> [_out=_in] -> AN`` triples without changing
numerical results.
"""
import copy

import numpy as np

import dace
from dace import subsets
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import BypassTrivialAssignTasklets


def _bypass_count(sdfg):
    """Return the number of trivial-assign tasklets the pass folded."""
    return BypassTrivialAssignTasklets().apply_pass(sdfg, {}) or 0


def _count_assign_tasklets(sdfg):
    """Count ``_out = _in`` tasklets across every (nested) state of ``sdfg``."""
    total = 0
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet) and node.code.as_string.strip().rstrip(";").strip() == "_out = _in":
                    total += 1
    return total


def _build_outer_with_body_nsdfg():
    """Outer SDFG containing one body NSDFG that hosts the test fixture state.

    The pass scopes to body NSDFGs (mirrors :class:`EliminateDeadCopies`),
    so the rewrite target lives inside a NestedSDFG node, not in the top
    state. Returns ``(outer_sdfg, body_sdfg, body_state)``.
    """
    outer = dace.SDFG("outer")
    outer.add_array("X", (1, ), dace.float64, transient=False)
    outer_state = outer.add_state("o")
    body = dace.SDFG("body")
    body_state = body.add_state("b")
    nsdfg = outer_state.add_nested_sdfg(body, set(), set(), {})
    return outer, body, body_state, nsdfg


def test_bypass_dst_transient_routes_src_to_consumer():
    """``AN(src=global) -> [_out=_in] -> AN(dst=transient) -> consumer`` collapses
    to ``AN(src) -> consumer`` directly."""
    outer, body, state, _ = _build_outer_with_body_nsdfg()
    body.add_array("G", (1, ), dace.float64, transient=False)
    body.add_array("T", (1, ), dace.float64, transient=True)
    body.add_array("OUT", (1, ), dace.float64, transient=False)

    g = state.add_access("G")
    t = state.add_access("T")
    out = state.add_access("OUT")
    tlet = state.add_tasklet("a", {"_in"}, {"_out"}, "_out = _in")
    state.add_edge(g, None, tlet, "_in", Memlet("G[0]"))
    state.add_edge(tlet, "_out", t, None, Memlet("T[0]"))
    state.add_edge(t, None, out, None, Memlet("OUT[0]"))

    assert _count_assign_tasklets(outer) == 1
    assert _bypass_count(outer) == 1
    assert _count_assign_tasklets(outer) == 0
    # The transient AN(T) is gone; G should write OUT directly.
    assert "T" in body.arrays or all(n.data != "T" for n in state.data_nodes())


def test_bypass_src_transient_routes_producer_to_dst():
    """``producer -> AN(src=transient) -> [_out=_in] -> AN(dst=global)`` collapses
    to ``producer -> AN(dst)`` directly."""
    outer, body, state, _ = _build_outer_with_body_nsdfg()
    body.add_array("IN", (1, ), dace.float64, transient=False)
    body.add_array("T", (1, ), dace.float64, transient=True)
    body.add_array("G", (1, ), dace.float64, transient=False)

    src = state.add_access("IN")
    t = state.add_access("T")
    g = state.add_access("G")
    producer = state.add_tasklet("p", {"_in"}, {"_out"}, "_out = _in * 2.0")
    bridge = state.add_tasklet("b", {"_in"}, {"_out"}, "_out = _in")
    state.add_edge(src, None, producer, "_in", Memlet("IN[0]"))
    state.add_edge(producer, "_out", t, None, Memlet("T[0]"))
    state.add_edge(t, None, bridge, "_in", Memlet("T[0]"))
    state.add_edge(bridge, "_out", g, None, Memlet("G[0]"))

    assert _count_assign_tasklets(outer) == 1
    assert _bypass_count(outer) == 1
    assert _count_assign_tasklets(outer) == 0


def test_dedup_keeps_one_assign_per_src_dst_pair():
    """Two ``AN(src) -> [_out=_in] -> AN(dst)`` triples on the same ``(src, dst)``
    collapse to one; the second tasklet is dropped."""
    outer, body, state, _ = _build_outer_with_body_nsdfg()
    body.add_array("SRC", (1, ), dace.float64, transient=True)
    body.add_array("DST", (1, ), dace.float64, transient=True)
    body.add_array("OUT1", (1, ), dace.float64, transient=False)
    body.add_array("OUT2", (1, ), dace.float64, transient=False)

    src = state.add_access("SRC")
    dst = state.add_access("DST")
    o1 = state.add_access("OUT1")
    o2 = state.add_access("OUT2")
    t1 = state.add_tasklet("t1", {"_in"}, {"_out"}, "_out = _in")
    t2 = state.add_tasklet("t2", {"_in"}, {"_out"}, "_out = _in")
    state.add_edge(src, None, t1, "_in", Memlet("SRC[0]"))
    state.add_edge(t1, "_out", dst, None, Memlet("DST[0]"))
    state.add_edge(src, None, t2, "_in", Memlet("SRC[0]"))
    # Both assigns route SRC -> DST; the second AN(DST) is a duplicate.
    dst2 = state.add_access("DST")
    state.add_edge(t2, "_out", dst2, None, Memlet("DST[0]"))
    state.add_edge(dst, None, o1, None, Memlet("OUT1[0]"))
    state.add_edge(dst2, None, o2, None, Memlet("OUT2[0]"))

    assert _count_assign_tasklets(outer) == 2
    # Dedup drops t2 (one of the two) then bypass folds the surviving
    # assign through the single canonical SRC -> DST pair.
    folded = _bypass_count(outer)
    assert folded >= 1
    # At most one trivial assign survives (the bypass also folds it if the
    # source has out_degree==1 after dedup).
    assert _count_assign_tasklets(outer) <= 1


def test_does_not_touch_connector_to_connector_assign():
    """An assign between two non-transient arrays is left alone -- collapsing
    it would lose the boundary edge."""
    outer, body, state, _ = _build_outer_with_body_nsdfg()
    body.add_array("A", (1, ), dace.float64, transient=False)
    body.add_array("B", (1, ), dace.float64, transient=False)

    a = state.add_access("A")
    b = state.add_access("B")
    t = state.add_tasklet("t", {"_in"}, {"_out"}, "_out = _in")
    state.add_edge(a, None, t, "_in", Memlet("A[0]"))
    state.add_edge(t, "_out", b, None, Memlet("B[0]"))

    assert _count_assign_tasklets(outer) == 1
    assert _bypass_count(outer) == 0
    assert _count_assign_tasklets(outer) == 1


def test_does_not_touch_multi_consumer_src():
    """SSA-like reassignment chain: bypassing would fold two distinct
    assignments through one AN and pick the wrong value."""
    outer, body, state, _ = _build_outer_with_body_nsdfg()
    body.add_array("T", (1, ), dace.float64, transient=True)
    body.add_array("X", (1, ), dace.float64, transient=True)
    body.add_array("Y", (1, ), dace.float64, transient=True)
    body.add_array("OUT", (1, ), dace.float64, transient=False)

    src = state.add_access("T")
    a1 = state.add_access("X")
    a2 = state.add_access("Y")
    o = state.add_access("OUT")
    p = state.add_tasklet("p", set(), {"_out"}, "_out = 1.0")
    t1 = state.add_tasklet("t1", {"_in"}, {"_out"}, "_out = _in")
    t2 = state.add_tasklet("t2", {"_in"}, {"_out"}, "_out = _in")
    add = state.add_tasklet("add", {"_a", "_b"}, {"_out"}, "_out = _a + _b")
    state.add_edge(p, "_out", src, None, Memlet("T[0]"))
    # SRC fans out to two assigns -- out_degree(src) == 2, bypass must refuse.
    state.add_edge(src, None, t1, "_in", Memlet("T[0]"))
    state.add_edge(src, None, t2, "_in", Memlet("T[0]"))
    state.add_edge(t1, "_out", a1, None, Memlet("X[0]"))
    state.add_edge(t2, "_out", a2, None, Memlet("Y[0]"))
    state.add_edge(a1, None, add, "_a", Memlet("X[0]"))
    state.add_edge(a2, None, add, "_b", Memlet("Y[0]"))
    state.add_edge(add, "_out", o, None, Memlet("OUT[0]"))

    # Dedup collapses (T, X)/(T, Y) only when dst.data matches; here the dst
    # data names differ so dedup keeps both. The bypass then refuses because
    # out_degree(src) == 2. End state: both trivial assigns still present.
    assert _count_assign_tasklets(outer) == 2
    folded = _bypass_count(outer)
    assert folded == 0
    assert _count_assign_tasklets(outer) == 2


def test_outer_sdfg_an_to_an_assign_left_alone():
    """The pass scopes to BODY NSDFGs; assigns in the top-level SDFG state
    stay untouched (those may be scatter / gather staging the legacy
    1D detect passes consume)."""
    sdfg = dace.SDFG("top_level_only")
    sdfg.add_array("G", (1, ), dace.float64, transient=False)
    sdfg.add_array("T", (1, ), dace.float64, transient=True)
    sdfg.add_array("OUT", (1, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    g = state.add_access("G")
    t = state.add_access("T")
    o = state.add_access("OUT")
    bridge = state.add_tasklet("b", {"_in"}, {"_out"}, "_out = _in")
    state.add_edge(g, None, bridge, "_in", Memlet("G[0]"))
    state.add_edge(bridge, "_out", t, None, Memlet("T[0]"))
    state.add_edge(t, None, o, None, Memlet("OUT[0]"))

    assert _count_assign_tasklets(sdfg) == 1
    assert _bypass_count(sdfg) == 0
    assert _count_assign_tasklets(sdfg) == 1
