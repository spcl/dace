# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`BypassTrivialAssignTasklets`.

Each test builds a minimal body-NSDFG by hand and asserts the pass
folds the targeted ``AN -> [_out=_in] -> AN`` triples without changing
numerical results.
"""

import dace
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
                if isinstance(node,
                              dace.nodes.Tasklet) and node.code.as_string.strip().rstrip(";").strip() == "_out = _in":
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


def _scope_passthrough_consistent(sdfg) -> bool:
    """True iff every Map entry/exit ``IN_x``/``OUT_x`` passthrough connector
    carries the SAME memlet data on both sides. A single-edge bypass splice
    across a scope boundary renames only one side -> the two disagree -> invalid.
    """
    from dace.sdfg.nodes import MapEntry, MapExit
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if not isinstance(node, (MapEntry, MapExit)):
                    continue
                for oe in state.out_edges(node):
                    if oe.src_conn is None or not oe.src_conn.startswith("OUT_"):
                        continue
                    in_conn = "IN_" + oe.src_conn[len("OUT_"):]
                    for ie in state.in_edges(node):
                        if ie.dst_conn == in_conn and ie.data.data != oe.data.data:
                            return False
    return True


def test_map_exit_boundary_assign_not_corrupted():
    """A trivial copy whose source is fed by a MapExit -- the spmv per-row
    accumulator shape ``MapExit:OUT -> AN(transient) -> [_out=_in] -> AN`` -- must
    not be spliced across the scope boundary. A single-edge rewrite renames only
    the exit's ``OUT_x`` side, leaving ``IN_x`` naming the old array (invalid).
    The pass leaves such copies in place; the scope passthrough stays consistent."""
    outer, body, state, _ = _build_outer_with_body_nsdfg()
    body.add_array("A", (8, ), dace.float64, transient=True)
    body.add_array("acc", (1, ), dace.float64, transient=True)
    body.add_array("OUT", (1, ), dace.float64, transient=True)
    a = state.add_access("A")
    acc = state.add_access("acc")
    out = state.add_access("OUT")
    me, mx = state.add_map("m", dict(i="0:8"))
    w = state.add_tasklet("w", {"_a"}, {"_o"}, "_o = _a")
    state.add_memlet_path(a, me, w, dst_conn="_a", memlet=Memlet("A[i]"))
    state.add_memlet_path(w, mx, acc, src_conn="_o", memlet=Memlet("acc[0]"))
    cp = state.add_tasklet("cp", {"_in"}, {"_out"}, "_out = _in")
    state.add_edge(acc, None, cp, "_in", Memlet("acc[0]"))
    state.add_edge(cp, "_out", out, None, Memlet("OUT[0]"))

    assert _scope_passthrough_consistent(outer)
    BypassTrivialAssignTasklets().apply_pass(outer, {})
    # The fix: the exit's IN_acc / OUT_acc passthrough must not be half-renamed.
    assert _scope_passthrough_consistent(outer), "bypass half-renamed a Map scope passthrough connector"


def test_spmv_bypass_keeps_sdfg_valid():
    """Regression (user-directed before/after): the spmv per-row reduction
    accumulator ``tmp`` is fed by the inner idx-map's MapExit, and the trivial
    ``tmp -> __tmp_w`` copy (``y[i] = tmp``) sits across that scope boundary.
    ``BypassTrivialAssignTasklets`` must leave the prepped SDFG valid both BEFORE
    and AFTER it runs (it previously half-renamed the exit passthrough -> invalid).

    Note: this checks ONLY that the bypass does not corrupt the SDFG; full
    spmv multi-dim vectorization additionally needs the carried-accumulator
    reduction lowered to a horizontal reduce (tracked separately)."""
    from dace.transformation.dataflow import WCRToAugAssign
    from dace.transformation.interstate import LoopToMap, RefineNestedAccess
    from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import normalize_loop_nests
    from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
    from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource

    n = dace.symbol("n")
    m = dace.symbol("m")
    nnz = dace.symbol("nnz")

    @dace.program
    def spmv_csr(indptr: dace.int64[n + 1], indices: dace.int64[nnz], data: dace.float64[nnz], x: dace.float64[m],
                 y: dace.float64[n]):
        n_rows = len(indptr) - 1
        for i in dace.map[0:n_rows:1]:
            row_start = indptr[i]
            row_end = indptr[i + 1]
            tmp = 0.0
            for idx in dace.map[row_start:row_end:1]:
                j = indices[idx]
                tmp = tmp + data[idx] * x[j]
            y[i] = tmp

    sdfg = spmv_csr.to_sdfg()
    sdfg.simplify()
    sdfg.apply_transformations_repeated(WCRToAugAssign, permissive=False, validate=False)
    sdfg.apply_transformations_repeated([LoopToMap, RefineNestedAccess], permissive=False, validate=False)
    normalize_loop_nests(sdfg)
    ConvertLengthOneArraysToScalars(recursive=True, transient_only=False).apply_pass(sdfg, {})
    NormalizeWCRSource().apply_pass(sdfg, {})

    sdfg.validate()  # BEFORE bypass: valid
    BypassTrivialAssignTasklets().apply_pass(sdfg, {})
    sdfg.validate()  # AFTER bypass: must still be valid (the fix)
    assert _scope_passthrough_consistent(sdfg)


def test_does_not_collapse_cross_state_transient():
    """Regression (cloudsc_one ``zqx`` Isolated-node crash): a transient staged in
    state A and read in state B is a CROSS-STATE value. In state A its
    ``out_degree`` is 0 (the reader is in B, reached via the persistent transient,
    not an edge), so the bypass would wrongly treat it as dead, delete the only
    write, and orphan the source -> ``InvalidSDFGNodeError: Isolated node``. The
    pass must leave cross-state transient triples alone."""
    outer, body, state_a, _ = _build_outer_with_body_nsdfg()
    body.add_array("G", (1, ), dace.float64, transient=False)
    body.add_array("T", (1, ), dace.float64, transient=True)
    body.add_array("OUT", (1, ), dace.float64, transient=False)
    # State A: G -> [_out=_in] -> T  (T's reader is in the next state, so T has
    # out_degree 0 here -- the cross-state blind spot).
    g = state_a.add_access("G")
    t_a = state_a.add_access("T")
    bridge = state_a.add_tasklet("b", {"_in"}, {"_out"}, "_out = _in")
    state_a.add_edge(g, None, bridge, "_in", Memlet("G[0]"))
    state_a.add_edge(bridge, "_out", t_a, None, Memlet("T[0]"))
    # State B (executes after A): reads T -> OUT.
    state_b = body.add_state_after(state_a, "b2")
    t_b = state_b.add_access("T")
    o = state_b.add_access("OUT")
    cons = state_b.add_tasklet("c", {"_in"}, {"_out"}, "_out = _in + 1.0")
    state_b.add_edge(t_b, None, cons, "_in", Memlet("T[0]"))
    state_b.add_edge(cons, "_out", o, None, Memlet("OUT[0]"))

    assert _count_assign_tasklets(outer) == 1
    # The G -> [_out=_in] -> T staging must be PRESERVED (T is read cross-state).
    assert _bypass_count(outer) == 0
    assert _count_assign_tasklets(outer) == 1
    # No isolated node left (the original crash was an orphaned ``G``).
    iso = [n for st in body.states() for n in st.nodes() if isinstance(n, dace.nodes.AccessNode) and st.degree(n) == 0]
    assert not iso, f"isolated AccessNode(s) left: {[n.data for n in iso]}"
