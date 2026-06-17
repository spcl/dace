# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit + e2e tests for the in-place-RMW WCR fix and its vectorization
preconditions.

Covers the three pass extensions made for in-place ``a[i] = a[i] + b[i]``:

* :class:`WCRToAugAssign` expr-2 (the ``AccessNode -[wcr]-> AccessNode`` copy
  that canonicalisation lowers an in-place RMW to) -> explicit read-modify-write
  tasklet, no WCR left behind.
* :class:`BypassTrivialAssignTasklets` CARRYING the WCR across a bypassed
  ``_out = _in`` copy (dropping it would degrade the reduction to ``a = b``).
* The two precondition checkers :func:`no_wcr_in_map_body` (legacy) and
  :func:`no_wcr_inside_nested_sdfgs` (multi-dim), incl. the allowed
  scalar-reduction-out form that must NOT trip them.

Plus an e2e in-place RMW through both vectorizers (legacy + tile-node).
"""
import numpy as np
import pytest

import dace
from dace.memlet import Memlet
from dace.transformation.dataflow import WCRToAugAssign
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import BypassTrivialAssignTasklets
from dace.transformation.passes.vectorization.utils.pass_invariants import (no_wcr_in_map_body,
                                                                            no_wcr_inside_nested_sdfgs)
from tests.passes.vectorization.helpers.harness import run_vectorization_test, N


def _all_edges_wcr_free(sdfg: dace.SDFG) -> bool:
    """True iff no edge anywhere in ``sdfg`` (recursively) carries a WCR."""
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for edge in state.edges():
                if edge.data is not None and edge.data.wcr is not None:
                    return False
    return True


def _outer_with_body_nsdfg():
    """``(outer, body, body_state)`` -- a body NSDFG to host fixtures (the
    bypass pass is body-NSDFG-scoped)."""
    outer = dace.SDFG("outer")
    outer.add_array("X", (1, ), dace.float64)
    ostate = outer.add_state("o")
    body = dace.SDFG("body")
    bstate = body.add_state("b")
    ostate.add_nested_sdfg(body, set(), set(), {})
    return outer, body, bstate


# ---------------------------------------------------------------------------
# WCRToAugAssign expr-2: AccessNode -[wcr]-> AccessNode copy.
# ---------------------------------------------------------------------------


def test_wcr_to_augassign_an_to_an_copy():
    """An ``AccessNode(P) -[+=]-> AccessNode(A)`` copy (what canonicalisation
    lowers an in-place ``a += b`` to) converts to an explicit ``A = P + A``
    read-modify-write tasklet, leaving no WCR -- and computing the right value."""
    sdfg = dace.SDFG("an_an_wcr")
    sdfg.add_array("A", (1, ), dace.float64)
    sdfg.add_array("P", (1, ), dace.float64)
    state = sdfg.add_state()
    p = state.add_access("P")
    a = state.add_access("A")
    m = Memlet("A[0]")
    m.wcr = "lambda x, y: x + y"
    state.add_edge(p, None, a, None, m)

    applied = sdfg.apply_transformations_repeated(WCRToAugAssign, permissive=False, validate=False)
    assert applied == 1
    assert _all_edges_wcr_free(sdfg), "WCR survived the AN->AN conversion"
    # An explicit augassign tasklet now materialises the read-modify-write.
    tasklets = [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet)]
    assert len(tasklets) == 1 and "__in1" in tasklets[0].in_connectors and "__in2" in tasklets[0].in_connectors

    sdfg.validate()
    A = np.array([5.0])
    P = np.array([10.0])
    sdfg(A=A, P=P)
    assert np.allclose(A, [15.0]), f"expected 15.0 (5 + 10), got {A}"


# ---------------------------------------------------------------------------
# BypassTrivialAssignTasklets carries the WCR across the bypass.
# ---------------------------------------------------------------------------


def test_bypass_carries_wcr_across_trivial_assign():
    """``B -> [_out=_in] -> WP -(+=)-> A`` (the trivial-copy + WCR shape an
    in-place RMW canonicalises to) -- bypassing the trivial copy must CARRY the
    WCR onto the spliced ``B -> A`` edge, not drop it (dropping degrades the
    reduction to ``A = B``)."""
    outer, body, state = _outer_with_body_nsdfg()
    body.add_array("B", (1, ), dace.float64)
    body.add_array("WP", (1, ), dace.float64, transient=True)
    body.add_array("A", (1, ), dace.float64)

    b = state.add_access("B")
    wp = state.add_access("WP")
    a = state.add_access("A")
    tlet = state.add_tasklet("asg", {"_in"}, {"_out"}, "_out = _in")
    state.add_edge(b, None, tlet, "_in", Memlet("B[0]"))
    state.add_edge(tlet, "_out", wp, None, Memlet("WP[0]"))
    wcr_m = Memlet("A[0]")
    wcr_m.wcr = "lambda x, y: x + y"
    state.add_edge(wp, None, a, None, wcr_m)

    folded = BypassTrivialAssignTasklets().apply_pass(outer, {}) or 0
    assert folded == 1, "the trivial assign should have been bypassed"
    # The WP transient is gone and B -> A now carries the reduction WCR.
    surviving = [e for e in state.edges() if isinstance(e.src, dace.nodes.AccessNode) and e.src.data == "B"]
    assert len(surviving) == 1
    assert surviving[0].data.wcr is not None, "WCR was dropped during the bypass (reduction lost)"


# ---------------------------------------------------------------------------
# Precondition checker: no_wcr_in_map_body (legacy).
# ---------------------------------------------------------------------------


def _map_with_optional_body_wcr(body_wcr: bool):
    """Reduction map ``s += A[i]``. ``body_wcr`` puts the WCR on the in-body
    ``tasklet -> MapExit`` edge (legacy violation); otherwise only the boundary
    ``MapExit -> AccessNode`` edge carries it (allowed -- lifted out of body)."""
    sdfg = dace.SDFG("redmap")
    sdfg.add_array("A", (8, ), dace.float64)
    sdfg.add_array("s", (1, ), dace.float64)
    state = sdfg.add_state()
    a = state.add_access("A")
    s = state.add_access("s")
    me, mx = state.add_map("m", dict(i="0:8"))
    t = state.add_tasklet("t", {"_a"}, {"_o"}, "_o = _a")
    state.add_memlet_path(a, me, t, dst_conn="_a", memlet=Memlet("A[i]"))
    mx.add_in_connector("IN_s")
    mx.add_out_connector("OUT_s")
    body_m = Memlet("s[0]")
    boundary_m = Memlet("s[0]")
    if body_wcr:
        body_m.wcr = "lambda x, y: x + y"
    else:
        boundary_m.wcr = "lambda x, y: x + y"
    state.add_edge(t, "_o", mx, "IN_s", body_m)
    state.add_edge(mx, "OUT_s", s, None, boundary_m)
    return sdfg


def test_no_wcr_in_map_body_detects_body_wcr():
    """A WCR on an in-body edge is flagged."""
    sdfg = _map_with_optional_body_wcr(body_wcr=True)
    assert no_wcr_in_map_body(sdfg) is not None


def test_no_wcr_in_map_body_allows_boundary_reduction():
    """A WCR only on the ``MapExit -> AccessNode`` boundary edge is allowed --
    that is where a reduction's WCR lives once lifted out of the body."""
    sdfg = _map_with_optional_body_wcr(body_wcr=False)
    assert no_wcr_in_map_body(sdfg) is None


# ---------------------------------------------------------------------------
# Precondition checker: no_wcr_inside_nested_sdfgs (multi-dim).
# ---------------------------------------------------------------------------


def test_no_wcr_inside_nested_sdfgs_detects_inner_wcr():
    """A WCR on an edge INSIDE a body NSDFG is flagged (the tile emitters would
    silently drop it)."""
    outer, body, state = _outer_with_body_nsdfg()
    body.add_array("P", (1, ), dace.float64, transient=True)
    body.add_array("A", (1, ), dace.float64)
    p = state.add_access("P")
    a = state.add_access("A")
    m = Memlet("A[0]")
    m.wcr = "lambda x, y: x + y"
    state.add_edge(p, None, a, None, m)
    assert no_wcr_inside_nested_sdfgs(outer) is not None


def test_no_wcr_inside_nested_sdfgs_allows_scalar_reduction_out():
    """The allowed multi-dim form: the body NSDFG writes a scalar that exits via
    a WCR reduction on the ``NestedSDFG -> MapExit`` edge in the PARENT state.
    That edge is OUTSIDE the nested SDFG, so the checker must not flag it (and
    the WCR-free body must pass)."""
    outer = dace.SDFG("outer_red")
    outer.add_array("A", (8, ), dace.float64)
    outer.add_array("s", (1, ), dace.float64)
    ostate = outer.add_state("o")
    a = ostate.add_access("A")
    s = ostate.add_access("s")
    me, mx = ostate.add_map("m", dict(i="0:8"))

    # Body NSDFG: reads a tile element, writes a scalar out connector. NO WCR inside.
    body = dace.SDFG("body")
    body.add_array("_in", (1, ), dace.float64)
    body.add_array("_out", (1, ), dace.float64)
    bstate = body.add_state("b")
    bi = bstate.add_access("_in")
    bo = bstate.add_access("_out")
    bt = bstate.add_tasklet("t", {"x"}, {"y"}, "y = x")
    bstate.add_edge(bi, None, bt, "x", Memlet("_in[0]"))
    bstate.add_edge(bt, "y", bo, None, Memlet("_out[0]"))

    nsdfg = ostate.add_nested_sdfg(body, {"_in"}, {"_out"}, {})
    ostate.add_memlet_path(a, me, nsdfg, dst_conn="_in", memlet=Memlet("A[i]"))
    mx.add_in_connector("IN_s")
    mx.add_out_connector("OUT_s")
    # The reduction WCR lives on the NSDFG -> MapExit boundary, in the PARENT state.
    out_m = Memlet("s[0]")
    out_m.wcr = "lambda x, y: x + y"
    ostate.add_edge(nsdfg, "_out", mx, "IN_s", out_m)
    ostate.add_edge(mx, "OUT_s", s, None, Memlet("s[0]"))

    assert no_wcr_inside_nested_sdfgs(outer) is None


# ---------------------------------------------------------------------------
# e2e: in-place RMW survives both vectorizers (legacy + tile-node).
# ---------------------------------------------------------------------------


@dace.program
def inplace_rmw(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = a[i] + b[i]


def _run_inplace_rmw(config: str):
    """``a[i] = a[i] + b[i]`` must compute ``a + b`` after vectorization (the
    reduction must not be dropped). Exercises the full WCR-elimination path
    (bypass-WCR-carry + WCRToAugAssign) on the given vectorizer."""
    Nv = 16
    a = np.random.random((Nv, ))
    b = np.random.random((Nv, ))
    run_vectorization_test(dace_func=inplace_rmw,
                           arrays={
                               "a": a,
                               "b": b
                           },
                           params={"N": Nv},
                           vector_width=8,
                           sdfg_name=f"inplace_rmw_{config}",
                           vectorize_config=config,
                           remainder_strategy="scalar")


def test_inplace_rmw_e2e_legacy():
    _run_inplace_rmw("legacy_cpu")


def test_inplace_rmw_e2e_tile_nodes():
    _run_inplace_rmw("tile_nodes")


if __name__ == "__main__":
    test_wcr_to_augassign_an_to_an_copy()
    test_bypass_carries_wcr_across_trivial_assign()
    test_no_wcr_in_map_body_detects_body_wcr()
    test_no_wcr_in_map_body_allows_boundary_reduction()
    test_no_wcr_inside_nested_sdfgs_detects_inner_wcr()
    test_no_wcr_inside_nested_sdfgs_allows_scalar_reduction_out()
    test_inplace_rmw_e2e_legacy()
    test_inplace_rmw_e2e_tile_nodes()
    print("all WCR precondition tests passed")
