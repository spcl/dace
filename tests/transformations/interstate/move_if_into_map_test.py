# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the MoveIfIntoMap transformation.

These mirror the ICON ``_for_it_44`` motif: a conditional block that lives in
the body of an outer map and guards an inner map. The transformation pushes
the guard past the inner map so the two maps can be fused/collapsed by later
passes.
"""
import copy

import numpy as np

import dace
from dace.sdfg.nodes import NestedSDFG
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import MoveIfIntoMap


def _count_conditional_blocks(sdfg: dace.SDFG) -> int:
    total = 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, ConditionalBlock):
            total += 1
    return total


def _inner_nsdfg_contains_conditional(outer_sdfg: dace.SDFG) -> bool:
    """After applying the transformation the conditional block should live
    inside an inner nested SDFG (the body of the innermost map)."""
    for node, _ in outer_sdfg.all_nodes_recursive():
        if not isinstance(node, NestedSDFG):
            continue
        inner = node.sdfg
        for block in inner.nodes():
            if isinstance(block, ConditionalBlock):
                return True
    return False


def _run_and_compare(prog, inputs, expected_apps: int, simplify: bool = False):
    sdfg: dace.SDFG = prog.to_sdfg(simplify=simplify)
    sdfg.validate()

    reference = copy.deepcopy(inputs)
    sdfg(**reference)

    sdfg2: dace.SDFG = prog.to_sdfg(simplify=simplify)
    applied = sdfg2.apply_transformations_repeated(MoveIfIntoMap)
    assert applied == expected_apps, f"expected {expected_apps} applications, got {applied}"
    sdfg2.validate()

    transformed = copy.deepcopy(inputs)
    sdfg2(**transformed)

    for key in reference:
        np.testing.assert_allclose(transformed[key],
                                   reference[key],
                                   rtol=1e-5,
                                   atol=1e-6,
                                   err_msg=f"Mismatch for {key}")

    return sdfg2


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_move_if_into_map_basic():
    """Core _for_it_44-style pattern: outer map, scalar guard, inner map."""

    N, M = 6, 5

    @dace.program
    def tester(A: dace.float64[N, M], cond: dace.int32):
        for i in dace.map[0:N]:
            if cond == 1:
                for j in dace.map[0:M]:
                    A[i, j] = A[i, j] + 1.0

    rng = np.random.default_rng(0)
    inputs_on = {
        "A": rng.random((N, M)).copy(),
        "cond": np.int32(1),
    }
    inputs_off = {
        "A": rng.random((N, M)).copy(),
        "cond": np.int32(0),
    }

    sdfg_on = _run_and_compare(tester, inputs_on, expected_apps=1)
    _run_and_compare(tester, inputs_off, expected_apps=1)

    assert _count_conditional_blocks(sdfg_on) == 1
    assert _inner_nsdfg_contains_conditional(sdfg_on)


def test_move_if_into_map_symbolic_condition():
    """Condition uses a combination of map parameter and outer symbol."""

    N, M = 8, 4

    @dace.program
    def tester(A: dace.float64[N, M], threshold: dace.int32):
        for i in dace.map[0:N]:
            if i < threshold:
                for j in dace.map[0:M]:
                    A[i, j] = A[i, j] * 2.0

    rng = np.random.default_rng(1)
    inputs = {
        "A": rng.random((N, M)).copy(),
        "threshold": np.int32(5),
    }

    sdfg = _run_and_compare(tester, inputs, expected_apps=1)
    assert _inner_nsdfg_contains_conditional(sdfg)


def test_move_if_into_map_multiple_reads_and_writes():
    """Body has multiple access nodes on both sides of the inner nested SDFG."""

    N, M = 4, 4

    @dace.program
    def tester(A: dace.float64[N, M], B: dace.float64[N, M], cond: dace.int32):
        for i in dace.map[0:N]:
            if cond == 1:
                for j in dace.map[0:M]:
                    A[i, j] = A[i, j] + B[i, j]

    rng = np.random.default_rng(2)
    inputs = {
        "A": rng.random((N, M)).copy(),
        "B": rng.random((N, M)).copy(),
        "cond": np.int32(1),
    }

    sdfg = _run_and_compare(tester, inputs, expected_apps=1)
    assert _inner_nsdfg_contains_conditional(sdfg)


def test_move_if_into_map_no_apply_missing_inner_map():
    """No inner map in the branch: should not match."""

    N = 6

    @dace.program
    def tester(A: dace.float64[N], cond: dace.int32):
        for i in dace.map[0:N]:
            if cond == 1:
                A[i] = A[i] + 1.0

    sdfg: dace.SDFG = tester.to_sdfg(simplify=False)
    applied = sdfg.apply_transformations_repeated(MoveIfIntoMap)
    assert applied == 0


def test_move_if_into_map_no_apply_else_branch():
    """Conditional has an else branch -> not supported."""

    N, M = 4, 4

    @dace.program
    def tester(A: dace.float64[N, M], cond: dace.int32):
        for i in dace.map[0:N]:
            if cond == 1:
                for j in dace.map[0:M]:
                    A[i, j] = A[i, j] + 1.0
            else:
                for j in dace.map[0:M]:
                    A[i, j] = A[i, j] - 1.0

    sdfg: dace.SDFG = tester.to_sdfg(simplify=False)
    applied = sdfg.apply_transformations_repeated(MoveIfIntoMap)
    assert applied == 0


def test_move_if_into_map_no_race_with_upstream_symbol_assignment():
    """If the ConditionalBlock's incoming interstate edge already assigns a
    symbol that the guard condition reads (``_if_cond_24`` produced from
    ``levmask[...]``), folding a new condition-symbol onto the same edge
    would create a race (read + write of the same symbol in one edge).

    The transformation avoids this by moving the condition evaluation
    INSIDE the inner NSDFG: the ``ConditionalBlock`` placed around the
    inner map body uses the original branch condition directly, and the
    free symbols it reads are passed in through ``symbol_mapping``. The
    outer/enclosing SDFG edges keep their pre-existing assignments
    unchanged, so the outer map body stays collapse-friendly.

    Structure built here:

        outer_sdfg:
            outer_state:
                MapEntry(jb=1:M+1) -> NestedSDFG(mid) -> MapExit(jb)

        mid (body of outer map, the ``enclosing_sdfg``):
            pre --[{_if_cond_24: levmask[jb-1, 0]}]--> cb
            cb: ConditionalBlock("(_if_cond_24 == 1)")
                branch_state:
                    MapEntry(jc=0:N) -> NestedSDFG(inner) -> MapExit(jc)

        inner: out[jb, jc] = in[jb, jc] * 2
    """
    from dace import memlet as mm
    from dace.properties import CodeBlock
    from dace.sdfg import SDFG
    from dace.sdfg.state import ControlFlowRegion

    M_sym = dace.symbol("M")
    N_sym = dace.symbol("N")

    # --- inner SDFG (body of inner map) ---
    inner = SDFG("inner_tester")
    inner.add_array("a_in", [1], dace.float64)
    inner.add_array("a_out", [1], dace.float64)
    is_ = inner.add_state("s", is_start_block=True)
    ir = is_.add_read("a_in")
    iw = is_.add_write("a_out")
    it = is_.add_tasklet("doub", {"x"}, {"y"}, "y = x * 2")
    is_.add_edge(ir, None, it, "x", mm.Memlet("a_in[0]"))
    is_.add_edge(it, "y", iw, None, mm.Memlet("a_out[0]"))

    # --- mid SDFG (body of outer map, contains the ConditionalBlock) ---
    mid = SDFG("mid_tester")
    mid.add_array("A_in", [M_sym, N_sym], dace.float64)
    mid.add_array("A_out", [M_sym, N_sym], dace.float64)
    mid.add_array("levmask", [M_sym, N_sym], dace.int32)
    mid.add_symbol("jb", dace.int32)
    mid.add_symbol("_if_cond_24", dace.int32)

    pre = mid.add_state("pre", is_start_block=True)
    cb = ConditionalBlock("cb")
    mid.add_node(cb)
    mid.add_edge(pre, cb, dace.InterstateEdge(
        assignments={"_if_cond_24": "levmask[jb, 0]"}))

    branch_body = ControlFlowRegion("branch", sdfg=mid)
    cb.add_branch(CodeBlock("(_if_cond_24 == 1)"), branch_body)
    bstate = branch_body.add_state("branch_state", is_start_block=True)
    ar = bstate.add_read("A_in")
    aw = bstate.add_write("A_out")
    me, mx = bstate.add_map("jc_map", {"jc": "0:N"})
    me.add_in_connector("IN_a"); me.add_out_connector("OUT_a")
    mx.add_in_connector("IN_a"); mx.add_out_connector("OUT_a")
    ns = bstate.add_nested_sdfg(inner, {"a_in"}, {"a_out"})
    bstate.add_edge(ar, None, me, "IN_a", mm.Memlet("A_in[0:M, 0:N]"))
    bstate.add_edge(me, "OUT_a", ns, "a_in", mm.Memlet("A_in[jb, jc]"))
    bstate.add_edge(ns, "a_out", mx, "IN_a", mm.Memlet("A_out[jb, jc]"))
    bstate.add_edge(mx, "OUT_a", aw, None, mm.Memlet("A_out[0:M, 0:N]"))

    # --- outer SDFG ---
    outer = SDFG("outer_tester")
    outer.add_array("A_in", [M_sym, N_sym], dace.float64)
    outer.add_array("A_out", [M_sym, N_sym], dace.float64)
    outer.add_array("levmask", [M_sym, N_sym], dace.int32)

    ostate = outer.add_state("outer_state", is_start_block=True)
    AR = ostate.add_read("A_in")
    LR = ostate.add_read("levmask")
    AW = ostate.add_write("A_out")
    OE, OX = ostate.add_map("jb_map", {"jb": "0:M"})
    for c in ("A_in", "levmask"):
        OE.add_in_connector("IN_" + c); OE.add_out_connector("OUT_" + c)
    OX.add_in_connector("IN_A_out"); OX.add_out_connector("OUT_A_out")
    mid_node = ostate.add_nested_sdfg(mid, {"A_in", "levmask"}, {"A_out"})
    ostate.add_edge(AR, None, OE, "IN_A_in", mm.Memlet("A_in[0:M, 0:N]"))
    ostate.add_edge(LR, None, OE, "IN_levmask", mm.Memlet("levmask[0:M, 0:N]"))
    ostate.add_edge(OE, "OUT_A_in", mid_node, "A_in", mm.Memlet("A_in[0:M, 0:N]"))
    ostate.add_edge(OE, "OUT_levmask", mid_node, "levmask", mm.Memlet("levmask[0:M, 0:N]"))
    ostate.add_edge(mid_node, "A_out", OX, "IN_A_out", mm.Memlet("A_out[0:M, 0:N]"))
    ostate.add_edge(OX, "OUT_A_out", AW, None, mm.Memlet("A_out[0:M, 0:N]"))

    outer.validate()

    # The offending pre-transform edge: _if_cond_24 is assigned AND would be
    # read on the same edge if MoveIfIntoMap put the condition there.
    pre_found = False
    for e, _ in outer.all_edges_recursive():
        if isinstance(e.data, dace.InterstateEdge) and "_if_cond_24" in e.data.assignments:
            pre_found = True
            break
    assert pre_found

    applied = outer.apply_transformations_repeated(MoveIfIntoMap)
    assert applied == 1, applied
    outer.validate()  # Race condition would have thrown here.

    # The enclosing edges must not gain a cond-symbol assignment that
    # would race with ``_if_cond_24``.
    for e, g in outer.all_edges_recursive():
        if g is outer:  # only the outer SDFG, not nested ones
            continue
        if not isinstance(e.data, dace.InterstateEdge):
            continue
        if g.name == "mid_tester":  # the enclosing SDFG
            for sym in e.data.assignments:
                assert not sym.endswith("_cond"), (
                    f"condition should be moved inside inner NSDFG, found {sym} on "
                    f"mid-level edge: {e.data.assignments}")

    # The inner NSDFG body should now contain a ConditionalBlock whose
    # guard reads ``_if_cond_24`` (piped through symbol_mapping).
    found_moved_cb = False
    for g in outer.all_sdfgs_recursive():
        if g.name.startswith("inner_tester"):
            for block in g.nodes():
                if isinstance(block, ConditionalBlock):
                    guard = block.branches[0][0].as_string
                    if "_if_cond_24" in guard:
                        found_moved_cb = True
    assert found_moved_cb, "no ConditionalBlock reading _if_cond_24 inside inner NSDFG"

    # Numerical verification: the kernel computes ``A_out[jb, jc] = 2 *
    # A_in[jb, jc]`` when ``levmask[jb, 0] == 1``, and leaves A_out at its
    # initial value otherwise.
    M_val, N_val = 4, 3
    a_in = np.arange(M_val * N_val, dtype=np.float64).reshape(M_val, N_val).copy()
    levmask = np.zeros((M_val, N_val), dtype=np.int32)
    levmask[1:3, 0] = 1  # predicate fires for two of the four blocks
    a_out = np.full((M_val, N_val), -1.0, dtype=np.float64)
    expected = np.where(levmask[:, 0:1] == 1, 2 * a_in, a_out)

    outer(A_in=a_in, A_out=a_out, levmask=levmask, M=M_val, N=N_val)
    np.testing.assert_allclose(a_out, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    test_move_if_into_map_basic()
    test_move_if_into_map_symbolic_condition()
    test_move_if_into_map_multiple_reads_and_writes()
    test_move_if_into_map_no_apply_missing_inner_map()
    test_move_if_into_map_no_apply_else_branch()
    test_move_if_into_map_no_race_with_upstream_symbol_assignment()
    print("All tests passed.")
