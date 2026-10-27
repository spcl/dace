# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Preconditions ``MoveIfIntoMap`` must enforce before rewriting.

Each case here was once accepted by ``can_be_applied`` and then crashed mid-rewrite or produced a silently wrong or
invalid SDFG, or was refused although moving the guard is sound.
"""

import copy

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.nodes import MapEntry, NestedSDFG
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.dataflow import MapCollapse
from dace.transformation.interstate import InlineSDFG
from dace.transformation.interstate.move_if_into_map import MoveIfIntoMap
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = 8
M = 8


def _apply(sdfg):
    return PatternMatchAndApplyRepeated([MoveIfIntoMap()]).apply_pass(sdfg, {})


def test_a_guard_over_a_map_nest_moves_under_the_innermost_map():
    """``if c: for j: for k:`` -- the guard is pushed into ``j``, then on into ``k``."""

    @dace.program
    def kern_a_guard_over_a_map_nest_moves_under_the_innermost_map(b: dace.float64[N, N], flag: dace.int64):
        for t in dace.map[0:1]:
            if flag > 0:
                for j in dace.map[0:N]:
                    for k in dace.map[0:N]:
                        b[j, k] = 1.0

    sdfg = kern_a_guard_over_a_map_nest_moves_under_the_innermost_map.to_sdfg(simplify=True)
    assert _apply(sdfg) is not None
    sdfg.validate()

    innermost = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MapEntry) and n.map.params == ["k"])
    guarded = [b.sdfg for b in sdfg.all_control_flow_blocks(recursive=True) if isinstance(b, ConditionalBlock)]
    assert len(guarded) == 1 and guarded[0].parent_nsdfg_node is not None
    assert guarded[0].parent.entry_node(guarded[0].parent_nsdfg_node) is innermost

    for flag in (0, 1):
        out = np.zeros((N, N))
        sdfg(b=out, flag=flag)
        assert np.array_equal(out, np.full((N, N), float(flag)))


def test_data_dependent_guard_does_not_produce_an_invalid_sdfg():
    """A guard whose definition reads data (``a[i] > thr``) moves inside with that data piped in."""

    @dace.program
    def kern_data_dependent_guard_does_not_produce_an_invalid_sdfg(
        a: dace.float64[N], b: dace.float64[N, N], thr: dace.float64
    ):
        for i in dace.map[0:N]:
            if a[i] > thr:
                for j in dace.map[0:N]:
                    b[i, j] = 1.0

    sdfg = kern_data_dependent_guard_does_not_produce_an_invalid_sdfg.to_sdfg(simplify=True)
    assert _apply(sdfg) is not None
    sdfg.validate()
    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks()), "guard left outside"

    a = np.array([1.0, -1.0] * (N // 2))
    out = np.zeros((N, N))
    sdfg(a=a, b=out, thr=0.0)
    assert np.allclose(out, np.where((a > 0.0)[:, None], 1.0, 0.0), rtol=1e-9, atol=1e-9, equal_nan=True)


def _two_predecessor_guard():
    """``S1 -[k=0]-> cond <-[k=1]- S2`` with guard ``k == 1``."""
    sdfg = dace.SDFG("twodefs")
    sdfg.add_symbol("k", dace.int64)
    sdfg.add_symbol("sel", dace.int64)
    sdfg.add_array("b", [N], dace.float64)

    outer = dace.SDFG("outer_two_predecessor_guard")
    outer.add_symbol("k", dace.int64)
    outer.add_array("bo", [N], dace.float64)
    outer.add_symbol("sel", dace.int64)
    entry = outer.add_state("entry", is_start_block=True)
    s1 = outer.add_state("s1")
    s2 = outer.add_state("s2")
    cb = ConditionalBlock("guard")
    outer.add_node(cb)
    # Diamond: both predecessors define the guard symbol, differently.
    outer.add_edge(entry, s1, InterstateEdge(condition=CodeBlock("sel > 0")))
    outer.add_edge(entry, s2, InterstateEdge(condition=CodeBlock("sel <= 0")))
    outer.add_edge(s1, cb, InterstateEdge(assignments={"k": "0"}))
    outer.add_edge(s2, cb, InterstateEdge(assignments={"k": "1"}))
    region = ControlFlowRegion("then", sdfg=outer)
    bs = region.add_state("w", is_start_block=True)
    me, mx = bs.add_map("inner", dict(j="0:%d" % N))
    t = bs.add_tasklet("one", {}, {"o"}, "o = 1.0")
    bs.add_edge(me, None, t, None, dace.Memlet())
    bs.add_memlet_path(t, mx, bs.add_access("bo"), src_conn="o", memlet=dace.Memlet("bo[j]"))
    cb.add_branch(CodeBlock("k == 1"), region)

    st = sdfg.add_state("main", is_start_block=True)
    ome, omx = st.add_map("outer_map", dict(i="0:1"))
    ns = st.add_nested_sdfg(outer, {}, {"bo"}, symbol_mapping={"k": "k", "sel": "sel"})
    st.add_edge(ome, None, ns, None, dace.Memlet())
    st.add_memlet_path(ns, omx, st.add_access("b"), src_conn="bo", memlet=dace.Memlet("b[0:%d]" % N))
    return sdfg


def test_guard_symbol_defined_on_two_edges_is_refused():
    """Two predecessors defining the guard symbol differently.

    ``apply`` collects definitions into a flat dict (last one wins) yet deletes
    every edge's copy, so both paths would evaluate the guard with whichever
    definition iteration order landed on.
    """
    sdfg = _two_predecessor_guard()
    assert _apply(sdfg) is None, "ambiguous guard definition must be refused"


def test_guard_symbol_used_by_a_map_range_is_not_deleted():
    """The moved guard symbol must survive if dataflow still reads it.

    ``remove_symbol`` also strips the name from the parent's symbol_mapping, so
    deleting a symbol that a map range or memlet still mentions leaves it
    undefined at every level.
    """

    n2 = dace.symbol("n2")

    @dace.program
    def kern_guard_symbol_used_by_a_map_range_is_not_deleted(b: dace.float64[N]):
        for i in dace.map[0:1]:
            if n2 > 0:
                for j in dace.map[0:n2]:
                    b[j] = 1.0

    sdfg = kern_guard_symbol_used_by_a_map_range_is_not_deleted.to_sdfg(simplify=True)
    _apply(sdfg)
    sdfg.validate()  # would fail if n2 were removed while the range uses it

    out = np.zeros(N)
    sdfg(b=out, n2=N)
    assert np.allclose(out, np.ones(N), rtol=1e-9, atol=1e-9, equal_nan=True)


def levmask_guarded_rows() -> dace.SDFG:
    """``do jb: if (levmask(jb)) do jc: b(jc, jb) = 2 * a(jc, jb)``, with the guard read by a tasklet into a transient.

    The shape dace-fortran gives a ``MASK``-guarded level loop once LoopToMap has nested every loop body.
    """
    mid = dace.SDFG("levmask_rows_body")
    mid.add_array("a", [M, N], dace.float64)
    mid.add_array("b", [M, N], dace.float64)
    mid.add_array("levmask", [M], dace.int32)
    mid.add_scalar("if_cond", dace.int32, transient=True)
    mid.add_symbol("jb", dace.int64)
    guard = mid.add_state("guard", is_start_block=True)
    read = guard.add_tasklet("read_mask", {"m"}, {"c"}, "c = m")
    guard.add_edge(guard.add_read("levmask"), None, read, "m", dace.Memlet("levmask[jb]"))
    guard.add_edge(read, "c", guard.add_write("if_cond"), None, dace.Memlet("if_cond[0]"))
    cb = ConditionalBlock("if_levmask")
    mid.add_node(cb)
    mid.add_edge(guard, cb, InterstateEdge())
    region = ControlFlowRegion("then", sdfg=mid)
    rows = region.add_state("rows", is_start_block=True)
    rows.add_mapped_tasklet(
        "double",
        {"jc": f"0:{N}"},
        {"x": dace.Memlet("a[jb, jc]")},
        "y = 2 * x",
        {"y": dace.Memlet("b[jb, jc]")},
        external_edges=True,
    )
    cb.add_branch(CodeBlock("if_cond != 0"), region)

    sdfg = dace.SDFG("levmask_guarded_rows")
    sdfg.add_array("a", [M, N], dace.float64)
    sdfg.add_array("b", [M, N], dace.float64)
    sdfg.add_array("levmask", [M], dace.int32)
    st = sdfg.add_state("main", is_start_block=True)
    me, mx = st.add_map("levels", {"jb": f"0:{M}"})
    ns = st.add_nested_sdfg(mid, {"a": None, "levmask": None}, {"b": None}, symbol_mapping={"jb": "jb"})
    for name in ("a", "levmask"):
        st.add_memlet_path(
            st.add_read(name), me, ns, dst_conn=name, memlet=dace.Memlet.from_array(name, sdfg.arrays[name])
        )
    st.add_memlet_path(ns, mx, st.add_write("b"), src_conn="b", memlet=dace.Memlet.from_array("b", sdfg.arrays["b"]))
    return sdfg


def test_a_guard_computed_by_a_state_moves_in_and_the_levels_collapse_into_one_map():
    sdfg = levmask_guarded_rows()
    sdfg.validate()
    reference = copy.deepcopy(sdfg)
    assert sdfg.apply_transformations_repeated(MoveIfIntoMap) == 1
    sdfg.validate()
    body = next(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, NestedSDFG))
    assert "if_cond" not in body.arrays, "the guard transient stays in the outer body"
    assert not any(isinstance(b, ConditionalBlock) for b in body.nodes())

    sdfg.apply_transformations_repeated(InlineSDFG)
    assert sdfg.apply_transformations_repeated(MapCollapse) == 1
    sdfg.validate()
    maps = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MapEntry)]
    assert [m.map.params for m in maps] == [["jb", "jc"]]

    rng = np.random.default_rng(7)
    a = rng.random((M, N))
    levmask = np.array([1, 0, 1, 1, 0, 0, 1, 0][:M], dtype=np.int32)
    want, got = np.full((M, N), -1.0), np.full((M, N), -1.0)
    reference(a=a, b=want, levmask=levmask)
    sdfg(a=a, b=got, levmask=levmask)
    assert np.array_equal(got, want)
    assert np.array_equal(got, np.where(levmask[:, None] != 0, 2 * a, -1.0))


if __name__ == "__main__":
    test_a_guard_over_a_map_nest_moves_under_the_innermost_map()
    test_data_dependent_guard_does_not_produce_an_invalid_sdfg()
    test_guard_symbol_defined_on_two_edges_is_refused()
    test_guard_symbol_used_by_a_map_range_is_not_deleted()
    test_a_guard_computed_by_a_state_moves_in_and_the_levels_collapse_into_one_map()
