# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MoveIfIntoMap``: ``map i: { guard; if c: map j }`` becomes ``map i: map j: { guard; if c }``, one test per shape
of guard, plus the shapes it must refuse."""

import copy

import numpy as np
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg.nodes import MapEntry, NestedSDFG
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.dataflow import MapCollapse
from dace.transformation.interstate import InlineSDFG, MoveIfIntoMap

N, M = 6, 5


def guarding_maps(sdfg: dace.SDFG) -> list[list[str]]:
    """For every conditional block, the parameters of the map whose body holds it (``[]`` outside any map)."""
    params = []
    for block in sdfg.all_control_flow_blocks(recursive=True):
        if isinstance(block, ConditionalBlock):
            owner = block.sdfg
            entry = owner.parent.entry_node(owner.parent_nsdfg_node) if owner.parent_nsdfg_node is not None else None
            params.append(entry.map.params if entry is not None else [])
    return sorted(params)


@dace.program
def move_if_flag_guard(A: dace.float64[N, M], B: dace.float64[N, M], flag: dace.int32):
    for i in dace.map[0:N]:
        if flag == 1:
            for j in dace.map[0:M]:
                B[i, j] = A[i, j] + 1.0


def test_a_flag_guard_moves_into_the_inner_map():
    sdfg = move_if_flag_guard.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)
    assert sdfg.apply_transformations_repeated(MoveIfIntoMap) == 1
    sdfg.validate()
    assert guarding_maps(sdfg) == [["j"]]

    a = np.random.default_rng(0).random((N, M))
    for flag in (0, 1):
        want, got = np.full((N, M), -1.0), np.full((N, M), -1.0)
        reference(A=a, B=want, flag=flag)
        sdfg(A=a, B=got, flag=flag)
        assert np.array_equal(got, want)
        assert np.array_equal(got, a + 1.0 if flag else np.full((N, M), -1.0))


@dace.program
def move_if_outer_parameter_guard(A: dace.float64[N, M], B: dace.float64[N, M], t: dace.int32):
    for i in dace.map[0:N]:
        if i < t:
            for j in dace.map[0:M]:
                B[i, j] = A[i, j] * 2.0


def test_a_guard_on_the_outer_parameter_is_evaluated_per_inner_iteration():
    sdfg = move_if_outer_parameter_guard.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MoveIfIntoMap) == 1
    sdfg.validate()
    assert guarding_maps(sdfg) == [["j"]]

    a = np.random.default_rng(1).random((N, M))
    got = np.zeros((N, M))
    sdfg(A=a, B=got, t=3)
    assert np.array_equal(got, np.where(np.arange(N)[:, None] < 3, 2.0 * a, 0.0))


@dace.program
def move_if_sibling_maps(A: dace.float64[N, M], B: dace.float64[N, M], C: dace.float64[N, M], flag: dace.int32):
    for i in dace.map[0:N]:
        if flag > 0:
            for j in dace.map[0:M]:
                B[i, j] = A[i, j] + 1.0
            for k in dace.map[0:M]:
                C[i, k] = A[i, k] * 2.0


def test_every_sibling_map_under_the_guard_gets_its_own_copy():
    sdfg = move_if_sibling_maps.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MoveIfIntoMap) == 1
    sdfg.validate()
    assert guarding_maps(sdfg) == [["j"], ["k"]]

    a = np.random.default_rng(2).random((N, M))
    for flag in (0, 1):
        b, c = np.zeros((N, M)), np.zeros((N, M))
        sdfg(A=a, B=b, C=c, flag=flag)
        assert np.array_equal(b, a + 1.0 if flag else np.zeros((N, M)))
        assert np.array_equal(c, a * 2.0 if flag else np.zeros((N, M)))


@dace.program
def move_if_data_guard(a: dace.float64[N], b: dace.float64[N, M], thr: dace.float64):
    for i in dace.map[0:N]:
        if a[i] > thr:
            for j in dace.map[0:M]:
                b[i, j] = 1.0


def test_a_guard_assigned_from_data_moves_in_with_that_data():
    """The frontend assigns ``a_index = a[i]`` on an interstate edge; ``a`` is piped into the inner map."""
    sdfg = move_if_data_guard.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MoveIfIntoMap) == 1
    sdfg.validate()
    assert guarding_maps(sdfg) == [["j"]]

    a = np.array([1.0, -1.0, 2.0, 0.0, 3.0, -2.0])
    got = np.zeros((N, M))
    sdfg(a=a, b=got, thr=0.0)
    assert np.array_equal(got, np.broadcast_to((a > 0.0)[:, None], (N, M)).astype(np.float64))


def levmask_guarded_rows(implicit_entry: bool) -> dace.SDFG:
    """``do jb: if (levmask(jb)) do jc: b(jc, jb) = 2 * a(jc, jb)``, the guard read by a tasklet into a transient:
    the shape dace-fortran gives a mask-guarded level loop once LoopToMap has nested every loop body."""
    mid = dace.SDFG("levmask_rows_body")
    mid.add_array("a", [N, M], dace.float64)
    mid.add_array("b", [N, M], dace.float64)
    mid.add_array("levmask", [N], dace.int32)
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
        {"jc": f"0:{M}"},
        {"x": dace.Memlet("a[jb, jc]")},
        "y = 2 * x",
        {"y": dace.Memlet("b[jb, jc]")},
        external_edges=True,
    )
    cb.add_branch(CodeBlock("if_cond != 0"), region)
    if implicit_entry:  # What canonicalization leaves: one source block, so the entry is derived, not stored
        mid._start_block = None
        mid._cached_start_block = None

    sdfg = dace.SDFG("levmask_guarded_rows")
    for name, desc in mid.arrays.items():
        if not desc.transient:
            sdfg.add_datadesc(name, copy.deepcopy(desc))
    st = sdfg.add_state("main", is_start_block=True)
    me, mx = st.add_map("levels", {"jb": f"0:{N}"})
    ns = st.add_nested_sdfg(mid, {"a": None, "levmask": None}, {"b": None}, symbol_mapping={"jb": "jb"})
    for name in ("a", "levmask"):
        memlet = dace.Memlet.from_array(name, sdfg.arrays[name])
        st.add_memlet_path(st.add_read(name), me, ns, dst_conn=name, memlet=memlet)
    st.add_memlet_path(ns, mx, st.add_write("b"), src_conn="b", memlet=dace.Memlet.from_array("b", sdfg.arrays["b"]))
    return sdfg


@pytest.mark.parametrize("implicit_entry", [False, True])
def test_a_guard_computed_by_a_state_moves_in_and_the_levels_collapse_into_one_map(implicit_entry: bool):
    sdfg = levmask_guarded_rows(implicit_entry)
    sdfg.validate()
    reference = copy.deepcopy(sdfg)
    assert sdfg.apply_transformations_repeated(MoveIfIntoMap) == 1
    sdfg.validate()
    body = next(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, NestedSDFG))
    assert "if_cond" not in body.arrays, "the guard transient stays in the outer body"
    assert guarding_maps(sdfg) == [["jc"]]

    sdfg.apply_transformations_repeated(InlineSDFG)
    assert sdfg.apply_transformations_repeated(MapCollapse) == 1
    sdfg.validate()
    assert [n.map.params for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MapEntry)] == [["jb", "jc"]]

    a = np.random.default_rng(7).random((N, M))
    levmask = np.array([1, 0, 1, 1, 0, 1], dtype=np.int32)
    want, got = np.full((N, M), -1.0), np.full((N, M), -1.0)
    reference(a=a, b=want, levmask=levmask)
    sdfg(a=a, b=got, levmask=levmask)
    assert np.array_equal(got, want)
    assert np.array_equal(got, np.where(levmask[:, None] != 0, 2 * a, -1.0))


@dace.program
def move_if_map_nest(b: dace.float64[N, M], flag: dace.int64):
    for t in dace.map[0:1]:
        if flag > 0:
            for j in dace.map[0:N]:
                for k in dace.map[0:M]:
                    b[j, k] = 1.0


def test_a_guard_over_a_map_nest_moves_under_the_innermost_map():
    sdfg = move_if_map_nest.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MoveIfIntoMap) == 2
    sdfg.validate()
    assert guarding_maps(sdfg) == [["k"]]

    for flag in (0, 1):
        out = np.zeros((N, M))
        sdfg(b=out, flag=flag)
        assert np.array_equal(out, np.full((N, M), float(flag)))


n2 = dace.symbol("n2")


@dace.program
def move_if_symbol_guard_and_range(b: dace.float64[N]):
    for i in dace.map[0:1]:
        if n2 > 0:
            for j in dace.map[0:n2]:
                b[j] = 1.0


def test_a_guard_symbol_the_inner_range_uses_stays_declared():
    sdfg = move_if_symbol_guard_and_range.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MoveIfIntoMap) == 1
    sdfg.validate()

    out = np.zeros(N)
    sdfg(b=out, n2=4)
    assert np.array_equal(out, np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]))


@dace.program
def move_if_else_branch(A: dace.float64[N, M], flag: dace.int32):
    for i in dace.map[0:N]:
        if flag == 1:
            for j in dace.map[0:M]:
                A[i, j] = A[i, j] + 1.0
        else:
            for j in dace.map[0:M]:
                A[i, j] = A[i, j] - 1.0


@dace.program
def move_if_no_inner_map(A: dace.float64[N], flag: dace.int32):
    for i in dace.map[0:N]:
        if flag == 1:
            A[i] = A[i] + 1.0


def two_predecessor_guard() -> dace.SDFG:
    """``s1 -[k=0]-> guard <-[k=1]- s2``: no single definition of the guard symbol to move inside."""
    outer = dace.SDFG("outer_two_predecessor_guard")
    outer.add_symbol("k", dace.int64)
    outer.add_symbol("sel", dace.int64)
    outer.add_array("bo", [N], dace.float64)
    entry = outer.add_state("entry", is_start_block=True)
    s1 = outer.add_state("s1")
    s2 = outer.add_state("s2")
    cb = ConditionalBlock("guard")
    outer.add_node(cb)
    outer.add_edge(entry, s1, InterstateEdge(condition=CodeBlock("sel > 0")))
    outer.add_edge(entry, s2, InterstateEdge(condition=CodeBlock("sel <= 0")))
    outer.add_edge(s1, cb, InterstateEdge(assignments={"k": "0"}))
    outer.add_edge(s2, cb, InterstateEdge(assignments={"k": "1"}))
    region = ControlFlowRegion("then", sdfg=outer)
    bs = region.add_state("w", is_start_block=True)
    me, mx = bs.add_map("inner", {"j": f"0:{N}"})
    t = bs.add_tasklet("one", {}, {"o"}, "o = 1.0")
    bs.add_edge(me, None, t, None, dace.Memlet())
    bs.add_memlet_path(t, mx, bs.add_access("bo"), src_conn="o", memlet=dace.Memlet("bo[j]"))
    cb.add_branch(CodeBlock("k == 1"), region)

    sdfg = dace.SDFG("two_predecessor_guard")
    sdfg.add_symbol("sel", dace.int64)
    sdfg.add_array("b", [N], dace.float64)
    st = sdfg.add_state("main", is_start_block=True)
    ome, omx = st.add_map("outer_map", {"i": "0:1"})
    ns = st.add_nested_sdfg(outer, {}, {"bo": None}, symbol_mapping={"sel": "sel"})
    st.add_edge(ome, None, ns, None, dace.Memlet())
    st.add_memlet_path(ns, omx, st.add_access("b"), src_conn="bo", memlet=dace.Memlet(f"b[0:{N}]"))
    return sdfg


@pytest.mark.parametrize(
    "build",
    [
        lambda: move_if_else_branch.to_sdfg(simplify=False),
        lambda: move_if_no_inner_map.to_sdfg(simplify=False),
        two_predecessor_guard,
    ],
    ids=["else_branch", "no_inner_map", "two_predecessors"],
)
def test_a_refused_shape_leaves_the_graph_untouched(build):
    sdfg = build()
    before = sdfg.to_json()
    assert sdfg.apply_transformations_repeated(MoveIfIntoMap) == 0
    assert sdfg.to_json() == before


if __name__ == "__main__":
    test_a_flag_guard_moves_into_the_inner_map()
    test_a_guard_on_the_outer_parameter_is_evaluated_per_inner_iteration()
    test_every_sibling_map_under_the_guard_gets_its_own_copy()
    test_a_guard_assigned_from_data_moves_in_with_that_data()
    test_a_guard_computed_by_a_state_moves_in_and_the_levels_collapse_into_one_map(False)
    test_a_guard_computed_by_a_state_moves_in_and_the_levels_collapse_into_one_map(True)
    test_a_guard_over_a_map_nest_moves_under_the_innermost_map()
    test_a_guard_symbol_the_inner_range_uses_stays_declared()
    test_a_refused_shape_leaves_the_graph_untouched(lambda: move_if_else_branch.to_sdfg(simplify=False))
    test_a_refused_shape_leaves_the_graph_untouched(lambda: move_if_no_inner_map.to_sdfg(simplify=False))
    test_a_refused_shape_leaves_the_graph_untouched(two_predecessor_guard)
