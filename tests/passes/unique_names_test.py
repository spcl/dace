# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests of ``UniqueNames``: after it, no binding shadows another in one Python namespace."""

import ast

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.unique_names import UniqueNames

N = dace.symbol("N")


def map_params(sdfg: dace.SDFG) -> dict[str, list[str]]:
    return {n.map.label: list(n.map.params) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)}


def nested_maps_sdfg() -> dace.SDFG:
    """``A[i, j] = i * 10 + j`` where both maps call their parameter ``i``; the inner one reads the outer as ``k``."""
    sdfg = dace.SDFG("nested_maps")
    sdfg.add_array("A", [4, 4], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map("outer", {"i": "0:4"})
    k_entry, k_exit = state.add_map("bind_k", {"k": "i:i+1"})
    inner_entry, inner_exit = state.add_map("inner", {"i": "0:4"})
    tasklet = state.add_tasklet("t", {}, {"o": None}, "o = k * 10 + i")
    write = state.add_write("A")
    state.add_nedge(outer_entry, k_entry, dace.Memlet())
    state.add_nedge(k_entry, inner_entry, dace.Memlet())
    state.add_nedge(inner_entry, tasklet, dace.Memlet())
    state.add_memlet_path(tasklet, inner_exit, k_exit, outer_exit, write, src_conn="o", memlet=dace.Memlet("A[k, i]"))
    return sdfg


def test_an_inner_map_parameter_that_shadows_an_outer_one_is_renamed():
    sdfg = nested_maps_sdfg()

    UniqueNames().apply_pass(sdfg, {})

    assert map_params(sdfg) == {"outer": ["i"], "bind_k": ["k"], "inner": ["i_1"]}
    a = np.zeros((4, 4))
    sdfg(A=a)
    assert np.array_equal(a, np.arange(4)[:, None] * 10.0 + np.arange(4)[None, :])


def test_sibling_maps_keep_their_shared_parameter_name():
    @dace.program
    def siblings(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            A[i] = i
        for i in dace.map[0:N]:
            B[i] = 2 * i

    sdfg = siblings.to_sdfg(simplify=True)

    UniqueNames().apply_pass(sdfg, {})

    assert sorted(p for params in map_params(sdfg).values() for p in params) == ["i", "i"]


def test_a_tasklet_local_that_shadows_a_symbol_is_renamed():
    sdfg = dace.SDFG("local_shadows_symbol")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("A", [1], dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("t", {}, {"o": None}, "N = 5\no = N + 1")
    state.add_edge(tasklet, "o", state.add_write("A"), None, dace.Memlet("A[0]"))

    UniqueNames().apply_pass(sdfg, {})

    names = {n.id for n in ast.walk(ast.parse(tasklet.code.as_string)) if isinstance(n, ast.Name)}
    assert names == {"N_1", "o"}
    a = np.zeros(1, dtype=np.int64)
    sdfg(A=a, N=100)
    assert a[0] == 6


def test_a_tasklet_connector_that_spells_a_container_is_renamed_with_its_edge():
    sdfg = dace.SDFG("connector_shadows_container")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("t", {"A": None}, {"B": None}, "B = A + 1")
    state.add_edge(state.add_read("A"), None, tasklet, "A", dace.Memlet("A[0]"))
    state.add_edge(tasklet, "B", state.add_write("B"), None, dace.Memlet("B[0]"))

    UniqueNames().apply_pass(sdfg, {})

    assert set(tasklet.in_connectors) == {"A_1"} and set(tasklet.out_connectors) == {"B_1"}
    assert {e.dst_conn for e in state.in_edges(tasklet)} == {"A_1"}
    b = np.zeros(1)
    sdfg(A=np.array([2.0]), B=b)
    assert b[0] == 3.0


def test_a_nested_container_bound_to_its_parents_namesake_keeps_its_name():
    @dace.program
    def nested(A: dace.float64[4, 4]):
        for i in range(4):
            for j in dace.map[0:4]:
                A[i, j] = A[i, j] + i

    sdfg = nested.to_sdfg(simplify=False)
    inner = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))

    UniqueNames().apply_pass(sdfg, {})

    assert "A" in inner.sdfg.arrays and "A_1" not in inner.sdfg.arrays
    a = np.ones((4, 4))
    sdfg(A=a)
    assert np.array_equal(a, 1.0 + np.arange(4)[:, None] * np.ones((1, 4)))


def swapped_names_sdfg() -> dace.SDFG:
    """``B[:] = 2 * A``, through a nested SDFG whose connector ``A`` is bound to the parent's ``B``."""
    inner = dace.SDFG("doubled")
    inner.add_array("A", [4], dace.float64)
    inner.add_array("X", [4], dace.float64)
    state = inner.add_state()
    state.add_mapped_tasklet(
        "double", {"k": "0:4"}, {"x": dace.Memlet("X[k]")}, "a = 2 * x", {"a": dace.Memlet("A[k]")}, external_edges=True
    )
    outer = dace.SDFG("swapped_names")
    outer.add_array("A", [4], dace.float64)
    outer.add_array("B", [4], dace.float64)
    state = outer.add_state()
    node = state.add_nested_sdfg(inner, {"X"}, {"A"})
    state.add_edge(state.add_read("A"), None, node, "X", dace.Memlet("A[0:4]"))
    state.add_edge(node, "A", state.add_write("B"), None, dace.Memlet("B[0:4]"))
    return outer


def test_a_nested_container_named_like_another_parent_container_is_renamed():
    sdfg = swapped_names_sdfg()
    inner = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))

    UniqueNames().apply_pass(sdfg, {})

    assert set(inner.sdfg.arrays) == {"A_1", "X"} and set(inner.out_connectors) == {"A_1"}
    a, b = np.arange(4.0), np.zeros(4)
    sdfg(A=a, B=b)
    assert np.array_equal(b, 2 * np.arange(4.0))


def test_a_second_application_renames_nothing():
    sdfg = nested_maps_sdfg()
    UniqueNames().apply_pass(sdfg, {})

    renamed = UniqueNames().apply_pass(sdfg, {})

    assert renamed is None


def test_renaming_nothing_leaves_tile_expressions_alone():
    from dace.libraries.tileops import TileIota
    from dace.transformation.passes.unique_names import rename_strings

    sdfg = dace.SDFG("tile_strings")
    sdfg.add_array("t", [8], dace.int64, transient=True, storage=dace.StorageType.Register)
    state = sdfg.add_state()
    node = TileIota("iota", widths=(8,), expr="i + __l0")
    state.add_node(node)
    state.add_edge(node, "_dst", state.add_write("t"), None, dace.Memlet("t[0:8]"))

    rename_strings(sdfg, {})

    assert node.expr == "i + __l0"
