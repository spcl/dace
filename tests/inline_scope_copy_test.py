# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map whose body is one copy, ``MapEntry -> A -> copy -> MapExit``: a nested SDFG inside a map whose input is
copied straight into its output inlines to it, and it compiles and nests back."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.interstate import InlineSDFG

N = 8


def reversed_copy_in_a_map(memlet_names_destination: bool) -> dace.SDFG:
    """``for i in map[0:N]: B[i] = A[N - 1 - i]`` as a nested copy ``a -> b``, its memlet naming ``b`` or ``a``."""
    sdfg = dace.SDFG("reversed_copy_in_a_map")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    state = sdfg.add_state()
    inner = dace.SDFG("copy_body")
    inner.add_array("a", [N], dace.float64)
    inner.add_array("b", [N], dace.float64)
    inner.add_symbol("i", dace.int64)
    body = inner.add_state()
    if memlet_names_destination:
        memlet = dace.Memlet(data="b", subset="i", other_subset=f"{N - 1} - i")
    else:
        memlet = dace.Memlet(data="a", subset=f"{N - 1} - i", other_subset="i")
    body.add_edge(body.add_read("a"), None, body.add_write("b"), None, memlet)
    nsdfg = state.add_nested_sdfg(inner, {"a"}, {"b"}, {"i": "i"})
    entry, exit_ = state.add_map("rows", {"i": f"0:{N}"})
    state.add_memlet_path(state.add_read("A"), entry, nsdfg, dst_conn="a", memlet=dace.Memlet(f"A[0:{N}]"))
    state.add_memlet_path(nsdfg, exit_, state.add_write("B"), src_conn="b", memlet=dace.Memlet(f"B[0:{N}]"))
    return sdfg


def reverses(sdfg: dace.SDFG) -> bool:
    a = np.arange(N, dtype=np.float64)
    b = np.zeros(N)
    sdfg(A=a, B=b)
    return np.array_equal(b, a[::-1])


def test_a_copy_between_two_connectors_inside_a_map_inlines_and_compiles():
    for names_destination in (True, False):
        sdfg = reversed_copy_in_a_map(names_destination)
        assert sdfg.apply_transformations(InlineSDFG) == 1
        sdfg.validate()
        assert not any(isinstance(n, nodes.NestedSDFG) for n, _ in sdfg.all_nodes_recursive())
        assert reverses(sdfg)


def test_a_map_body_that_is_one_copy_nests_back():
    sdfg = reversed_copy_in_a_map(memlet_names_destination=True)
    sdfg.apply_transformations(InlineSDFG)
    state = sdfg.start_state
    entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    body = state.scope_subgraph(entry, include_entry=False, include_exit=False)
    nest_state_subgraph(sdfg, state, body)
    sdfg.validate()
    assert reverses(sdfg)


if __name__ == "__main__":
    test_a_copy_between_two_connectors_inside_a_map_inlines_and_compiles()
    test_a_map_body_that_is_one_copy_nests_back()
