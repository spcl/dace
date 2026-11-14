# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""InlineSDFG inlines a nested SDFG in a map whose in/out connector (one outer container, one window) is also accessed
inside a map of the nested SDFG: the outer paths reattach to the access nodes outside that map."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.interstate.sdfg_nesting import InlineSDFG

N = 8


def shared_in_inner_map(name: str, top_level_access: bool) -> dace.SDFG:
    """``map i: map j: A[i, j] = 2 * A[i, j]; C[i, j] = A[i, j] + 1`` with ``A`` one in/out connector whose access node
    in the ``j`` map is written and read; without ``top_level_access`` ``A`` is never accessed outside that map."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [N, 4], dace.float64)
    sdfg.add_array("C", [N, 4], dace.float64)
    inner = dace.SDFG(f"{name}_inner")
    inner.add_symbol("i", dace.int64)
    inner.add_array("A", [N, 4], dace.float64)
    inner.add_array("C", [N, 4], dace.float64)
    nstate = inner.add_state()
    entry, exit_ = nstate.add_map("j_map", {"j": "0:4"})
    double = nstate.add_tasklet(
        "double", {"x"} if top_level_access else {}, {"y"}, "y = 2 * x" if top_level_access else "y = 2 * j"
    )
    copy = nstate.add_tasklet("copy", {"x"}, {"y"}, "y = x + 1")
    mid = nstate.add_access("A")
    if top_level_access:
        nstate.add_memlet_path(nstate.add_read("A"), entry, double, dst_conn="x", memlet=dace.Memlet("A[i, j]"))
    else:
        nstate.add_nedge(entry, double, dace.Memlet())
    nstate.add_edge(double, "y", mid, None, dace.Memlet("A[i, j]"))
    nstate.add_edge(mid, None, copy, "x", dace.Memlet("A[i, j]"))
    nstate.add_memlet_path(copy, exit_, nstate.add_write("C"), src_conn="y", memlet=dace.Memlet("C[i, j]"))
    if top_level_access:
        nstate.add_memlet_path(mid, exit_, nstate.add_write("A"), memlet=dace.Memlet("A[i, j]"))

    state = sdfg.add_state()
    oentry, oexit = state.add_map("i_map", {"i": f"0:{N}"})
    node = state.add_nested_sdfg(inner, {"A": None}, {"A": None, "C": None}, symbol_mapping={"i": "i"})
    state.add_memlet_path(state.add_read("A"), oentry, node, dst_conn="A", memlet=dace.Memlet("A[i, 0:4]"))
    state.add_memlet_path(node, oexit, state.add_write("A"), src_conn="A", memlet=dace.Memlet("A[i, 0:4]"))
    state.add_memlet_path(node, oexit, state.add_write("C"), src_conn="C", memlet=dace.Memlet("C[i, 0:4]"))
    sdfg.validate()
    return sdfg


def test_inline_shared_inout_accessed_in_inner_map():
    sdfg = shared_in_inner_map("inline_shared_inout_inner_map", top_level_access=True)
    reference = shared_in_inner_map("inline_shared_inout_inner_map_ref", top_level_access=True)
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    InlineSDFG.apply_to(sdfg, nested_sdfg=node)
    sdfg.validate()
    assert not any(isinstance(n, nodes.NestedSDFG) for n, _ in sdfg.all_nodes_recursive())
    a = np.random.rand(N, 4)
    a_ref, c, c_ref = a.copy(), np.zeros((N, 4)), np.zeros((N, 4))
    sdfg(A=a, C=c)
    reference(A=a_ref, C=c_ref)
    assert np.allclose(a, a_ref) and np.allclose(c, c_ref)


def test_inline_shared_inout_only_in_inner_map_refused():
    sdfg = shared_in_inner_map("inline_shared_inout_only_inner_map", top_level_access=False)
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    assert not InlineSDFG.can_be_applied_to(sdfg, nested_sdfg=node)


if __name__ == "__main__":
    test_inline_shared_inout_accessed_in_inner_map()
    test_inline_shared_inout_only_in_inner_map_refused()
