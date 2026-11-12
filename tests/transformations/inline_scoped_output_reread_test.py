# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""InlineSDFG inlines a nested SDFG in a map whose output is read again inside the nested SDFG after it is written."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.interstate.sdfg_nesting import InlineSDFG

N = 8


def scoped_reread(name: str, rewrite: bool) -> dace.SDFG:
    """``map i: b = 2 * a; c = b + 1; b = 3 * b`` with ``b`` an output read again between its writes; without
    ``rewrite`` the read-back node is the only one of ``b``, so no sink carries the output."""
    sdfg = dace.SDFG(name)
    for data in "ABC":
        sdfg.add_array(data, [N], dace.float64)
    inner = dace.SDFG(f"{name}_inner")
    inner.add_symbol("i", dace.int64)
    for data in "abc":
        inner.add_array(data, [N], dace.float64)
    nstate = inner.add_state()
    double = nstate.add_tasklet("double", {"x"}, {"y"}, "y = 2 * x")
    inc = nstate.add_tasklet("inc", {"x"}, {"y"}, "y = x + 1")
    mid = nstate.add_access("b")
    nstate.add_edge(nstate.add_read("a"), None, double, "x", dace.Memlet("a[i]"))
    nstate.add_edge(double, "y", mid, None, dace.Memlet("b[i]"))
    nstate.add_edge(mid, None, inc, "x", dace.Memlet("b[i]"))
    nstate.add_edge(inc, "y", nstate.add_write("c"), None, dace.Memlet("c[i]"))
    if rewrite:
        triple = nstate.add_tasklet("triple", {"x"}, {"y"}, "y = 3 * x")
        nstate.add_edge(mid, None, triple, "x", dace.Memlet("b[i]"))
        nstate.add_edge(triple, "y", nstate.add_write("b"), None, dace.Memlet("b[i]"))

    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", {"i": f"0:{N}"})
    node = state.add_nested_sdfg(inner, {"a": None}, {"b": None, "c": None}, symbol_mapping={"i": "i"})
    state.add_memlet_path(state.add_read("A"), entry, node, dst_conn="a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(node, exit_, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    state.add_memlet_path(node, exit_, state.add_write("C"), src_conn="c", memlet=dace.Memlet("C[i]"))
    sdfg.validate()
    return sdfg


def nested_node(sdfg: dace.SDFG) -> nodes.NestedSDFG:
    return next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))


def test_inline_scoped_output_read_after_write():
    sdfg = scoped_reread("inline_scoped_output_reread", rewrite=True)
    InlineSDFG.apply_to(sdfg, nested_sdfg=nested_node(sdfg))
    sdfg.validate()
    assert not any(isinstance(n, nodes.NestedSDFG) for n, _ in sdfg.all_nodes_recursive())
    a = np.random.rand(N)
    b, c = np.zeros(N), np.zeros(N)
    sdfg(A=a, B=b, C=c)
    assert np.allclose(b, 6 * a)
    assert np.allclose(c, 2 * a + 1)


def test_inline_scoped_output_without_sink_refused():
    sdfg = scoped_reread("inline_scoped_output_without_sink", rewrite=False)
    assert not InlineSDFG.can_be_applied_to(sdfg, nested_sdfg=nested_node(sdfg))


if __name__ == "__main__":
    test_inline_scoped_output_read_after_write()
    test_inline_scoped_output_without_sink_refused()
