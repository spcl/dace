# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Integrating a nested SDFG whose window-bound input is read after an empty ordering edge."""

import numpy as np

import dace
from dace.sdfg import nodes


def windowed_read_after_ordering_edge() -> dace.SDFG:
    """``b[i] = a[i] + 1`` through a nested SDFG bound to one element of ``A`` (the legacy window form).
    Inside, an unrelated tasklet is ordered before the read of ``a`` by an empty edge into its access node."""
    inner = dace.SDFG("ordered_read")
    inner.add_array("a", [1], dace.float64)
    inner.add_array("b", [1], dace.float64)
    inner.add_scalar("s", dace.float64, transient=True)
    state = inner.add_state()
    first = state.add_tasklet("first", {}, {"o"}, "o = 0")
    state.add_edge(first, "o", state.add_write("s"), None, dace.Memlet("s[0]"))
    read = state.add_access("a")
    state.add_nedge(first, read, dace.Memlet())
    add = state.add_tasklet("add", {"x"}, {"y"}, "y = x + 1")
    state.add_edge(read, None, add, "x", dace.Memlet("a[0]"))
    state.add_edge(add, "y", state.add_write("b"), None, dace.Memlet("b[0]"))

    sdfg = dace.SDFG("integrate_ordering_edge")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_array("B", [4], dace.float64)
    outer = sdfg.add_state()
    me, mx = outer.add_map("rows", {"i": "0:4"})
    nsdfg = outer.add_nested_sdfg(inner, {"a"}, {"b"})
    outer.add_memlet_path(outer.add_read("A"), me, nsdfg, dst_conn="a", memlet=dace.Memlet("A[i]"))
    outer.add_memlet_path(nsdfg, mx, outer.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    return sdfg


def test_an_ordering_edge_into_a_read_window_does_not_write_it_back():
    """The empty edge orders the read; taken as a write it adds a copy back into the input-only ``A``."""
    sdfg = windowed_read_after_ordering_edge()
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    node.integrate_into_parent()
    sdfg.validate()
    assert "A" not in node.out_connectors
    A = np.arange(4, dtype=np.float64)
    B = np.zeros(4)
    sdfg(A=A, B=B)
    assert np.array_equal(B, A + 1)


if __name__ == "__main__":
    test_an_ordering_edge_into_a_read_window_does_not_write_it_back()
