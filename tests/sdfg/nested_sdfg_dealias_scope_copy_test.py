# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Widening a window connector moves a copy that reads it inside a scope, whose memlet names the copy's target."""

import numpy as np

import dace
from dace.sdfg import SDFG, dealias, nodes

N, START, LEN = 16, 4, 8


def window_read_by_a_scope_copy() -> SDFG:
    """``b[k] = w[k]`` for a window ``w = A[4:12]``, the read a copy from ``w`` into a scalar inside the map; the
    legacy window shape, which validation refuses until the connector is widened."""
    inner = SDFG("scope_copy_inner")
    inner.add_array("w", [LEN], dace.float64)
    inner.add_array("b", [LEN], dace.float64)
    inner.add_scalar("s", dace.float64, transient=True)
    state = inner.add_state()
    entry, exit_ = state.add_map("m", {"k": f"0:{LEN}"})
    scalar = state.add_access("s")
    entry.add_in_connector("IN_w")
    entry.add_out_connector("OUT_w")
    state.add_edge(state.add_read("w"), None, entry, "IN_w", dace.Memlet(f"w[0:{LEN}]"))
    state.add_edge(entry, "OUT_w", scalar, None, dace.Memlet(data="s", subset="0", other_subset="k"))
    tasklet = state.add_tasklet("t", {"a"}, {"o"}, "o = a")
    state.add_edge(scalar, None, tasklet, "a", dace.Memlet("s[0]"))
    state.add_memlet_path(tasklet, exit_, state.add_write("b"), src_conn="o", memlet=dace.Memlet("b[k]"))

    top = SDFG("scope_copy_window")
    top.add_array("A", [N], dace.float64)
    top.add_array("B", [LEN], dace.float64)
    outer = top.add_state()
    node = outer.add_nested_sdfg(inner, {"w"}, {"b"})
    outer.add_edge(outer.add_read("A"), None, node, "w", dace.Memlet(f"A[{START}:{START + LEN}]"))
    outer.add_edge(node, "b", outer.add_write("B"), None, dace.Memlet(f"B[0:{LEN}]"))
    return top


def test_a_scope_copy_from_a_widened_window_reads_the_same_elements():
    sdfg = window_read_by_a_scope_copy()

    dealias.convert_legacy_nested_sdfgs(sdfg)

    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    assert tuple(node.sdfg.arrays["w"].shape) == (N,)
    A, B = np.arange(N, dtype=np.float64), np.zeros(LEN)
    sdfg(A=A, B=B)
    assert np.array_equal(B, A[START : START + LEN])


if __name__ == "__main__":
    test_a_scope_copy_from_a_widened_window_reads_the_same_elements()
