# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Widening a window connector moves both sides of a copy within the window."""

import numpy as np

import dace
from dace.sdfg import SDFG, dealias, nodes

N, START, LEN = 16, 4, 8
HALF = LEN // 2


def window_copied_within_itself() -> SDFG:
    """``w[HALF:LEN] = w[0:HALF]`` for a window ``w = A[4:12]``, the legacy window shape."""
    inner = SDFG("self_copy_inner")
    inner.add_array("w", [LEN], dace.float64)
    state = inner.add_state()
    state.add_edge(
        state.add_read("w"),
        None,
        state.add_write("w"),
        None,
        dace.Memlet(data="w", subset=f"0:{HALF}", other_subset=f"{HALF}:{LEN}"),
    )

    top = SDFG("self_copy_window")
    top.add_array("A", [N], dace.float64)
    outer = top.add_state()
    node = outer.add_nested_sdfg(inner, {"w"}, {"w"})
    outer.add_edge(outer.add_read("A"), None, node, "w", dace.Memlet(f"A[{START}:{START + LEN}]"))
    outer.add_edge(node, "w", outer.add_write("A"), None, dace.Memlet(f"A[{START}:{START + LEN}]"))
    return top


def test_a_copy_within_a_widened_window_moves_both_sides():
    sdfg = window_copied_within_itself()

    dealias.convert_legacy_nested_sdfgs(sdfg)

    node = next(n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    assert tuple(node.sdfg.arrays["w"].shape) == (N,)
    A = np.arange(N, dtype=np.float64)
    want = A.copy()
    want[START + HALF : START + LEN] = A[START : START + HALF]
    sdfg(A=A)
    assert np.array_equal(A, want)


if __name__ == "__main__":
    test_a_copy_within_a_widened_window_moves_both_sides()
