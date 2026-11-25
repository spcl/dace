# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Widening a window connector keeps a reduction over it folding the dimensions it folded."""

import numpy as np

import dace
from dace.sdfg import SDFG, dealias

ROWS, WIDTH, ROW = 4, 8, 2


def row_summed_through_a_window() -> SDFG:
    """``out[0] = sum(w)`` for a window ``w = A[2, 0:8]`` of a ``4 x 8`` array, the legacy window shape."""
    inner = SDFG("reduce_inner")
    inner.add_array("w", [WIDTH], dace.float64)
    inner.add_array("out", [1], dace.float64)
    state = inner.add_state()
    reduction = state.add_reduce("lambda x, y: x + y", [0], 0)
    state.add_edge(state.add_read("w"), None, reduction, None, dace.Memlet(f"w[0:{WIDTH}]"))
    state.add_edge(reduction, None, state.add_write("out"), None, dace.Memlet("out[0]"))

    top = SDFG("reduce_window")
    top.add_array("A", [ROWS, WIDTH], dace.float64)
    top.add_array("out", [1], dace.float64)
    outer = top.add_state()
    node = outer.add_nested_sdfg(inner, {"w"}, {"out"})
    outer.add_edge(outer.add_read("A"), None, node, "w", dace.Memlet(f"A[{ROW}, 0:{WIDTH}]"))
    outer.add_edge(node, "out", outer.add_write("out"), None, dace.Memlet("out[0]"))
    return top


def test_a_reduction_over_a_widened_window_folds_the_dimension_it_folded():
    sdfg = row_summed_through_a_window()

    dealias.convert_legacy_nested_sdfgs(sdfg)

    reduction = next(n for n, parent in sdfg.all_nodes_recursive() if type(n).__name__ == "Reduce")
    assert reduction.axes == [1]
    A, out = np.random.default_rng(3).random((ROWS, WIDTH)), np.zeros(1)
    sdfg(A=A, out=out)
    assert np.isclose(out[0], A[ROW].sum())


if __name__ == "__main__":
    test_a_reduction_over_a_widened_window_folds_the_dimension_it_folded()
