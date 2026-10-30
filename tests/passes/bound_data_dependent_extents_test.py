# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A transient sized by a symbol read from data takes the extent that bounds it."""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.bound_data_dependent_extents import BoundDataDependentExtents


def per_row_window(sign: int) -> dace.SDFG:
    """``lo = 0 if A[p] > 0 else 1`` per row ``p``, then ``out[p] = sum(win)`` over ``win[N - lo]``
    (``N + lo`` for ``sign = 1``), the window ``A[lo:N]`` filled by a map."""
    sdfg = dace.SDFG("per_row_window_" + ("up" if sign > 0 else "down"))
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("A", ["N + 1"], dace.float64)
    sdfg.add_array("out", ["N"], dace.float64)
    sdfg.add_array("low", ["N"], dace.int64, transient=True)
    sdfg.add_symbol("lo", dace.int64)
    sdfg.add_array("win", [f"N {'+' if sign > 0 else '-'} lo"], dace.float64, transient=True)
    first = sdfg.add_state("first", is_start_block=True)
    first.add_mapped_tasklet(
        "low",
        {"i": "0:N"},
        {"a": dace.Memlet("A[i]")},
        "o = 0 if a > 0 else 1",
        {"o": dace.Memlet("low[i]")},
        external_edges=True,
    )
    loop = LoopRegion("rows", "p < N", "p", "p = 0", "p = p + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(first, loop, dace.InterstateEdge())
    head = loop.add_state("head", is_start_block=True)
    fill = loop.add_state("fill")
    loop.add_edge(head, fill, dace.InterstateEdge(assignments={"lo": "low[p]"}))
    fill.add_mapped_tasklet(
        "window",
        {"j": "0:N - lo"},
        {"a": dace.Memlet("A[j + lo]")},
        "o = a",
        {"o": dace.Memlet("win[j]")},
        external_edges=True,
    )
    total = loop.add_state_after(fill, "total")
    tasklet = total.add_tasklet("zero", {}, {"o"}, "o = 0")
    total.add_edge(tasklet, "o", total.add_write("out"), None, dace.Memlet("out[p]"))
    acc = loop.add_state_after(total, "acc")
    acc.add_mapped_tasklet(
        "sum",
        {"j": "0:N - lo"},
        {"w": dace.Memlet("win[j]")},
        "o = w",
        {"o": dace.Memlet("out[p]", wcr="lambda x, y: x + y")},
        external_edges=True,
    )
    return sdfg


def test_a_nonnegative_data_symbol_subtracted_from_an_extent_bounds_it_by_zero():
    sdfg = per_row_window(-1)
    A = np.arange(9, dtype=np.float64) - 3.0
    expected = np.zeros(8)
    sdfg(A=A.copy(), out=expected, N=8)

    assert BoundDataDependentExtents().apply_pass(sdfg, {}) == 1
    assert [str(extent) for extent in sdfg.arrays["win"].shape] == ["N"]
    got = np.zeros(8)
    sdfg(A=A.copy(), out=got, N=8)
    assert np.allclose(got, expected)


def test_an_extent_growing_with_the_data_symbol_is_left_alone():
    sdfg = per_row_window(1)
    assert BoundDataDependentExtents().apply_pass(sdfg, {}) is None
    assert [str(extent) for extent in sdfg.arrays["win"].shape] == ["N + lo"]


if __name__ == "__main__":
    pytest.main([__file__])
