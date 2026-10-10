# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Anchor tests for the :class:`CShift` library node.

The lib node's ``pure`` expansion lowers ``CSHIFT(arr, shift [, dim])``
to a single Map whose source memlet subset rotates the chosen axis
(``FtnModulo(__i + shift, n)``), so the tasklet body is just
``__out = __in``.  These tests exercise many shape / shift / dim
combinations of the construction path, verify the pure expansion's
numerics against ``numpy.roll``, and pin the loud-fail contract when
``shift`` was never set.
"""

import re

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import CShift


def _build(in_shape, dtype, *, dim=1, shift=None):
    """Wire a CShift lib node into a fresh (unexpanded) SDFG with full-array memlets, ``in_shape``
    on both sides. ``shift=None`` means the runtime symbol ``__shift``; an integer or symbolic
    expression pins the value at construct time."""
    shift_tag = "none" if shift is None else re.sub(r"\W", "_", str(shift).replace("-", "m"))
    label = f"cshift_dim{dim}_{'_'.join(map(str, in_shape))}_shift{shift_tag}"
    sdfg = dace.SDFG(label)
    sdfg.add_array("v", list(in_shape), dtype)
    sdfg.add_array("out", list(in_shape), dtype)
    if shift is None and "__shift" not in sdfg.symbols:
        sdfg.add_symbol("__shift", dace.int64)
    state = sdfg.add_state()
    node = CShift("cshift", dim=dim, shift=shift)
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, "_x", dace.Memlet.from_array("v", sdfg.arrays["v"]))
    state.add_edge(node, "_out", state.add_write("out"), None, dace.Memlet.from_array("out", sdfg.arrays["out"]))
    return sdfg


# Construct-and-validate coverage: many shape / dim combinations, each
# wired with a full-dimension memlet on both connectors.


@pytest.mark.parametrize("shift", [2, -1, 0, 1, 4])
def test_cshift_pure_expansion_computes_circular_shift(shift):
    """``CSHIFT(arr, s)`` rotates LEFT by ``s`` (== ``np.roll(arr, -s)``);
    the floored ``FtnModulo`` keeps a negative shift in range."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    sdfg = _build((5,), dace.float64, dim=1, shift=shift)
    sdfg.expand_library_nodes()
    sdfg.validate()
    out = np.zeros(5, dtype=np.float64)
    sdfg(v=arr.copy(), out=out)
    np.testing.assert_allclose(out, np.roll(arr, -shift))


@pytest.mark.parametrize("dim,shift", [(1, 1), (2, 1), (1, -1), (2, 2)])
def test_cshift_pure_expansion_2d_axis(dim, shift):
    """Whole-array rotate along a chosen 2-D axis -- every cross-section
    perpendicular to ``dim`` rotates independently."""
    arr = np.arange(12, dtype=np.float64).reshape((3, 4))
    sdfg = _build((3, 4), dace.float64, dim=dim, shift=shift)
    sdfg.expand_library_nodes()
    sdfg.validate()
    out = np.zeros((3, 4), dtype=np.float64)
    sdfg(v=arr.copy(), out=out)
    np.testing.assert_allclose(out, np.roll(arr, -shift, axis=dim - 1))


if __name__ == "__main__":
    for shift in (2, -1, 0, 1, 4):
        test_cshift_pure_expansion_computes_circular_shift(shift)
    for dim, shift in [(1, 1), (2, 1), (1, -1), (2, 2)]:
        test_cshift_pure_expansion_2d_axis(dim, shift)
