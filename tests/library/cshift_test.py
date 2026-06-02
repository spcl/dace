# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Anchor tests for the :class:`CShift` library node.

``CSHIFT`` does not appear in any current target Fortran workload, so
the lib node's pure expansion is a stub that raises
``NotImplementedError``.  These tests exercise many shape / shift /
dim combinations of the construction path -- each wires the lib node
with a full-array memlet, validates, and (for one representative
case) verifies the lowering refusal.  Implementing the expansion
later will produce a clean test signal across all of them.
"""
import pytest

import dace
from dace.libraries.standard.nodes import CShift


def _build(in_shape, dtype, *, dim=1, shift=None):
    """Wire a CShift lib node into a fresh SDFG with full-array memlets.

    :param in_shape: source array shape (also the output shape).
    :param dtype: element type.
    :param dim: Fortran 1-based rotation axis.
    :param shift: ``None`` means the runtime symbol ``__shift``; an
        integer or symbolic expression pins the value at construct time.
    :returns: the constructed (unexpanded) SDFG.
    """
    label = f"cshift_dim{dim}_{'_'.join(map(str, in_shape))}"
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
_SHAPE_DIM_CASES = [
    ((5, ), 1),
    ((1, ), 1),
    ((128, ), 1),
    ((3, 4), 1),
    ((3, 4), 2),
    ((1, 1), 1),
    ((1, 1), 2),
    ((16, 16), 1),
    ((16, 16), 2),
    ((2, 3, 4), 1),
    ((2, 3, 4), 2),
    ((2, 3, 4), 3),
    ((5, 5, 5, 5), 1),
    ((5, 5, 5, 5), 4),
]


@pytest.mark.parametrize("shape,dim", _SHAPE_DIM_CASES)
def test_cshift_construct_validates_runtime_shift(shape, dim):
    """Runtime ``__shift`` symbol -- SDFG validates regardless of rank / axis."""
    sdfg = _build(shape, dace.float64, dim=dim)
    sdfg.validate()


@pytest.mark.parametrize("shape,dim", _SHAPE_DIM_CASES)
def test_cshift_construct_validates_compile_time_shift(shape, dim):
    """Compile-time constant shifts -- lib node accepts the SymbolicProperty value."""
    sdfg = _build(shape, dace.float64, dim=dim, shift=3)
    sdfg.validate()


@pytest.mark.parametrize("shift", [-7, -1, 0, 1, 7, 128])
def test_cshift_construct_handles_various_shift_magnitudes(shift):
    """Negative, zero, and large shifts all construct cleanly."""
    sdfg = _build((16, ), dace.float64, dim=1, shift=shift)
    sdfg.validate()


@pytest.mark.parametrize("dtype", [dace.int32, dace.int64, dace.float32, dace.float64])
def test_cshift_construct_validates_various_dtypes(dtype):
    """Lib node is dtype-agnostic at construct time."""
    sdfg = _build((8, 8), dtype, dim=1)
    sdfg.validate()


def test_cshift_validate_rejects_out_of_range_dim():
    """``dim`` outside ``[1, rank]`` raises."""
    sdfg = _build((4, 4), dace.float64, dim=2)
    sdfg.validate()
    bad = dace.SDFG("cshift_bad_dim")
    bad.add_array("v", [4, 4], dace.float64)
    bad.add_array("out", [4, 4], dace.float64)
    state = bad.add_state()
    node = CShift("cshift", dim=5)  # rank-2 input, dim=5 is out of range
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, "_x", dace.Memlet.from_array("v", bad.arrays["v"]))
    state.add_edge(node, "_out", state.add_write("out"), None, dace.Memlet.from_array("out", bad.arrays["out"]))
    with pytest.raises(ValueError, match="dim=5 out of range"):
        node.validate(bad, state)


def test_cshift_validate_rejects_mismatched_shapes():
    """Input and output must carry the same shape."""
    bad = dace.SDFG("cshift_shape_mismatch")
    bad.add_array("v", [4, 4], dace.float64)
    bad.add_array("out", [4, 5], dace.float64)
    state = bad.add_state()
    node = CShift("cshift", dim=1)
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, "_x", dace.Memlet.from_array("v", bad.arrays["v"]))
    state.add_edge(node, "_out", state.add_write("out"), None, dace.Memlet.from_array("out", bad.arrays["out"]))
    with pytest.raises(ValueError, match="input shape .* != output shape"):
        node.validate(bad, state)


def test_cshift_pure_expansion_raises_until_implemented():
    """The stub expansion refuses lowering with a clear message."""
    sdfg = _build((5, ), dace.float64, dim=1)
    with pytest.raises(NotImplementedError, match="CShift pure expansion is not yet implemented"):
        sdfg.expand_library_nodes()
