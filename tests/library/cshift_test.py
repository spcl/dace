# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Anchor tests for the :class:`CShift` library node.

The lib node's ``pure`` expansion lowers ``CSHIFT(arr, shift [, dim])``
to a single Map whose source memlet subset rotates the chosen axis
(``fortran_mod(__i + shift, n)``), so the tasklet body is just
``__out = __in``.  These tests exercise many shape / shift / dim
combinations of the construction path, verify the pure expansion's
numerics against ``numpy.roll``, and pin the loud-fail contract when
``shift`` was never set.
"""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import CShift
from dace.libraries.standard.nodes.cshift import ShiftDirection


def _build(in_shape, dtype, *, dim=1, shift=None, direction=ShiftDirection.FORTRAN):
    """Wire a CShift lib node into a fresh SDFG with full-array memlets.

    :param in_shape: source array shape (also the output shape).
    :param dtype: element type.
    :param dim: Fortran 1-based rotation axis.
    :param shift: ``None`` means the runtime symbol ``__shift``; an
        integer or symbolic expression pins the value at construct time.
    :param direction: FORTRAN rotates ``CSHIFT``'s way, NUMPY rotates
        ``numpy.roll``'s way (the opposite sign).
    :returns: the constructed (unexpanded) SDFG.
    """
    label = f"cshift_dim{dim}_{'_'.join(map(str, in_shape))}"
    sdfg = dace.SDFG(label)
    sdfg.add_array("v", list(in_shape), dtype)
    sdfg.add_array("out", list(in_shape), dtype)
    if shift is None and "__shift" not in sdfg.symbols:
        sdfg.add_symbol("__shift", dace.int64)
    state = sdfg.add_state()
    node = CShift("cshift", dim=dim, shift=shift, direction=direction)
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


def test_cshift_validate_accepts_reparsed_symbol_instances():
    """A same-named dim reaching CShift.validate as two distinct sympy instances (one
    array's dace.int32 shape vs another's dace.int64 shape) must compare equal by
    name -- and a genuine shape mismatch must still be rejected."""
    N32 = dace.symbol("N", dace.int32)
    N64 = dace.symbol("N", dace.int64)
    sdfg = dace.SDFG("cshift_validate_symbol_identity")
    sdfg.add_array("v", [N32], dace.float64)
    sdfg.add_array("out", [N64], dace.float64)
    state = sdfg.add_state()
    node = CShift("cshift", dim=1)
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, "_x", dace.Memlet.from_array("v", sdfg.arrays["v"]))
    state.add_edge(node, "_out", state.add_write("out"), None, dace.Memlet.from_array("out", sdfg.arrays["out"]))
    node.validate(sdfg, state)  # must not raise

    sdfg.arrays["out"].shape = (dace.symbol("P", dace.int32), )
    out_edge = next(e for e in state.out_edges(node) if e.src_conn == "_out")
    out_edge.data = dace.Memlet.from_array("out", sdfg.arrays["out"])
    with pytest.raises(ValueError, match="input shape .* != output shape"):
        node.validate(sdfg, state)


def test_cshift_pure_expansion_requires_shift():
    """The pure expansion fails loud when ``shift`` was never set -- a
    fabricated fallback symbol would leak as an unbound free symbol the
    SDFG would then demand as a call argument."""
    sdfg = _build((5, ), dace.float64, dim=1)  # shift=None
    with pytest.raises(ValueError, match="shift is None"):
        sdfg.expand_library_nodes()


@pytest.mark.parametrize("shift", [2, -1, 0, 1, 4])
def test_cshift_pure_expansion_computes_circular_shift(shift):
    """``CSHIFT(arr, s)`` rotates LEFT by ``s`` (== ``np.roll(arr, -s)``);
    the floored ``fortran_mod`` keeps a negative shift in range."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    sdfg = _build((5, ), dace.float64, dim=1, shift=shift)
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


@pytest.mark.parametrize("shift", [2, -1, 0, 1, 4])
def test_cshift_numpy_direction_rotates_the_other_way(shift):
    """The same node with ``direction=NUMPY`` is ``numpy.roll``, not ``CSHIFT``. Both sit on one
    expansion, so a regression that drops the sign shows up as one of these two matching the wrong
    reference -- and at ``shift=0``, where the two agree, as neither of them failing."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    sdfg = _build((5, ), dace.float64, dim=1, shift=shift, direction=ShiftDirection.NUMPY)
    sdfg.expand_library_nodes()
    sdfg.validate()
    out = np.zeros(5, dtype=np.float64)
    sdfg(v=arr.copy(), out=out)
    np.testing.assert_allclose(out, np.roll(arr, shift))


@pytest.mark.parametrize("dim,shift", [(1, 1), (2, 1), (1, -1), (2, 2)])
def test_cshift_numpy_direction_2d_axis(dim, shift):
    arr = np.arange(12, dtype=np.float64).reshape((3, 4))
    sdfg = _build((3, 4), dace.float64, dim=dim, shift=shift, direction=ShiftDirection.NUMPY)
    sdfg.expand_library_nodes()
    sdfg.validate()
    out = np.zeros((3, 4), dtype=np.float64)
    sdfg(v=arr.copy(), out=out)
    np.testing.assert_allclose(out, np.roll(arr, shift, axis=dim - 1))


def test_cshift_defaults_to_the_fortran_direction():
    """The node is Fortran's ``CSHIFT`` first; the HLFIR bridge builds it without naming a
    direction and must keep getting that rotation."""
    assert CShift("cshift", dim=1, shift=2).direction is ShiftDirection.FORTRAN
