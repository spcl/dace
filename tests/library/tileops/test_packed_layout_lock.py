# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the design section 2.3 packed-layout lock.

``TileLoad._src`` and ``TileStore._dst`` must each carry an array whose
stride pattern is either packed C (row-major, no padding) or packed
Fortran (column-major, no padding). Any other layout raises
``NotImplementedError`` at ``validate()`` time -- padded layouts will
land when per-arch codegen supports them.
"""
import pytest

import dace
from dace.libraries.tileops import TileLoad, TileStore
from dace.libraries.tileops._pure_codegen import (_strides_match_packed, validate_packed_layout)
from dace.memlet import Memlet

# ---- _strides_match_packed -----------------------------------------------


def test_packed_c_layout_match_returns_true_for_canonical_strides():
    """``(M, N)`` with strides ``(N, 1)`` is packed C."""
    assert _strides_match_packed(shape=(8, 16), strides=(16, 1), order="C")


def test_packed_c_layout_match_returns_false_for_padded_inner_dim():
    """``(M, N)`` with strides ``(N+4, 1)`` is NOT packed C."""
    assert not _strides_match_packed(shape=(8, 16), strides=(20, 1), order="C")


def test_packed_fortran_layout_match_returns_true_for_canonical_strides():
    """``(M, N)`` with strides ``(1, M)`` is packed Fortran."""
    assert _strides_match_packed(shape=(8, 16), strides=(1, 8), order="F")


def test_packed_fortran_layout_match_returns_false_for_padded():
    """``(M, N)`` with strides ``(1, M+4)`` is NOT packed Fortran."""
    assert not _strides_match_packed(shape=(8, 16), strides=(1, 12), order="F")


def test_packed_match_returns_false_for_length_mismatch():
    """Stride / shape length mismatch is refused."""
    assert not _strides_match_packed(shape=(8, ), strides=(1, 1), order="C")


# ---- validate_packed_layout -----------------------------------------------


def _array_with(shape, strides, dtype=dace.float64):
    sdfg = dace.SDFG("layout_fixture")
    sdfg.add_array("A", shape, dtype, strides=strides, transient=False)
    return sdfg.arrays["A"]


def test_validate_accepts_packed_c_2d():
    desc = _array_with(shape=(8, 16), strides=(16, 1))
    validate_packed_layout("tl", "_src", desc)


def test_validate_accepts_packed_fortran_2d():
    desc = _array_with(shape=(8, 16), strides=(1, 8))
    validate_packed_layout("tl", "_src", desc)


def test_validate_accepts_packed_c_3d():
    desc = _array_with(shape=(4, 8, 16), strides=(128, 16, 1))
    validate_packed_layout("tl", "_src", desc)


def test_validate_refuses_padded_inner_dim():
    desc = _array_with(shape=(8, 16), strides=(20, 1))
    with pytest.raises(NotImplementedError, match=r"non-packed stride pattern"):
        validate_packed_layout("tl", "_src", desc)


def test_validate_refuses_padded_3d():
    desc = _array_with(shape=(4, 8, 16), strides=(192, 24, 1))
    with pytest.raises(NotImplementedError, match=r"non-packed stride pattern"):
        validate_packed_layout("tl", "_src", desc)


def test_validate_accepts_1d_unit_stride():
    desc = _array_with(shape=(16, ), strides=(1, ))
    validate_packed_layout("tl", "_src", desc)


def test_validate_refuses_1d_non_unit_stride():
    desc = _array_with(shape=(16, ), strides=(2, ))
    with pytest.raises(NotImplementedError, match=r"non-unit stride"):
        validate_packed_layout("tl", "_src", desc)


def test_validate_accepts_scalar_descriptor_as_noop():
    """Scalars have no per-dim stride; the validator is a no-op."""
    sdfg = dace.SDFG("scalar_fixture")
    sdfg.add_scalar("S", dace.float64, transient=False)
    validate_packed_layout("tl", "_src", sdfg.arrays["S"])


# ---- end-to-end through TileLoad / TileStore -----------------------------


def test_tileload_refuses_padded_source_at_validate():
    """A wired ``_src`` with padded strides triggers ``NotImplementedError``."""
    sdfg = dace.SDFG("tl_padded")
    sdfg.add_array("Src", (8, 16), dace.float64, strides=(20, 1), transient=False)
    sdfg.add_array("Dst", (4, 8), dace.float64, transient=True)
    state = sdfg.add_state("s")
    src = state.add_access("Src")
    dst = state.add_access("Dst")
    node = TileLoad("tl", widths=(4, 8))
    state.add_node(node)
    state.add_edge(src, None, node, "_src", Memlet("Src[0:8, 0:16]"))
    state.add_edge(node, "_dst", dst, None, Memlet("Dst[0:4, 0:8]"))
    with pytest.raises(NotImplementedError, match=r"non-packed stride pattern"):
        node.validate(sdfg, state)


def test_tilestore_refuses_padded_dest_at_validate():
    """A wired ``_dst`` with padded strides triggers ``NotImplementedError``."""
    sdfg = dace.SDFG("ts_padded")
    sdfg.add_array("Src", (4, 8), dace.float64, transient=True)
    sdfg.add_array("Dst", (8, 16), dace.float64, strides=(20, 1), transient=False)
    state = sdfg.add_state("s")
    src = state.add_access("Src")
    dst = state.add_access("Dst")
    node = TileStore("ts", widths=(4, 8))
    state.add_node(node)
    state.add_edge(src, None, node, "_src", Memlet("Src[0:4, 0:8]"))
    state.add_edge(node, "_dst", dst, None, Memlet("Dst[0:8, 0:16]"))
    with pytest.raises(NotImplementedError, match=r"non-packed stride pattern"):
        node.validate(sdfg, state)
