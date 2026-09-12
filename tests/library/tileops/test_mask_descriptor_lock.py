# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the design section 10.2 mask descriptor lock.

Each lib node that carries a mask connector (``TileMaskGen._o``,
``TileLoad._mask``, ``TileStore._mask``) must enforce the lock at
``validate()`` time: ``Array(shape=widths, dtype=bool_, storage=Register,
transient=True)``. Any other descriptor is rejected with a named error.
"""
import pytest

import dace
from dace import dtypes
from dace.libraries.tileops import TileLoad, TileMaskGen, TileStore
from dace.memlet import Memlet


def _add_mask(sdfg, name, shape, dtype=dace.bool_, storage=dtypes.StorageType.Register, transient=True):
    sdfg.add_array(name, shape, dtype, storage=storage, transient=transient)
    return name


# ---- TileMaskGen ---------------------------------------------------------


def _build_mg(mask_shape, mask_dtype=dace.bool_, mask_storage=dtypes.StorageType.Register, mask_transient=True):
    sdfg = dace.SDFG("mg_fixture")
    _add_mask(sdfg, "M", mask_shape, mask_dtype, mask_storage, mask_transient)
    state = sdfg.add_state("s")
    m_an = state.add_access("M")
    node = TileMaskGen(name="mg", widths=(4, 8), iter_vars=("i", "j"), global_ubs=("M_i", "M_j"))
    state.add_node(node)
    state.add_edge(node, "_o", m_an, None, Memlet(f"M[{','.join(f'0:{w}' for w in mask_shape)}]"))
    return sdfg, state, node


def test_tilemaskgen_accepts_locked_descriptor():
    sdfg, state, node = _build_mg(mask_shape=(4, 8))
    node.validate(sdfg, state)


def test_tilemaskgen_refuses_shape_mismatch():
    sdfg, state, node = _build_mg(mask_shape=(4, 16))  # j width should be 8
    with pytest.raises(ValueError, match=r"shape.*must match widths"):
        node.validate(sdfg, state)


def test_tilemaskgen_refuses_wrong_dtype():
    sdfg, state, node = _build_mg(mask_shape=(4, 8), mask_dtype=dace.int64)
    with pytest.raises(ValueError, match=r"dtype.*must be bool_"):
        node.validate(sdfg, state)


def test_tilemaskgen_refuses_non_register_storage():
    sdfg, state, node = _build_mg(mask_shape=(4, 8), mask_storage=dtypes.StorageType.CPU_Heap)
    with pytest.raises(ValueError, match=r"storage.*must be.*Register"):
        node.validate(sdfg, state)


def test_tilemaskgen_refuses_non_transient_mask():
    sdfg, state, node = _build_mg(mask_shape=(4, 8), mask_transient=False)
    with pytest.raises(ValueError, match=r"must be transient"):
        node.validate(sdfg, state)


# ---- TileLoad._mask -------------------------------------------------------


def _build_load_with_mask(mask_shape, **mask_kwargs):
    """Build a TileLoad with has_mask=True + a wired _mask transient."""
    sdfg = dace.SDFG("tl_mask_fixture")
    sdfg.add_array("Src", (16, 32), dace.float64, transient=False)
    sdfg.add_array("Dst", (4, 8), dace.float64, transient=True)
    _add_mask(sdfg, "M", mask_shape, **mask_kwargs)
    state = sdfg.add_state("s")
    src_an = state.add_access("Src")
    dst_an = state.add_access("Dst")
    m_an = state.add_access("M")
    node = TileLoad(name="tl", widths=(4, 8), has_mask=True)
    state.add_node(node)
    state.add_edge(src_an, None, node, "_src", Memlet("Src[0:16, 0:32]"))
    state.add_edge(m_an, None, node, "_mask", Memlet(f"M[{','.join(f'0:{w}' for w in mask_shape)}]"))
    state.add_edge(node, "_dst", dst_an, None, Memlet("Dst[0:4, 0:8]"))
    return sdfg, state, node


def test_tileload_accepts_locked_mask():
    sdfg, state, node = _build_load_with_mask(mask_shape=(4, 8))
    node.validate(sdfg, state)


def test_tileload_refuses_mask_shape_mismatch():
    sdfg, state, node = _build_load_with_mask(mask_shape=(4, 16))
    with pytest.raises(ValueError, match=r"shape.*must match widths"):
        node.validate(sdfg, state)


def test_tileload_refuses_non_register_mask_storage():
    sdfg, state, node = _build_load_with_mask(mask_shape=(4, 8), storage=dtypes.StorageType.CPU_Heap)
    with pytest.raises(ValueError, match=r"storage.*must be.*Register"):
        node.validate(sdfg, state)


def test_tileload_refuses_non_transient_mask():
    sdfg, state, node = _build_load_with_mask(mask_shape=(4, 8), transient=False)
    with pytest.raises(ValueError, match=r"must be transient"):
        node.validate(sdfg, state)


# ---- TileStore._mask ------------------------------------------------------


def _build_store_with_mask(mask_shape, **mask_kwargs):
    sdfg = dace.SDFG("ts_mask_fixture")
    sdfg.add_array("Src", (4, 8), dace.float64, transient=True)
    sdfg.add_array("Dst", (16, 32), dace.float64, transient=False)
    _add_mask(sdfg, "M", mask_shape, **mask_kwargs)
    state = sdfg.add_state("s")
    src_an = state.add_access("Src")
    dst_an = state.add_access("Dst")
    m_an = state.add_access("M")
    node = TileStore(name="ts", widths=(4, 8), has_mask=True)
    state.add_node(node)
    state.add_edge(src_an, None, node, "_src", Memlet("Src[0:4, 0:8]"))
    state.add_edge(m_an, None, node, "_mask", Memlet(f"M[{','.join(f'0:{w}' for w in mask_shape)}]"))
    state.add_edge(node, "_dst", dst_an, None, Memlet("Dst[0:4, 0:8]"))
    return sdfg, state, node


def test_tilestore_accepts_locked_mask():
    sdfg, state, node = _build_store_with_mask(mask_shape=(4, 8))
    node.validate(sdfg, state)


def test_tilestore_refuses_mask_wrong_dtype():
    sdfg, state, node = _build_store_with_mask(mask_shape=(4, 8), dtype=dace.int64)
    with pytest.raises(ValueError, match=r"dtype.*must be bool_"):
        node.validate(sdfg, state)
