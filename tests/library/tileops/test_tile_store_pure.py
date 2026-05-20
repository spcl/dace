# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end correctness for the ``pure`` expansion of :class:`TileStore`.

Symmetric to ``test_tile_load_pure``; the lib node copies a tile
transient into a destination-array region.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileStore


def _build_store_sdfg(dst_shape, widths, has_mask, dtype=dace.float64):
    """Build a minimal SDFG: tile transient -> TileStore -> destination array."""
    sdfg = dace.SDFG(f"tile_store_pure_{'x'.join(str(w) for w in widths)}_{'m' if has_mask else 'nm'}")
    sdfg.add_array("SRC", widths, dtype, transient=False)
    sdfg.add_array("DST", dst_shape, dtype, transient=False)
    if has_mask:
        sdfg.add_array("M", widths, dace.bool_, transient=False)

    state = sdfg.add_state("main")
    src_node = state.add_access("SRC")
    dst_node = state.add_access("DST")
    node = TileStore(name="ts", widths=widths, has_mask=has_mask)
    state.add_node(node)

    src_subset = ",".join(f"0:{w}" for w in widths)
    dst_subset = ",".join(f"0:{w}" for w in widths)
    state.add_edge(src_node, None, node, "_src", dace.Memlet(f"SRC[{src_subset}]"))
    state.add_edge(node, "_dst", dst_node, None, dace.Memlet(f"DST[{dst_subset}]"))
    if has_mask:
        m_node = state.add_access("M")
        state.add_edge(m_node, None, node, "_mask", dace.Memlet(f"M[{src_subset}]"))

    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("widths", [(8,), (4, 8), (2, 4, 8)])
def test_tile_store_pure_unmasked_contiguous(widths):
    """Unmasked store copies SRC into the leading tile region of DST."""
    sdfg = _build_store_sdfg(dst_shape=widths, widths=widths, has_mask=False)
    rng = np.random.default_rng(seed=23)
    SRC = rng.random(widths)
    DST = np.zeros(widths)
    sdfg(SRC=SRC, DST=DST)
    np.testing.assert_allclose(DST, SRC, rtol=0, atol=0)


def test_tile_store_pure_masked_preserves_destination_on_inactive_lanes():
    """Masked store leaves inactive lanes untouched — matches the cuTile
    ``cuda.tile.store(..., mask=)`` semantic ('no write on inactive
    lanes'). Active lanes get SRC; inactive lanes keep the destination's
    original value (99.0 here)."""
    widths = (4, 8)
    sdfg = _build_store_sdfg(dst_shape=widths, widths=widths, has_mask=True)
    rng = np.random.default_rng(seed=24)
    SRC = rng.random(widths)
    DST = np.full(widths, 99.0)
    M = np.zeros(widths, dtype=bool)
    M[:, :4] = True
    sdfg(SRC=SRC, DST=DST, M=M)
    ref = np.where(M, SRC, 99.0)
    np.testing.assert_allclose(DST, ref, rtol=0, atol=0)


def test_tile_store_rejects_invalid_K():
    """Constructor refuses K outside ``{1, 2, 3}`` and stride / width length mismatch."""
    with pytest.raises(ValueError, match="length in"):
        TileStore(name="bad_K", widths=())
    with pytest.raises(ValueError, match="dim_strides length"):
        TileStore(name="bad_stride_len", widths=(8,), dim_strides=(1, 1))
