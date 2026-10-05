# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the design section 2.3 packed-layout lock.

``TileGather._src`` and ``TileScatter._dst`` must each carry an array whose
stride pattern is either packed C (row-major, no padding) or packed
Fortran (column-major, no padding). Any other layout raises
``NotImplementedError`` at ``validate()`` time -- padded layouts will
land when per-arch codegen supports them.
"""
import pytest

import dace
from dace.libraries.tileops import TileGather, TileScatter
from dace.libraries.tileops.validation import strides_match_packed, validate_packed_layout
from dace.memlet import Memlet


@pytest.mark.parametrize("shape,strides,order,packed", [
    ((8, 16), (16, 1), "C", True),
    ((8, 16), (20, 1), "C", False),
    ((8, 16), (1, 8), "F", True),
    ((8, 16), (1, 12), "F", False),
    ((8, ), (1, 1), "C", False),
],
                         ids=["c", "c_padded", "fortran", "fortran_padded", "length_mismatch"])
def test_strides_match_packed_only_for_unpadded_strides_of_the_shape(shape, strides, order, packed):
    assert strides_match_packed(shape=shape, strides=strides, order=order) is packed


# validate_packed_layout


def _array_with(shape, strides, dtype=dace.float64):
    sdfg = dace.SDFG("layout_fixture")
    sdfg.add_array("A", shape, dtype, strides=strides, transient=False)
    return sdfg.arrays["A"]


@pytest.mark.parametrize("shape,strides", [
    ((8, 16), (16, 1)),
    ((8, 16), (1, 8)),
    ((4, 8, 16), (128, 16, 1)),
    ((16, ), (1, )),
],
                         ids=["packed_c_2d", "packed_fortran_2d", "packed_c_3d", "unit_stride_1d"])
def test_validate_accepts_a_packed_layout(shape, strides):
    """Packed C, packed Fortran and unit-stride 1-D layouts pass; the validator raises on anything else."""
    validate_packed_layout("tl", "_src", _array_with(shape=shape, strides=strides))


@pytest.mark.parametrize("shape,strides,message", [
    ((8, 16), (20, 1), "non-packed stride pattern"),
    ((4, 8, 16), (192, 24, 1), "non-packed stride pattern"),
    ((16, ), (2, ), "non-unit stride"),
],
                         ids=["padded_inner_dim_2d", "padded_3d", "non_unit_stride_1d"])
def test_validate_refuses_a_non_packed_layout(shape, strides, message):
    with pytest.raises(NotImplementedError, match=message):
        validate_packed_layout("tl", "_src", _array_with(shape=shape, strides=strides))


def test_validate_accepts_scalar_descriptor_as_noop():
    """Scalars have no per-dim stride; the validator is a no-op."""
    sdfg = dace.SDFG("scalar_fixture")
    sdfg.add_scalar("S", dace.float64, transient=False)
    validate_packed_layout("tl", "_src", sdfg.arrays["S"])


# end-to-end through TileGather / TileScatter


def test_tile_gather_refuses_padded_source_at_validate():
    """A wired ``_src`` with padded strides triggers ``NotImplementedError``."""
    sdfg = dace.SDFG("tl_padded")
    sdfg.add_array("Src", (8, 16), dace.float64, strides=(20, 1), transient=False)
    sdfg.add_array("Dst", (4, 8), dace.float64, transient=True)
    state = sdfg.add_state("s")
    src = state.add_access("Src")
    dst = state.add_access("Dst")
    node = TileGather("tl", widths=(4, 8))
    state.add_node(node)
    state.add_edge(src, None, node, "_src", Memlet("Src[0:8, 0:16]"))
    state.add_edge(node, "_dst", dst, None, Memlet("Dst[0:4, 0:8]"))
    with pytest.raises(NotImplementedError, match=r"non-packed stride pattern"):
        node.validate(sdfg, state)


def test_tile_scatter_refuses_padded_dest_at_validate():
    """A wired ``_dst`` with padded strides triggers ``NotImplementedError``."""
    sdfg = dace.SDFG("ts_padded")
    sdfg.add_array("Src", (4, 8), dace.float64, transient=True)
    sdfg.add_array("Dst", (8, 16), dace.float64, strides=(20, 1), transient=False)
    state = sdfg.add_state("s")
    src = state.add_access("Src")
    dst = state.add_access("Dst")
    node = TileScatter("ts", widths=(4, 8))
    state.add_node(node)
    state.add_edge(src, None, node, "_src", Memlet("Src[0:4, 0:8]"))
    state.add_edge(node, "_dst", dst, None, Memlet("Dst[0:8, 0:16]"))
    with pytest.raises(NotImplementedError, match=r"non-packed stride pattern"):
        node.validate(sdfg, state)


if __name__ == "__main__":
    for case in [((8, 16), (16, 1), "C", True), ((8, 16), (20, 1), "C", False), ((8, 16), (1, 8), "F", True),
                 ((8, 16), (1, 12), "F", False), ((8, ), (1, 1), "C", False)]:
        test_strides_match_packed_only_for_unpadded_strides_of_the_shape(*case)
    for case in [((8, 16), (16, 1)), ((8, 16), (1, 8)), ((4, 8, 16), (128, 16, 1)), ((16, ), (1, ))]:
        test_validate_accepts_a_packed_layout(*case)
    for case in [((8, 16), (20, 1), "non-packed stride pattern"),
                 ((4, 8, 16), (192, 24, 1), "non-packed stride pattern"), ((16, ), (2, ), "non-unit stride")]:
        test_validate_refuses_a_non_packed_layout(*case)
    test_validate_accepts_scalar_descriptor_as_noop()
    test_tile_gather_refuses_padded_source_at_validate()
    test_tile_scatter_refuses_padded_dest_at_validate()
