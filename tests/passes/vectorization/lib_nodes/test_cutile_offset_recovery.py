# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``cutile_tile_dim_offsets`` hardening (bug 16 follow-up).

The offset-recovery helper used to substitute EVERY enclosing CuTile-map
parameter with 0 to isolate the constant begin offset. That silently zeroed
FOREIGN map parameters in coupled begins (``A[__i0 + __i1]`` — exactly the
silent-wrong bug class 16 was about), silently dropped a nonzero map start,
and swallowed parse failures via a bare ``except: offsets.append(0)``.

The hardened helper substitutes only the matched dim's OWN parameter — with
its range START, so a nonzero-start map (the vectorizer's remainder map)
folds the start into the offset — and raises ``NotImplementedError`` loudly
for coupled begins and unparseable begins. Offsets free of map parameters
(including symbolic SDFG-symbol offsets) are returned for the caller's
divisibility/gather decision.

No GPU required: these operate on the symbolic helper directly.
"""
import pytest

import dace
from dace.dtypes import ScheduleType
from dace.libraries.tileops import TileGather
from dace.libraries.tileops.cutile import cutile_tile_dim_offsets


def _node_in_cutile_map(map_ranges: dict) -> tuple:
    """Build a ``TileGather`` inside a CuTile-scheduled map.

    :param map_ranges: Map parameter -> range string (e.g. ``{"__i0": "0:64:8"}``).
    :returns: ``(node, state, sdfg)`` for calling the offset helper.
    """
    sdfg = dace.SDFG(f"offs_{abs(hash(tuple(sorted(map_ranges.items())))) % 10**8}")
    state = sdfg.add_state("main")
    sdfg.add_array("src", (128, ), dace.float64)
    sdfg.add_array("out", (128, ), dace.float64)
    sdfg.add_array("dst", (8, ), dace.float64, transient=True)
    me, mx = state.add_map("m", map_ranges, schedule=ScheduleType.CuTile)
    node = TileGather(name="L", widths=(8, ))
    state.add_node(node)
    src = state.add_access("src")
    dst = state.add_access("dst")
    state.add_memlet_path(src, me, node, dst_conn="_src", memlet=dace.Memlet.from_array("src", sdfg.arrays["src"]))
    state.add_edge(node, "_dst", dst, None, dace.Memlet.from_array("dst", sdfg.arrays["dst"]))
    state.add_memlet_path(dst, mx, state.add_access("out"), memlet=dace.Memlet("out[0:8]"))
    sdfg.fill_scope_connectors()
    return node, state, sdfg


def test_constant_offset_recovered():
    """``__i0 + 3`` with a 0-start map yields offset 3."""
    node, state, sdfg = _node_in_cutile_map({"__i0": "0:64:8"})
    offs = cutile_tile_dim_offsets(node, state, sdfg, (0, ), ["__i0 + 3"], 1)
    assert len(offs) == 1 and int(offs[0]) == 3


def test_zero_offset_recovered():
    """A block-anchored begin (``__i0``) yields offset 0."""
    node, state, sdfg = _node_in_cutile_map({"__i0": "0:64:8"})
    offs = cutile_tile_dim_offsets(node, state, sdfg, (0, ), ["__i0"], 1)
    assert len(offs) == 1 and int(offs[0]) == 0


def test_symbolic_offset_recovered():
    """A begin offset by a non-map SDFG symbol (``__i0 + N``) is returned
    symbolically (the caller decides divisibility / gather routing)."""
    node, state, sdfg = _node_in_cutile_map({"__i0": "0:64:8"})
    offs = cutile_tile_dim_offsets(node, state, sdfg, (0, ), ["__i0 + N"], 1)
    assert len(offs) == 1 and str(offs[0]) == "N"


def test_coupled_params_raise():
    """A begin coupling two map parameters (``__i0 + __i1``) must raise, not
    silently zero the foreign parameter."""
    node, state, sdfg = _node_in_cutile_map({"__i0": "0:64:8", "__i1": "0:64:8"})
    with pytest.raises(NotImplementedError, match="couples"):
        cutile_tile_dim_offsets(node, state, sdfg, (0, ), ["__i0 + __i1"], 1)


def test_nonzero_map_start_folded_into_offset():
    """A map whose own parameter starts at 4 folds the start into the offset
    (``__pid * W`` counts blocks relative to the range start): begin
    ``__i0 + 3`` with range ``4:64:8`` recovers offset ``4 + 3 = 7``."""
    node, state, sdfg = _node_in_cutile_map({"__i0": "4:64:8"})
    offs = cutile_tile_dim_offsets(node, state, sdfg, (0, ), ["__i0 + 3"], 1)
    assert len(offs) == 1 and int(offs[0]) == 7


def test_remainder_map_symbolic_start_folded():
    """The vectorizer's remainder map (start ``8*int_floor(N, 8)``) folds its
    symbolic start into the offset instead of raising — the shape that arises
    for non-divisible symbolic sizes on the CPU/remainder track."""
    node, state, sdfg = _node_in_cutile_map({"__i0": "8*int_floor(N, 8):N:8"})
    offs = cutile_tile_dim_offsets(node, state, sdfg, (0, ), ["__i0"], 1)
    assert len(offs) == 1 and str(offs[0]) == "8*int_floor(N, 8)"


def test_nonzero_map_start_other_param_ok():
    """A nonzero start on a DIFFERENT (unmatched) map parameter does not
    affect this dim's recovery."""
    node, state, sdfg = _node_in_cutile_map({"__i0": "0:64:8", "__i1": "4:64:8"})
    offs = cutile_tile_dim_offsets(node, state, sdfg, (0, ), ["__i0 + 2"], 1)
    assert len(offs) == 1 and int(offs[0]) == 2


def test_single_iteration_param_substituted_exactly():
    """A coupled begin whose second parameter has a provably single-iteration
    range (``0:4:4`` -> always 0) is exact, not coupled: no raise, offset 0.
    (Arises from flattened views, e.g. ``A[4*__i0 + __i1]``.)"""
    node, state, sdfg = _node_in_cutile_map({"__i0": "0:4:1", "__i1": "0:4:4"})
    offs = cutile_tile_dim_offsets(node, state, sdfg, (0, ), ["4*__i0 + __i1"], 1)
    assert len(offs) == 1 and int(offs[0]) == 0


def test_scaled_own_param_offset_recovered():
    """A strided begin (``2*__i0 + 1``) removes the whole own-parameter term
    and recovers the additive constant 1 (the coefficient is carried by
    ``dim_strides``, not the offset)."""
    node, state, sdfg = _node_in_cutile_map({"__i0": "0:64:8"})
    offs = cutile_tile_dim_offsets(node, state, sdfg, (0, ), ["2*__i0 + 1"], 1)
    assert len(offs) == 1 and int(offs[0]) == 1


if __name__ == "__main__":
    test_constant_offset_recovered()
    test_zero_offset_recovered()
    test_symbolic_offset_recovered()
    test_coupled_params_raise()
    test_nonzero_map_start_folded_into_offset()
    test_remainder_map_symbolic_start_folded()
    test_nonzero_map_start_other_param_ok()
    test_single_iteration_param_substituted_exactly()
    test_scaled_own_param_offset_recovered()
    print("offset recovery hardening tests passed")
