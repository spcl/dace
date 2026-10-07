# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileGather`` broadcast source kinds (symmetric to ``TileScatter``).

The base ``TileGather`` reads a tile-shape source via ``_src`` (the
``src_kind="Tile"`` default). Two new source kinds let the same lib
node express broadcasts without a strided tile source:

- ``src_kind="Symbol"`` — broadcast a literal / symbolic expression
  (``src_expr``) to every lane. The ``_src`` connector is omitted.
- ``src_kind="Scalar"`` — broadcast a length-1 array or
  ``dace.data.Scalar`` value read via ``_src``. The expansion uses
  ``_src[0]`` for length-1 arrays and bare ``_src`` for true Scalars
  (DaCe codegen passes Scalar connectors by value).
"""

import dace
import pytest


def test_tile_gather_symbol_minimal():
    """``src_kind="Symbol"`` declares no ``_src`` input."""
    from dace.libraries.tileops import TileGather

    node = TileGather("tl_sym", widths=(8, 8), src_kind="Symbol", src_expr="0.0")
    assert "_src" not in node.in_connectors
    assert "_dst" in node.out_connectors
    assert node.src_expr == "0.0"


def test_tile_gather_symbol_requires_expr():
    """``src_kind="Symbol"`` without ``src_expr`` raises at construction."""
    from dace.libraries.tileops import TileGather

    with pytest.raises(ValueError, match="src_expr"):
        TileGather("tl_sym_bad", widths=(8,), src_kind="Symbol")


def test_tile_gather_scalar_keeps_src_connector():
    """``src_kind="Scalar"`` still declares ``_src`` (length-1 source)."""
    from dace.libraries.tileops import TileGather

    node = TileGather("tl_scalar", widths=(8,), src_kind="Scalar")
    assert "_src" in node.in_connectors
    assert "_dst" in node.out_connectors


def test_tile_gather_unknown_src_kind():
    """Unknown ``src_kind`` rejected at construction."""
    from dace.libraries.tileops import TileGather

    with pytest.raises(ValueError, match="src_kind"):
        TileGather("tl_bad", widths=(8,), src_kind="Bogus")


def test_tile_gather_symbol_pure_expansion():
    """End-to-end: a ``TileGather(src_kind="Symbol")`` expands to a CPP
    tasklet that writes the literal to every lane. Only one Tasklet
    survives after ``expand_library_nodes``; no ``_src`` edge required."""
    from dace.libraries.tileops import TileGather

    sdfg = dace.SDFG("tl_sym_smoke")
    sdfg.add_array("OUT", [8], dace.float64)
    sdfg.add_array("_tile", [8], dace.float64, storage=dace.dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map("m", {"i": "0:1"})
    load = TileGather("tl_sym_x", widths=(8,), src_kind="Symbol", src_expr="3.14")
    state.add_node(load)
    state.add_nedge(me, load, dace.Memlet())
    tile_acc = state.add_access("_tile")
    state.add_edge(load, "_dst", tile_acc, None, dace.Memlet("_tile[0:8]"))
    state.add_nedge(tile_acc, mx, dace.Memlet())
    out_acc = state.add_access("OUT")
    state.add_nedge(mx, out_acc, dace.Memlet())
    sdfg.validate()

    sdfg.expand_library_nodes()
    sdfg.validate()
    n_lib = sum(1 for n, parent in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.LibraryNode))
    n_tasklet = sum(1 for n, parent in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    assert n_lib == 0
    assert n_tasklet == 1


if __name__ == "__main__":
    test_tile_gather_symbol_minimal()
    test_tile_gather_symbol_requires_expr()
    test_tile_gather_scalar_keeps_src_connector()
    test_tile_gather_unknown_src_kind()
    test_tile_gather_symbol_pure_expansion()
