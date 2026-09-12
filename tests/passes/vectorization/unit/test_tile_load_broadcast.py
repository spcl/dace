# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileLoad`` broadcast source kinds (symmetric to ``TileStore``).

The base ``TileLoad`` reads a tile-shape source via ``_src`` (the
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


def test_tile_load_symbol_minimal():
    """``src_kind="Symbol"`` declares no ``_src`` input."""
    from dace.libraries.tileops import TileLoad
    node = TileLoad("tl_sym", widths=(8, 8), src_kind="Symbol", src_expr="0.0")
    assert "_src" not in node.in_connectors
    assert "_dst" in node.out_connectors
    assert node.src_expr == "0.0"


def test_tile_load_symbol_requires_expr():
    """``src_kind="Symbol"`` without ``src_expr`` raises at construction."""
    from dace.libraries.tileops import TileLoad
    with pytest.raises(ValueError, match="src_expr"):
        TileLoad("tl_sym_bad", widths=(8, ), src_kind="Symbol")


def test_tile_load_scalar_keeps_src_connector():
    """``src_kind="Scalar"`` still declares ``_src`` (length-1 source)."""
    from dace.libraries.tileops import TileLoad
    node = TileLoad("tl_scalar", widths=(8, ), src_kind="Scalar")
    assert "_src" in node.in_connectors
    assert "_dst" in node.out_connectors


def test_tile_load_unknown_src_kind():
    """Unknown ``src_kind`` rejected at construction."""
    from dace.libraries.tileops import TileLoad
    with pytest.raises(ValueError, match="src_kind"):
        TileLoad("tl_bad", widths=(8, ), src_kind="Bogus")


def test_tile_load_symbol_pure_expansion():
    """End-to-end: a ``TileLoad(src_kind="Symbol")`` expands to a CPP
    tasklet that writes the literal to every lane. Only one Tasklet
    survives after ``expand_library_nodes``; no ``_src`` edge required."""
    from dace.libraries.tileops import TileLoad
    sdfg = dace.SDFG("tl_sym_smoke")
    sdfg.add_array("OUT", [8], dace.float64)
    sdfg.add_array("_tile", [8], dace.float64, storage=dace.dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map("m", {"i": "0:1"})
    load = TileLoad("tl_sym_x", widths=(8, ), src_kind="Symbol", src_expr="3.14")
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
    n_lib = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.LibraryNode))
    n_tasklet = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    assert n_lib == 0
    assert n_tasklet == 1


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
