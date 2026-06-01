# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileIota`` lib node — affine per-lane index-tile fill.

Constructed by the K-dim emitter / NSDFG-body promoter to materialize
integer index tiles for :class:`TileGather` / :class:`TileScatter` and
any other per-lane affine fill (constant arange, diagonal index,
strided base). The pure expansion lowers to a K-fold nested CPP loop;
the IR-level node is what keeps the K-dim path tile-only.

Two pinning tests:

- Constructing a no-extra-input node declares only ``_dst``.
- Constructing with ``extra_inputs=("_idx",)`` declares ``_idx`` as
  an in-connector.
"""
import dace
import pytest


def test_tile_iota_no_extra_inputs():
    """A bare ``TileIota`` has no in-connectors and one ``_dst`` out."""
    from dace.libraries.tileops import TileIota
    node = TileIota("iota_aff", widths=(8, ), expr="i + __l0")
    assert "_dst" in node.out_connectors
    assert node.in_connectors == {}
    assert node.expr == "i + __l0"
    assert list(node.widths) == [8]
    assert list(node.extra_inputs) == []


def test_tile_iota_with_idx_input():
    """``extra_inputs`` are reflected in the in-connector set."""
    from dace.libraries.tileops import TileIota
    node = TileIota("iota_idx", widths=(8, ), expr="_idx[__l0]", extra_inputs=("_idx", ))
    assert "_dst" in node.out_connectors
    assert "_idx" in node.in_connectors
    assert node.extra_inputs == ["_idx"]


def test_tile_iota_requires_expr():
    """Constructing with empty ``expr`` raises immediately."""
    from dace.libraries.tileops import TileIota
    with pytest.raises(ValueError, match="expr"):
        TileIota("iota_empty", widths=(8, ), expr="")


def test_tile_iota_widths_length_capped():
    """``widths`` must have length in ``{1, 2, 3}``."""
    from dace.libraries.tileops import TileIota
    with pytest.raises(ValueError, match="widths length"):
        TileIota("iota_K4", widths=(2, 2, 2, 2), expr="__l0 + __l1 + __l2 + __l3")


def test_tile_iota_pure_expansion_smoke():
    """End-to-end: a ``TileIota`` wired into a K=1 inner map expands to a
    CPP tasklet that fills a tile with ``i + __l0``. Validation passes;
    code-gen is deferred (we only check the expansion shape)."""
    from dace.libraries.tileops import TileIota
    sdfg = dace.SDFG("iota_smoke")
    sdfg.add_array("OUT", [8], dace.int64)
    sdfg.add_array("_tile", [8], dace.int64, storage=dace.dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map("m", {"i": "0:1"})
    iota = TileIota("iota_x", widths=(8, ), expr="i + __l0")
    state.add_node(iota)
    state.add_nedge(me, iota, dace.Memlet())
    tile_acc = state.add_access("_tile")
    state.add_edge(iota, "_dst", tile_acc, None, dace.Memlet("_tile[0:8]"))
    state.add_nedge(tile_acc, mx, dace.Memlet())
    out_acc = state.add_access("OUT")
    state.add_nedge(mx, out_acc, dace.Memlet())
    sdfg.validate()

    sdfg.expand_library_nodes()
    sdfg.validate()
    # After expansion the lib node is gone; a CPP tasklet replaces it.
    n_lib_nodes = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.LibraryNode))
    n_tasklets = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    assert n_lib_nodes == 0
    assert n_tasklets == 1


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
