# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``fan_out_tile_gather_index_symbols``.

The tile-path gather descent mirrors the 1D pipeline: this fan-out pass
gathers a per-lane gather index — held by the frontend in a length-1
connector ``C_idx`` (``= idx[i]``) and read by an interstate-edge
assignment ``__sym = C_idx`` — into a ``(W,)`` index tile, fanning the
symbol out into ``__sym_laneid_<l> = C_idx[l]`` per lane (the analog of
the 1D ``sym_laneid_<i>`` fan). A later collapse pass turns the fan into a
``TileGather`` and simplifies the symbols away.
"""
import dace

from dace.transformation.passes.vectorization.utils.lane_expansion import fan_out_tile_gather_index_symbols

N = dace.symbol("N")


def _build_gather_body(width: int):
    """Build a minimal parent state + body NSDFG with the gather pattern.

    Parent: an access node ``idx`` feeding the body connector ``idxc`` via
    ``idx[i]``. Body: interstate assignment ``__sym = idxc`` (the per-lane
    index read).

    :param width: Tile width W.
    :returns: ``(outer_sdfg, inner_sdfg, nsdfg_node, parent_state)``.
    """
    outer = dace.SDFG("outer")
    outer.add_array("idx", [N], dace.int64)
    pstate = outer.add_state(is_start_block=True)

    inner = dace.SDFG("body")
    inner.add_array("idxc", [1], dace.int64)
    s0 = inner.add_state("s0", is_start_block=True)
    s1 = inner.add_state("s1")
    inner.add_edge(s0, s1, dace.InterstateEdge(assignments={"__sym": "idxc"}))

    nsdfg = pstate.add_nested_sdfg(inner, {"idxc"}, set(), name="body")
    idx_an = pstate.add_access("idx")
    pstate.add_edge(idx_an, None, nsdfg, "idxc", dace.Memlet("idx[i]"))
    return outer, inner, nsdfg, pstate


def test_fan_out_widens_index_connector_and_fans_symbol():
    W = 8
    _outer, inner, nsdfg, pstate = _build_gather_body(W)

    widened = fan_out_tile_gather_index_symbols(inner, nsdfg, pstate, W, "i")

    # The (1,) index connector is gathered into a (W,) tile.
    assert widened == {"idxc"}
    assert tuple(inner.arrays["idxc"].shape) == (W, )

    # The outer edge grows from idx[i] to the tile region idx[i:i+W].
    outer_edge = next(e for e in pstate.in_edges(nsdfg) if e.dst_conn == "idxc")
    assert str(outer_edge.data.subset) == f"i:i + {W}"

    # The symbol fans out into one per-lane variant indexing the array,
    # plus the original symbol bound to lane 0.
    iedge = next(e for e in inner.all_interstate_edges() if e.data.assignments)
    a = iedge.data.assignments
    for lane in range(W):
        assert a[f"__sym_laneid_{lane}"] == f"idxc[{lane}]"
    assert a["__sym"] == "idxc[0]"


def test_fan_out_idempotent_and_noop_without_index_connector():
    """A second fan-out is a no-op (the connector is already (W,)), and an
    assignment with no length-1 index connector is left untouched."""
    W = 8
    _outer, inner, nsdfg, pstate = _build_gather_body(W)
    fan_out_tile_gather_index_symbols(inner, nsdfg, pstate, W, "i")
    # Re-running keeps the (W,) shape and the already-fanned assignments.
    again = fan_out_tile_gather_index_symbols(inner, nsdfg, pstate, W, "i")
    assert again == set()
    assert tuple(inner.arrays["idxc"].shape) == (W, )
