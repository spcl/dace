# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Splat a single-element source across a full tile.

A Scalar (or length-1 Array) source carries ONE value where a tile needs ``W`` lanes. Copying it
into a ``(W, ...)`` buffer defines lane 0 and leaves lanes 1..W-1 uninitialized, so such a source
must be BROADCAST. Shared by :class:`InsertTileLoadStore` and :class:`ConvertTaskletsToTileOps`,
which run in that order over the same bodies. The splat is a ``TileLoad`` and NOT a CPP tasklet on
purpose: ``map_body_has_foreign_language_tasklet`` would disqualify the whole map from every later
tile pass, while the vectorizer's own tile ops stay transparent to that gate.
"""
from dace import data as dd, symbolic
from dace.libraries.tileops import TileLoad
from dace.memlet import Memlet
from dace.sdfg.graph import Edge
from dace.sdfg.nodes import AccessNode, Node
from dace.sdfg.state import SDFGState


def is_scalar_or_len1_source(state: SDFGState, edge: Edge) -> bool:
    """True when ``edge.src`` reads a Scalar or an all-extent-1 Array.

    :param state: State holding ``edge``.
    :param edge: Edge whose source descriptor is inspected.
    :returns: Whether the source can supply at most one element.
    """
    if not isinstance(edge.src, AccessNode):
        return False
    desc = state.sdfg.arrays.get(edge.src.data)
    if desc is None:
        return False
    if isinstance(desc, dd.Scalar):
        return True
    if isinstance(desc, dd.Array):
        try:
            return all(bool(symbolic.simplify(s - 1) == 0) for s in desc.shape)
        except Exception:  # noqa: BLE001
            return False
    return False


def splat_scalar_to_tile(state: SDFGState, name: str, src_node: Node, src_conn: str | None, src_memlet: Memlet,
                         dst_node: Node, dst_conn: str | None, dst_data: str, widths: tuple[int, ...]) -> TileLoad:
    """Wire a ``TileLoad(src_kind='Scalar')`` splatting one source element into every lane.

    :param state: State to build in.
    :param name: Label for the new node.
    :param src_node: Producer of the single-element value.
    :param src_conn: Source connector on ``src_node``, or ``None``.
    :param src_memlet: Memlet reading that value.
    :param dst_node: Endpoint receiving the full tile.
    :param dst_conn: Destination connector, or ``None``.
    :param dst_data: Name of the ``widths``-shaped tile the result lands in.
    :param widths: Per-tile-dim widths, innermost-last.
    :returns: The inserted node.
    """
    tile_load = TileLoad(name=name, widths=widths, src_kind="Scalar")
    state.add_node(tile_load)
    state.add_edge(src_node, src_conn, tile_load, "_src", Memlet.from_memlet(src_memlet))
    tile_subset = ", ".join(f"0:{w}" for w in widths)
    state.add_edge(tile_load, "_dst", dst_node, dst_conn, Memlet(data=dst_data, subset=tile_subset))
    return tile_load
