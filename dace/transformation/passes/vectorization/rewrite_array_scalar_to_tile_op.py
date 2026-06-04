# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrite direct ``AN(Array) <-> AN(tile-transient)`` copy edges to tile lib nodes.

In the multi-dim K=1 / K=2 staging design every write to a non-transient
global array is staged through a transient that is widened to the tile
shape, and every read from a global array mirrors the same shape. The
staging pass produces direct ``AccessNode -> AccessNode`` copy edges (no
intermediate copy tasklet); once the transient is at tile shape those
edges describe a per-tile load or store. This module exposes a single
helper that rewrites one such edge into either a :class:`TileLoad` or a
:class:`TileStore`, plus a thin :class:`ppl.Pass` wrapper that walks
every tile-tagged map in the SDFG and applies it to the qualifying edges.

The contract on the edge follows the user's spec:

* The edge is exactly ``AccessNode -> AccessNode`` (no path of nodes between).
* ``edge.data.data`` names ONE of the two endpoints; ``edge.data.subset``
  is that endpoint's subset.
* The OTHER endpoint's subset is the full tile region ``[0:W_0, ...,
  0:W_{K-1}]`` (length-1 / scalar endpoints map to a degenerate one-lane
  load on the corresponding dim).

Per-tile-dim classification piggy-backs on
:func:`classify_box_for_widths` so the resulting lib-node properties
(``dim_strides``, ``src_dims``) follow the same rules the in-place
descent already uses for connector-array copies inside body NSDFGs.
"""
from typing import Any, Dict, Optional, Set, Tuple

import dace
from dace import nodes, subsets
from dace.libraries.tileops import TileLoad, TileStore
from dace.sdfg import SDFG
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation

from dace.transformation.passes.vectorization.emit_tile_ops import _tile_region_subset
from dace.transformation.passes.vectorization.utils.name_schemes import TileConnectors
from dace.transformation.passes.vectorization.utils.promote_helpers import classify_box_for_widths
from dace.transformation.passes.vectorization.utils.tile_dims import TileDimSpec


def _is_tile_shaped(arr: dace.data.Data, widths: Tuple[int, ...]) -> bool:
    """True iff ``arr`` is an :class:`Array` whose shape matches ``widths``.

    A length-1 / :class:`Scalar` transient does NOT match; the rewrite
    only fires once the transient has been widened to the kernel tile
    shape (the staging pass's job).
    """
    if not isinstance(arr, dace.data.Array):
        return False
    return tuple(arr.shape) == tuple(widths)


def rewrite_array_scalar_copy_to_tile_op(
    state: SDFGState,
    edge: MultiConnectorEdge,
    iter_vars: Tuple[str, ...],
    widths: Tuple[int, ...],
    mask_node: Optional[nodes.AccessNode] = None,
) -> bool:
    """Rewrite one ``AN(Array) <-> AN(tile-transient)`` edge to TileLoad / TileStore.

    Direction is read from descriptors: the non-transient side is the
    global array; the transient side carries the tile shape.

    * ``AN(non-transient) -> AN(transient tile shape)`` becomes
      ``AN(non-transient) -> TileLoad -> AN(transient tile shape)``.
    * ``AN(transient tile shape) -> AN(non-transient)`` becomes
      ``AN(transient tile shape) -> TileStore -> AN(non-transient)``.

    Both endpoints must be :class:`AccessNode`; the edge memlet's
    ``data`` must name ONE of the two endpoints, and its ``subset`` is
    read on THAT endpoint. The other endpoint's subset is reconstructed
    as the full tile region ``[0:W_0, ..., 0:W_{K-1}]``. Returns
    ``False`` when the edge doesn't qualify (callers can use the return
    value to skip ineligible edges without try/except).

    :param state: State holding the edge.
    :param edge: AccessNode -> AccessNode edge to rewrite.
    :param iter_vars: Tile iter-var names (innermost-last).
    :param widths: Tile widths matching ``iter_vars``.
    :param mask_node: Optional mask :class:`AccessNode` in the same
        state. When provided, the emitted lib node carries
        ``has_mask=True`` and a wired ``_mask`` connector.
    :returns: ``True`` if the edge was rewritten; ``False`` if it
        didn't qualify.
    :raises NotImplementedError: On a non-box array-side subset
        (gather / structured); callers should refuse the kernel out loud.
    """
    if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
        return False
    if edge.data is None or edge.data.data is None:
        return False
    sdfg = state.sdfg
    src_desc = sdfg.arrays.get(edge.src.data)
    dst_desc = sdfg.arrays.get(edge.dst.data)
    if src_desc is None or dst_desc is None:
        return False
    src_is_global = not src_desc.transient and _is_tile_shaped(dst_desc, widths)
    dst_is_global = not dst_desc.transient and _is_tile_shaped(src_desc, widths)
    if not (src_is_global ^ dst_is_global):
        return False  # Both sides global, or both sides transient, or shapes don't match -- not our case.
    K = len(widths)
    tile_subset_str = ", ".join(f"0:{w}" for w in widths)
    has_mask = mask_node is not None
    if src_is_global:
        # Load: AN(global) -> TileLoad -> AN(tile transient).
        array_node, tile_node = edge.src, edge.dst
        array_desc = src_desc
        array_subset = _array_side_subset(edge, array_node.data)
        cls = classify_box_for_widths(array_subset, array_desc, iter_vars, widths)
        promoted = _tile_region_subset(array_subset, iter_vars, widths)
        load = TileLoad(name=f"load_{array_node.data}",
                        widths=widths,
                        dim_strides=tuple(cls.dim_strides),
                        src_dims=tuple(cls.match_dims),
                        has_mask=has_mask)
        state.add_node(load)
        state.add_edge(array_node, edge.src_conn, load, TileConnectors.SRC,
                       dace.Memlet(data=array_node.data, subset=promoted))
        if has_mask:
            state.add_edge(mask_node, None, load, TileConnectors.MASK, dace.Memlet(f"{mask_node.data}[{tile_subset_str}]"))
        state.add_edge(load, TileConnectors.DST, tile_node, edge.dst_conn,
                       dace.Memlet(f"{tile_node.data}[{tile_subset_str}]"))
    else:
        # Store: AN(tile transient) -> TileStore -> AN(global).
        tile_node, array_node = edge.src, edge.dst
        array_desc = dst_desc
        array_subset = _array_side_subset(edge, array_node.data)
        cls = classify_box_for_widths(array_subset, array_desc, iter_vars, widths)
        promoted = _tile_region_subset(array_subset, iter_vars, widths)
        store = TileStore(name=f"store_{array_node.data}",
                          widths=widths,
                          dim_strides=tuple(cls.dim_strides),
                          dst_dims=tuple(cls.match_dims),
                          has_mask=has_mask)
        state.add_node(store)
        state.add_edge(tile_node, edge.src_conn, store, TileConnectors.SRC,
                       dace.Memlet(f"{tile_node.data}[{tile_subset_str}]"))
        if has_mask:
            state.add_edge(mask_node, None, store, TileConnectors.MASK,
                           dace.Memlet(f"{mask_node.data}[{tile_subset_str}]"))
        state.add_edge(store, TileConnectors.DST, array_node, edge.dst_conn,
                       dace.Memlet(data=array_node.data, subset=promoted))
    state.remove_edge(edge)
    return True


def _array_side_subset(edge: MultiConnectorEdge, array_name: str) -> subsets.Range:
    """Return the array-side subset of ``edge``.

    Per the user's contract, ``edge.data.data`` names one endpoint and
    ``edge.data.subset`` is that endpoint's subset. The array-side
    subset comes from ``subset`` when ``data == array_name``, otherwise
    from ``other_subset``.
    """
    mem = edge.data
    if mem.data == array_name:
        return mem.subset
    if mem.other_subset is None:
        raise ValueError(f"Cannot resolve array-side subset for {array_name}: edge memlet "
                         f"data={mem.data!r} subset={mem.subset} has no other_subset")
    return mem.other_subset


@transformation.explicit_cf_compatible
class RewriteArrayScalarToTileOp(ppl.Pass):
    """Walk every tile-tagged map and rewrite ``AN <-> AN`` copies to tile ops.

    Body-NSDFG cases are handled by :class:`PromoteNSDFGBodyToTiles`;
    this pass targets the OUTER state (the multi-dim staging design's
    inlined Map body) where the copies sit alongside the tile transients.
    """

    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        """Adds lib nodes, rewires memlets."""
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Single fixed-point sweep is enough."""
        return False

    def depends_on(self) -> Set[type]:
        """Standalone pass."""
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Rewrite qualifying ``AN <-> AN`` copies in every tile-tagged map.

        Each map's tile spec is read from its
        :class:`~dace.transformation.passes.vectorization.utils.tile_dims.TileDimSpec`
        attribute (set by :class:`MarkTileDims`); maps without a spec
        are skipped.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Reads ``"MarkTileDims"`` when present; the
            pass is a no-op when the key is missing.
        :returns: Number of edges rewritten, or ``None`` if zero.
        """
        from dace.transformation.passes.vectorization.emit_tile_ops import _mask_name_for_map
        if not pipeline_results or "MarkTileDims" not in pipeline_results:
            return None
        specs: Dict[nodes.MapEntry, TileDimSpec] = pipeline_results["MarkTileDims"]
        total = 0
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, nodes.MapEntry) or not isinstance(g, SDFGState):
                continue
            spec = specs.get(n)
            if spec is None:
                continue
            mask_name = _mask_name_for_map(g, n)
            mask_node = None
            if mask_name is not None:
                for sn in g.scope_subgraph(n).nodes():
                    if isinstance(sn, nodes.AccessNode) and sn.data == mask_name:
                        mask_node = sn
                        break
            for edge in list(g.scope_subgraph(n).edges()):
                try:
                    if rewrite_array_scalar_copy_to_tile_op(g, edge, tuple(spec.iter_vars), tuple(spec.widths),
                                                            mask_node):
                        total += 1
                except NotImplementedError:
                    # Non-box subset -- leave for downstream gather/scatter
                    # promotion, or for the kernel-level refusal.
                    continue
        return total if total > 0 else None
