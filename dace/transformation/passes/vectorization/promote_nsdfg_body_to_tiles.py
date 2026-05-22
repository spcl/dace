# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PromoteNSDFGBodyToTiles`` — tile a flat body-NSDFG in place (Slice E.0).

A K-dim kernel whose per-iteration compute is a long scalar chain
(``vbor``-style) is lowered by the frontend into a body NestedSDFG
(``loop_body``) sitting inside the tile map's scope, with the tile var(s)
entering through ``NestedSDFG.symbol_mapping``. :class:`EmitTileOps` only
sees tasklets directly in the flat map scope, so it cannot tile such a
body and raises (the kernel clean-skips).

This pre-pass runs after :class:`StrideMapByTileWidths` and before
:class:`EmitTileOps`. For every inner map whose scope is exactly one
NestedSDFG with a *flat* body (only ``SDFGState`` nodes — no
``LoopRegion`` / ``ConditionalBlock``; the carried-dependency cases land
in a later slice), it promotes the body to tile ops **in place**:

* every length-1 transient becomes a ``(W, ...)`` register tile;
* a connector-array copy ``a[i] -> a1`` becomes a masked :class:`TileLoad`;
* a split binop tasklet ``__out = __in1 OP __in2`` becomes a
  :class:`TileBinop` over the same (now widened) operand arrays;
* a write to an output-connector array ``x[i]`` becomes a masked
  :class:`TileStore`;
* an internal ``__out = __inp`` transient copy becomes a ``(W, ...)``
  memlet copy.

The existing dataflow + state ordering is preserved unchanged, so a
reused (non-SSA) scalar like ``vbor``'s ``a1`` (sliced, read, then
reassigned across states) stays correct: only the connector boundaries
need masking; the interior is register-local. The iteration mask
produced by :class:`GenerateTileIterationMask` in the map scope is
threaded into the NSDFG through a fresh ``_tile_iter_mask`` input
connector.

:class:`EmitTileOps` reads this pass's result (the set of handled map
entries) and skips them; it still raises for any *un*handled NSDFG body
(those maps are already W-strided, so a scalar body would be wrong — the
loud failure keeps the kernel a clean skip until its descent slice lands).
"""
from typing import Dict, List, Optional, Set, Tuple

import dace
from dace import properties, subsets
from dace.sdfg.nodes import MapEntry
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.libraries.tileops import TileBinop, TileLoad, TileMaskGen, TileStore
from dace.transformation.passes.vectorization.emit_tile_ops import (
    _classify_binop_tasklet_body,
    _is_assign_tasklet,
    _is_numeric_literal,
    _tile_region_subset,
)
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.tile_dims import (
    TileAccessKind,
    TileDimSpec,
    classify_tile_access,
)

_INNER_MASK = "_tile_iter_mask"
_BOX_KINDS = (TileAccessKind.CONTIGUOUS, TileAccessKind.STRIDED)


@properties.make_properties
class PromoteNSDFGBodyToTiles(ppl.Pass):
    """Tile a flat body-NSDFG in place so :class:`EmitTileOps` can skip it.

    Handles only the flat (no ``LoopRegion`` / ``ConditionalBlock``) body
    shape (the ``vbor`` proof point). Carried-dependency bodies wrapping a
    sequential ``LoopRegion`` (s231 / s232 / s235 / s275) are left
    untouched for a later slice.
    """

    CATEGORY: str = "Vectorization Preparation"

    widths = properties.ListProperty(
        element_type=int,
        default=[8],
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8,)):
        """Build the pass.

        :param widths: Per-dim tile widths, innermost-last (1..3 entries).
        :raises ValueError: If ``widths`` length is not in ``{1, 2, 3}``.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"PromoteNSDFGBodyToTiles: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = list(widths)

    def modifies(self) -> ppl.Modifies:
        """Pass reshapes arrays and replaces tasklets with lib nodes.

        :returns: ``ppl.Modifies.Everything``.
        """
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Runs once.

        :param modified: Modifications from earlier passes (unused).
        :returns: ``False``.
        """
        return False

    def _spec_for(self, map_entry: MapEntry) -> TileDimSpec:
        """Rebuild a :class:`TileDimSpec` from a map's last K params.

        :param map_entry: Inner map entry.
        :returns: A fresh :class:`TileDimSpec` over the K innermost dims.
        """
        K = len(self.widths)
        params = list(map_entry.map.params)
        ranges = list(map_entry.map.range.ranges)
        iter_vars = tuple(params[-K:])
        global_ubs = tuple(str(r[1] + 1) for r in ranges[-K:])
        return TileDimSpec(iter_vars=iter_vars, widths=tuple(self.widths), global_ubs=global_ubs)

    def _mask_name_for_map(self, state: SDFGState, map_entry: MapEntry) -> str:
        """Return the mask transient produced inside ``map_entry``'s scope.

        :param state: Parent state.
        :param map_entry: Inner map entry.
        :returns: The ``TileMaskGen`` output array name.
        :raises NotImplementedError: If no mask producer is in scope.
        """
        scope = state.all_nodes_between(map_entry, state.exit_node(map_entry)) or set()
        for node in scope:
            if isinstance(node, TileMaskGen):
                oe = [e for e in state.out_edges(node) if e.src_conn == "_o"]
                if oe:
                    return oe[0].data.data
        raise NotImplementedError(
            f"PromoteNSDFGBodyToTiles: map {map_entry.label!r} has no TileMaskGen in scope; "
            f"run GenerateTileIterationMask first.")

    def _mask_access(self, state: SDFGState, mask_name: str) -> dace.nodes.AccessNode:
        """Return the producer-fed mask access node, or a fresh one.

        :param state: Parent state.
        :param mask_name: Iteration-mask transient name.
        :returns: An access node for ``mask_name``.
        """
        for n in state.data_nodes():
            if n.data == mask_name and state.in_edges(n):
                return n
        return state.add_access(mask_name)

    def _flat_body_nsdfg(self, state: SDFGState, map_entry: MapEntry) -> Optional[dace.nodes.NestedSDFG]:
        """Return the body NSDFG iff this is the flat-body case.

        Eligible when the map scope holds exactly one :class:`NestedSDFG`,
        no compute tasklets, and the inner SDFG is flat (only
        :class:`SDFGState` CFG nodes).

        :param state: Parent state.
        :param map_entry: Inner map entry.
        :returns: The body NestedSDFG, or ``None`` if ineligible.
        """
        scope = state.all_nodes_between(map_entry, state.exit_node(map_entry)) or set()
        nsdfgs = [n for n in scope if isinstance(n, dace.nodes.NestedSDFG)]
        tasklets = [n for n in scope if isinstance(n, dace.nodes.Tasklet)]
        if len(nsdfgs) != 1 or tasklets:
            return None
        nsdfg = nsdfgs[0]
        if any(not isinstance(cfg, SDFGState) for cfg in nsdfg.sdfg.nodes()):
            return None
        return nsdfg

    def _thread_mask(self, parent_state: SDFGState, map_entry: MapEntry, nsdfg_node: dace.nodes.NestedSDFG,
                     mask_name: str, widths: Tuple[int, ...]) -> None:
        """Pass the map-scope mask into the NSDFG via a new connector.

        :param parent_state: State holding the map + NSDFG.
        :param map_entry: Inner map entry.
        :param nsdfg_node: The body NestedSDFG node.
        :param mask_name: Map-scope mask transient name.
        :param widths: Tile widths.
        """
        inner = nsdfg_node.sdfg
        if _INNER_MASK not in inner.arrays:
            inner.add_array(_INNER_MASK, list(widths), dace.bool_,
                            storage=dace.dtypes.StorageType.Register, transient=False)
        nsdfg_node.add_in_connector(_INNER_MASK)
        subset = ", ".join(f"0:{w}" for w in widths)
        mask_access = self._mask_access(parent_state, mask_name)
        parent_state.add_edge(mask_access, None, nsdfg_node, _INNER_MASK,
                              dace.Memlet(f"{mask_name}[{subset}]"))

    def _reshape_transients(self, inner: dace.SDFG, widths: Tuple[int, ...]) -> Set[str]:
        """Reshape every length-1 transient to a ``(W, ...)`` register tile.

        Rewrites the memlets of the reshaped arrays to the full tile region.

        :param inner: The body SDFG.
        :param widths: Tile widths.
        :returns: The set of reshaped array names.
        """
        reshaped: Set[str] = set()
        for name, arr in list(inner.arrays.items()):
            if arr.transient and tuple(arr.shape) == (1, ):
                dtype = arr.dtype
                inner.remove_data(name, validate=False)
                inner.add_array(name, list(widths), dtype,
                                storage=dace.dtypes.StorageType.Register, transient=True)
                reshaped.add(name)
        tile_subset = subsets.Range([(0, w - 1, 1) for w in widths])
        for istate in inner.states():
            for ed in istate.edges():
                if ed.data is not None and ed.data.data in reshaped:
                    ed.data = dace.Memlet(data=ed.data.data, subset=subsets.Range(list(tile_subset.ranges)))
        return reshaped

    def _box_classification(self, subset: subsets.Range, arr: dace.data.Data, iter_vars: Tuple[str, ...]):
        """Classify a connector access and require a perfect box.

        :param subset: Per-iteration subset on the connector array.
        :param arr: The connector array descriptor.
        :param iter_vars: Tile iter-vars.
        :returns: The :class:`TileAccessClassification`.
        :raises NotImplementedError: For non-box (gather/structured) access.
        """
        cls = classify_tile_access(subset, tuple(arr.strides), iter_vars)
        if cls.kind not in _BOX_KINDS:
            raise NotImplementedError(
                f"PromoteNSDFGBodyToTiles: connector access {subset} is {cls.kind.value}; "
                f"only perfect-box (contiguous / strided) loads/stores are supported in this slice")
        return cls

    def _promote_loads(self, istate: SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                       mask_acc: dace.nodes.AccessNode, spec: TileDimSpec) -> None:
        """Replace connector-array copy edges with masked :class:`TileLoad`.

        :param istate: Inner state being rewritten.
        :param nsdfg_node: The body NestedSDFG node (for connector names).
        :param mask_acc: This state's inner mask access node.
        :param spec: Tile spec.
        """
        inner = istate.sdfg
        W = tuple(spec.widths)
        subset = ", ".join(f"0:{w}" for w in W)
        load_edges = [
            e for e in istate.edges()
            if isinstance(e.src, dace.nodes.AccessNode) and isinstance(e.dst, dace.nodes.AccessNode)
            and e.src.data in nsdfg_node.in_connectors and not inner.arrays[e.src.data].transient
            and e.src.data != _INNER_MASK and inner.arrays[e.dst.data].transient
        ]
        for ed in load_edges:
            src_name = ed.src.data
            cls = self._box_classification(ed.data.subset, inner.arrays[src_name], spec.iter_vars)
            promoted = _tile_region_subset(ed.data.subset, spec.iter_vars, W)
            load = TileLoad(name=f"load_{src_name}", widths=W, dim_strides=tuple(cls.dim_strides),
                            src_dims=tuple(cls.match_dims), has_mask=True)
            istate.add_node(load)
            istate.add_edge(ed.src, ed.src_conn, load, "_src", dace.Memlet(data=src_name, subset=promoted))
            istate.add_edge(mask_acc, None, load, "_mask", dace.Memlet(f"{_INNER_MASK}[{subset}]"))
            istate.add_edge(load, "_dst", ed.dst, ed.dst_conn, dace.Memlet(f"{ed.dst.data}[{subset}]"))
            istate.remove_edge(ed)

    def _promote_binops(self, istate: SDFGState, mask_acc: dace.nodes.AccessNode, spec: TileDimSpec) -> None:
        """Replace split binop tasklets with :class:`TileBinop`.

        :param istate: Inner state being rewritten.
        :param mask_acc: This state's inner mask access node.
        :param spec: Tile spec.
        """
        inner = istate.sdfg
        W = tuple(spec.widths)
        subset = ", ".join(f"0:{w}" for w in W)
        binops = []
        for t in istate.nodes():
            if not isinstance(t, dace.nodes.Tasklet) or _is_assign_tasklet(t):
                continue
            parsed = _classify_binop_tasklet_body(t)
            if parsed is not None:
                binops.append((t, parsed))
        for t, (out_conn, a_tok, op, b_tok) in binops:
            out_edge = istate.out_edges(t)[0]
            out_access = out_edge.dst

            def _operand(token):
                if _is_numeric_literal(token):
                    return "Symbol", token
                ie = [e for e in istate.in_edges(t) if e.dst_conn == token]
                if len(ie) != 1 or not isinstance(ie[0].src, dace.nodes.AccessNode):
                    raise NotImplementedError(
                        f"PromoteNSDFGBodyToTiles: binop {t.label!r} operand {token!r} not a single tile read")
                return "Tile", ie[0].src

            kind_a, info_a = _operand(a_tok)
            kind_b, info_b = _operand(b_tok)
            if kind_a != "Tile" and kind_b != "Tile":
                raise NotImplementedError(
                    f"PromoteNSDFGBodyToTiles: binop {t.label!r} has no tile operand ({kind_a}/{kind_b})")
            kwargs = dict(name=f"{t.label}_binop", widths=W, op=op, has_mask=True, kind_a=kind_a, kind_b=kind_b)
            if kind_a == "Symbol":
                kwargs["expr_a"] = info_a
            if kind_b == "Symbol":
                kwargs["expr_b"] = info_b
            binop = TileBinop(**kwargs)
            istate.add_node(binop)
            if kind_a == "Tile":
                istate.add_edge(info_a, None, binop, "_a", dace.Memlet(f"{info_a.data}[{subset}]"))
            if kind_b == "Tile":
                istate.add_edge(info_b, None, binop, "_b", dace.Memlet(f"{info_b.data}[{subset}]"))
            istate.add_edge(mask_acc, None, binop, "_mask", dace.Memlet(f"{_INNER_MASK}[{subset}]"))
            istate.add_edge(binop, "_c", out_access, out_edge.dst_conn, dace.Memlet(f"{out_access.data}[{subset}]"))
            for e in list(istate.in_edges(t)) + list(istate.out_edges(t)):
                istate.remove_edge(e)
            istate.remove_node(t)

    def _promote_stores(self, istate: SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                        mask_acc: dace.nodes.AccessNode, spec: TileDimSpec) -> None:
        """Replace writes to output-connector arrays with masked :class:`TileStore`.

        :param istate: Inner state being rewritten.
        :param nsdfg_node: The body NestedSDFG node (for connector names).
        :param mask_acc: This state's inner mask access node.
        :param spec: Tile spec.
        """
        inner = istate.sdfg
        W = tuple(spec.widths)
        subset = ", ".join(f"0:{w}" for w in W)
        store_edges = [
            e for e in istate.edges()
            if isinstance(e.dst, dace.nodes.AccessNode) and e.dst.data in nsdfg_node.out_connectors
            and e.data is not None and e.data.data is not None
        ]
        for ed in store_edges:
            dst_name = ed.dst.data
            cls = self._box_classification(ed.data.subset, inner.arrays[dst_name], spec.iter_vars)
            promoted = _tile_region_subset(ed.data.subset, spec.iter_vars, W)
            if isinstance(ed.src, dace.nodes.Tasklet) and _is_assign_tasklet(ed.src):
                assign = ed.src
                src_edge = istate.in_edges(assign)[0]
                src_access = src_edge.src
                to_remove_node = assign
            elif isinstance(ed.src, dace.nodes.AccessNode):
                src_access = ed.src
                to_remove_node = None
            else:
                raise NotImplementedError(
                    f"PromoteNSDFGBodyToTiles: store into {dst_name!r} from {type(ed.src).__name__} unsupported")
            store = TileStore(name=f"store_{dst_name}", widths=W, dim_strides=tuple(cls.dim_strides),
                              dst_dims=tuple(cls.match_dims), has_mask=True)
            istate.add_node(store)
            istate.add_edge(src_access, None, store, "_src", dace.Memlet(f"{src_access.data}[{subset}]"))
            istate.add_edge(mask_acc, None, store, "_mask", dace.Memlet(f"{_INNER_MASK}[{subset}]"))
            istate.add_edge(store, "_dst", ed.dst, ed.dst_conn, dace.Memlet(data=dst_name, subset=promoted))
            if to_remove_node is not None:
                for e in list(istate.in_edges(to_remove_node)) + list(istate.out_edges(to_remove_node)):
                    istate.remove_edge(e)
                istate.remove_node(to_remove_node)
            else:
                istate.remove_edge(ed)

    def _promote_internal_assigns(self, istate: SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                                  spec: TileDimSpec) -> None:
        """Turn transient->transient ``__out = __inp`` assigns into tile copies.

        :param istate: Inner state being rewritten.
        :param nsdfg_node: The body NestedSDFG node (for connector names).
        :param spec: Tile spec.
        """
        W = tuple(spec.widths)
        subset = ", ".join(f"0:{w}" for w in W)
        assigns = [
            t for t in istate.nodes()
            if isinstance(t, dace.nodes.Tasklet) and _is_assign_tasklet(t)
            and istate.in_edges(t) and istate.out_edges(t)
            and istate.out_edges(t)[0].dst.data not in nsdfg_node.out_connectors
        ]
        for t in assigns:
            in_e = istate.in_edges(t)[0]
            out_e = istate.out_edges(t)[0]
            istate.add_edge(in_e.src, in_e.src_conn, out_e.dst, out_e.dst_conn,
                            dace.Memlet(f"{out_e.dst.data}[{subset}]"))
            for e in list(istate.in_edges(t)) + list(istate.out_edges(t)):
                istate.remove_edge(e)
            istate.remove_node(t)

    def _drop_isolated(self, istate: SDFGState) -> None:
        """Remove access nodes left with no edges after the rewrite.

        :param istate: Inner state being rewritten.
        """
        for n in list(istate.nodes()):
            if isinstance(n, dace.nodes.AccessNode) and istate.degree(n) == 0:
                istate.remove_node(n)

    def _promote(self, parent_state: SDFGState, map_entry: MapEntry, nsdfg_node: dace.nodes.NestedSDFG,
                 spec: TileDimSpec, mask_name: str) -> None:
        """Promote one flat body NSDFG to tile ops in place.

        :param parent_state: State holding the map + NSDFG.
        :param map_entry: Inner map entry.
        :param nsdfg_node: The body NestedSDFG node.
        :param spec: Tile spec.
        :param mask_name: Map-scope mask transient name.
        """
        W = tuple(spec.widths)
        inner = nsdfg_node.sdfg
        self._thread_mask(parent_state, map_entry, nsdfg_node, mask_name, W)
        self._reshape_transients(inner, W)
        for istate in inner.states():
            mask_acc = istate.add_access(_INNER_MASK)
            self._promote_loads(istate, nsdfg_node, mask_acc, spec)
            self._promote_binops(istate, mask_acc, spec)
            self._promote_stores(istate, nsdfg_node, mask_acc, spec)
            self._promote_internal_assigns(istate, nsdfg_node, spec)
            if istate.degree(mask_acc) == 0:
                istate.remove_node(mask_acc)
            self._drop_isolated(istate)

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Optional[Dict]) -> Optional[Set[MapEntry]]:
        """Promote every eligible flat body-NSDFG map to tile ops.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Reads ``"MarkTileDims"`` when present.
        :returns: The set of handled :class:`MapEntry` nodes, or ``None``.
        """
        specs: Optional[Dict[MapEntry, TileDimSpec]] = None
        if pipeline_results and "MarkTileDims" in pipeline_results:
            specs = pipeline_results["MarkTileDims"]
        K = len(self.widths)
        handled: Set[MapEntry] = set()
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, MapEntry) or not isinstance(g, SDFGState):
                continue
            if not is_innermost_map(g, n):
                continue
            if specs is not None and n not in specs:
                continue
            if len(n.map.params) < K:
                continue
            nsdfg = self._flat_body_nsdfg(g, n)
            if nsdfg is None:
                continue
            spec = specs[n] if specs is not None and n in specs else self._spec_for(n)
            mask_name = self._mask_name_for_map(g, n)
            self._promote(g, n, nsdfg, spec, mask_name)
            handled.add(n)
        return handled or None
