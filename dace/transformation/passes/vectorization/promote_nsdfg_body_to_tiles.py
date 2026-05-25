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

from dace.libraries.tileops import TileBinop, TileGather, TileLoad, TileMaskGen, TileStore
from dace.transformation.passes.vectorization.emit_tile_ops import (
    _classify_binop_tasklet_body,
    _is_assign_tasklet,
    _is_numeric_literal,
    _tile_region_subset,
)
from dace.transformation.passes.vectorization.utils.lane_expansion import fan_out_tile_gather_index_symbols
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme, TileConnectors
from dace.transformation.passes.vectorization.utils.tile_dims import (
    TileAccessClassification,
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
                oe = [e for e in state.out_edges(node) if e.src_conn == TileConnectors.O]
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

    def _tilevar_in(self, expr, tile_var_set: Set[str]) -> Optional[str]:
        """Return the tile iter-var referenced by ``expr``, or ``None``.

        :param expr: A subset begin expression.
        :param tile_var_set: The tile iter-var names.
        :returns: The first tile iter-var name in ``expr``, else ``None``.
        """
        syms = expr.free_symbols if hasattr(expr, "free_symbols") else set()
        for s in syms:
            if str(s) in tile_var_set:
                return str(s)
        return None

    def _widen_boundary_connectors(self, parent_state: SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                                   spec: TileDimSpec) -> None:
        """Widen length-1 boundary connectors carrying a tile-var offset to ``(W,)`` tiles.

        The frontend lowers a single-element boundary access ``a[i]`` (the
        destination, or an extra elementwise input ``e[i]``) into a length-1
        NSDFG connector whose tile-var offset sits in the *outer* edge, with
        the inner access at ``[0]`` — unlike a ``vbor``-style full-array
        connector accessed at ``a[i]``. To tile such a body the connector is
        reshaped ``(1,) -> (W,)``, its outer edge grown to the tile region
        (``a[i] -> a[i:i+W]``), and its inner ``[0]`` memlets to the full tile
        ``[0:W]`` — after which it is a trivial contiguous full-tile
        load/store (see :meth:`_box_classification`). The gather index
        connector (already ``(W,)`` after the fan-out) and lane-invariant
        scalar inputs (outer edge with no tile var) are left untouched.

        K=1 only — the separable / K-D boundary cases land with the wider
        gather-descent slice.

        :param parent_state: State holding the map + NSDFG.
        :param nsdfg_node: The body NestedSDFG node.
        :param spec: Tile spec.
        """
        inner = nsdfg_node.sdfg
        W = tuple(spec.widths)
        tile_var_set = set(spec.iter_vars)
        tile_var_to_width = dict(zip(spec.iter_vars, W))
        full = subsets.Range([(0, w - 1, 1) for w in W])
        conn_edges = ([(e.dst_conn, e) for e in parent_state.in_edges(nsdfg_node)] +
                      [(e.src_conn, e) for e in parent_state.out_edges(nsdfg_node)])
        for conn, oe in conn_edges:
            if conn is None or conn not in inner.arrays or conn == _INNER_MASK:
                continue
            arr = inner.arrays[conn]
            if arr.transient or tuple(arr.shape) != (1, ) or oe.data is None or oe.data.subset is None:
                continue
            if not any(self._tilevar_in(b, tile_var_set) is not None for (b, _e, _s) in oe.data.subset):
                continue
            dtype = arr.dtype
            inner.remove_data(conn, validate=False)
            inner.add_array(conn, list(W), dtype, storage=dace.dtypes.StorageType.Register, transient=False)
            new_ranges = []
            for (b, e, s) in oe.data.subset:
                tvar = self._tilevar_in(b, tile_var_set)
                if tvar is not None:
                    new_ranges.append((b, b + (tile_var_to_width[tvar] - 1), 1))
                else:
                    new_ranges.append((b, e, s))
            oe.data.subset = subsets.Range(new_ranges)
            for istate in inner.states():
                for ed in istate.edges():
                    if ed.data is not None and ed.data.data == conn:
                        ed.data = dace.Memlet(data=conn, subset=subsets.Range(list(full.ranges)))

    def _box_classification(self, subset: subsets.Range, arr: dace.data.Data, iter_vars: Tuple[str, ...]):
        """Classify a connector access and require a perfect box.

        A register-tile boundary connector (shape exactly the tile widths,
        widened from a length-1 per-lane connector by
        :meth:`_widen_boundary_connectors`) carries its tile-var offset in
        the NSDFG's *outer* edge, so the inner access is the full tile and
        :func:`classify_tile_access` would see a tile-var-free subset
        (``BROADCAST_SYMBOL``). Treat it directly as a contiguous full-tile
        load/store.

        :param subset: Per-iteration subset on the connector array.
        :param arr: The connector array descriptor.
        :param iter_vars: Tile iter-vars.
        :returns: The :class:`TileAccessClassification`.
        :raises NotImplementedError: For non-box (gather/structured) access.
        """
        if tuple(arr.shape) == tuple(self.widths):
            K = len(self.widths)
            return TileAccessClassification(kind=TileAccessKind.CONTIGUOUS,
                                            dim_strides=(1, ) * K, match_dims=tuple(range(K)))
        cls = classify_tile_access(subset, tuple(arr.strides), iter_vars)
        if cls.kind not in _BOX_KINDS:
            raise NotImplementedError(
                f"PromoteNSDFGBodyToTiles: connector access {subset} is {cls.kind.value}; "
                f"only perfect-box (contiguous / strided) loads/stores are supported in this slice")
        return cls

    def _gather_index_symbols(self, subset: subsets.Range, tile_var_set: set) -> List[Tuple[int, str]]:
        """Find source dims indexed by a (non-tile-var) gather symbol.

        After the fan-out pass, a data gather reads ``src[__sym]`` where
        ``__sym`` is a point-access symbol that is NOT a tile iter-var (it is
        bound, via an interstate assignment, to a widened index tile element).

        :param subset: The source-access subset.
        :param tile_var_set: The tile iter-var names.
        :returns: ``[(source_dim, symbol_name), ...]`` for the gather dims.
        """
        out: List[Tuple[int, str]] = []
        for d, (b, e, _s) in enumerate(subset):
            if not hasattr(b, "free_symbols"):
                continue
            fs = {str(x) for x in b.free_symbols}
            if not fs or (fs & tile_var_set):
                continue
            try:
                is_point = bool(dace.symbolic.simplify(e - b) == 0)
            except Exception:
                is_point = False
            if is_point and len(fs) == 1 and str(b) in fs:
                out.append((d, next(iter(fs))))
        return out

    def _index_array_for_symbol(self, inner: dace.SDFG, sym: str) -> Optional[str]:
        """Return the widened index-tile array that a gather symbol reads.

        The fan-out rebinds ``__sym = idxc[0]`` (lane 0 of the widened ``(W,)``
        index tile). This extracts ``idxc``.

        :param inner: The body SDFG.
        :param sym: The gather index symbol name.
        :returns: The index-tile array name, or ``None`` if not resolvable.
        """
        for ie in inner.all_interstate_edges():
            if sym not in ie.data.assignments:
                continue
            # ``idxc[0]`` is a Subscript node, whose array head is reported by
            # ``arrays`` (not ``free_symbols_and_functions``); union both so a
            # bare ``idxc`` reference is also covered.
            expr = ie.data.assignments[sym]
            for name in set(dace.symbolic.arrays(expr)) | set(dace.symbolic.free_symbols_and_functions(expr)):
                if name in inner.arrays:
                    return name
        return None

    def _collapse_tile_gathers(self, istate: SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                               mask_acc: dace.nodes.AccessNode, spec: TileDimSpec) -> set:
        """Collapse a fanned-out data gather into a masked :class:`TileGather`.

        Mirrors the 1D ``DetectGather`` collapse: a ``src[__sym]`` read whose
        ``__sym`` was fanned out into a widened index tile becomes
        ``TileGather(src, idx_tile, mask) -> src_tile`` and the consumer is
        rerouted to ``src_tile``. The gather symbols are returned for the
        caller to simplify away.

        :param istate: Inner state being rewritten.
        :param nsdfg_node: The body NestedSDFG node.
        :param mask_acc: This state's inner mask access node.
        :param spec: Tile spec.
        :returns: The set of consumed gather base-symbol names.
        :raises NotImplementedError: For gather shapes this slice does not
            handle (mixed affine + data-gather source dims, unresolved index).
        """
        inner = istate.sdfg
        W = tuple(spec.widths)
        tile_var_set = set(spec.iter_vars)
        out_subset = ", ".join(f"0:{w}" for w in W)
        consumed: set = set()
        for tasklet in [n for n in istate.nodes() if isinstance(n, dace.nodes.Tasklet)]:
            for e in list(istate.in_edges(tasklet)):
                if not isinstance(e.src, dace.nodes.AccessNode):
                    continue
                src_name = e.src.data
                if (src_name not in nsdfg_node.in_connectors or inner.arrays[src_name].transient
                        or src_name == _INNER_MASK or e.data is None or e.data.subset is None):
                    continue
                gather_syms = self._gather_index_symbols(e.data.subset, tile_var_set)
                if not gather_syms:
                    continue
                src_arr = inner.arrays[src_name]
                src_ndim = len(src_arr.shape)
                if len(gather_syms) != src_ndim:
                    raise NotImplementedError(
                        f"PromoteNSDFGBodyToTiles: gather on {src_name!r} indexes {len(gather_syms)} of "
                        f"{src_ndim} source dims; mixed affine + data-gather source dims not yet supported")
                gather = TileGather(name=f"gather_{src_name}", widths=W, source_ndim=src_ndim, has_mask=True)
                istate.add_node(gather)
                istate.add_edge(e.src, None, gather, TileConnectors.SRC, dace.Memlet.from_array(src_name, src_arr))
                for k, (_dim, sym) in enumerate(gather_syms):
                    idx_arr = self._index_array_for_symbol(inner, sym)
                    if idx_arr is None:
                        raise NotImplementedError(
                            f"PromoteNSDFGBodyToTiles: gather index {sym!r} on {src_name!r} did not resolve "
                            f"to a widened index tile; run fan_out_tile_gather_index_symbols first")
                    istate.add_edge(istate.add_access(idx_arr), None, gather, TileConnectors.idx(k),
                                    dace.Memlet(f"{idx_arr}[{out_subset}]"))
                    consumed.add(sym)
                istate.add_edge(mask_acc, None, gather, TileConnectors.MASK,
                                dace.Memlet(f"{_INNER_MASK}[{out_subset}]"))
                tile_name = f"{src_name}_gather"
                suffix = 0
                while tile_name in inner.arrays:
                    suffix += 1
                    tile_name = f"{src_name}_gather_{suffix}"
                inner.add_array(tile_name, list(W), src_arr.dtype,
                                storage=dace.dtypes.StorageType.Register, transient=True)
                tile_acc = istate.add_access(tile_name)
                istate.add_edge(gather, TileConnectors.DST, tile_acc, None, dace.Memlet(f"{tile_name}[{out_subset}]"))
                istate.remove_edge(e)
                istate.add_edge(tile_acc, None, tasklet, e.dst_conn, dace.Memlet(f"{tile_name}[{out_subset}]"))
        return consumed

    def _drop_gather_symbols(self, inner: dace.SDFG, base_syms: set) -> None:
        """Simplify away the gather symbols consumed by :meth:`_collapse_tile_gathers`.

        Removes the base symbol and its per-lane ``_laneid_`` fan from every
        interstate edge, so no stray index symbol remains (the tile mirror of
        the 1D laneid simplification).

        :param inner: The body SDFG.
        :param base_syms: The consumed gather base-symbol names.
        """
        for ie in inner.all_interstate_edges():
            kept = {}
            for k, v in ie.data.assignments.items():
                parsed = LaneIdScheme.parse(k)
                base = parsed[0] if parsed is not None else k
                if base in base_syms:
                    continue
                kept[k] = v
            ie.data.assignments = kept
        # Drop the now-undefined gather symbols from the symbol registry so
        # they no longer count as free symbols requiring an outer mapping.
        for name in list(inner.symbols.keys()):
            parsed = LaneIdScheme.parse(name)
            base = parsed[0] if parsed is not None else name
            if base in base_syms:
                inner.remove_symbol(name)

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
            istate.add_edge(ed.src, ed.src_conn, load, TileConnectors.SRC, dace.Memlet(data=src_name, subset=promoted))
            istate.add_edge(mask_acc, None, load, TileConnectors.MASK, dace.Memlet(f"{_INNER_MASK}[{subset}]"))
            istate.add_edge(load, TileConnectors.DST, ed.dst, ed.dst_conn, dace.Memlet(f"{ed.dst.data}[{subset}]"))
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
                istate.add_edge(info_a, None, binop, TileConnectors.A, dace.Memlet(f"{info_a.data}[{subset}]"))
            if kind_b == "Tile":
                istate.add_edge(info_b, None, binop, TileConnectors.B, dace.Memlet(f"{info_b.data}[{subset}]"))
            istate.add_edge(mask_acc, None, binop, TileConnectors.MASK, dace.Memlet(f"{_INNER_MASK}[{subset}]"))
            istate.add_edge(binop, TileConnectors.C, out_access, out_edge.dst_conn, dace.Memlet(f"{out_access.data}[{subset}]"))
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
            istate.add_edge(src_access, None, store, TileConnectors.SRC, dace.Memlet(f"{src_access.data}[{subset}]"))
            istate.add_edge(mask_acc, None, store, TileConnectors.MASK, dace.Memlet(f"{_INNER_MASK}[{subset}]"))
            istate.add_edge(store, TileConnectors.DST, ed.dst, ed.dst_conn, dace.Memlet(data=dst_name, subset=promoted))
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
        # Gather descent (mirrors 1D expand -> DetectGather): fan a per-lane
        # gather index ``__sym = idx[i]`` into a widened ``(W,)`` index tile,
        # then collapse the ``src[__sym]`` reads into a TileGather and drop the
        # now-unused index symbols. K=1 only for now (the separable / K-D index
        # cases land in a later slice).
        if len(W) == 1:
            fan_out_tile_gather_index_symbols(inner, nsdfg_node, parent_state, W[0], spec.iter_vars[0])
            self._widen_boundary_connectors(parent_state, nsdfg_node, spec)
        self._reshape_transients(inner, W)
        consumed: set = set()
        for istate in inner.states():
            mask_acc = istate.add_access(_INNER_MASK)
            consumed |= self._collapse_tile_gathers(istate, nsdfg_node, mask_acc, spec)
            self._promote_loads(istate, nsdfg_node, mask_acc, spec)
            self._promote_binops(istate, mask_acc, spec)
            self._promote_stores(istate, nsdfg_node, mask_acc, spec)
            self._promote_internal_assigns(istate, nsdfg_node, spec)
            if istate.degree(mask_acc) == 0:
                istate.remove_node(mask_acc)
            self._drop_isolated(istate)
        if consumed:
            self._drop_gather_symbols(inner, consumed)

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
