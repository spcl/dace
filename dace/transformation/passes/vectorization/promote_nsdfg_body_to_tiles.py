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
import re
from typing import Dict, List, Optional, Set, Tuple

import dace
from dace import properties, subsets
from dace.sdfg.nodes import MapEntry
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl

from dace.libraries.tileops import TileBinop, TileGather, TileLoad, TileMerge, TileScatter, TileStore, TileUnop
from dace.libraries.tileops._pure_codegen import nested_loops, tile_offset
from dace.transformation.passes.vectorization.emit_tile_ops import (
    _classify_binop_tasklet_body,
    _classify_unop_tasklet_body,
    _constant_store_value,
    _is_assign_tasklet,
    _is_numeric_literal,
    _mask_name_for_map,
    _tile_region_subset,
)
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER, TILE_MAIN_MARKER)
from dace.transformation.passes.vectorization.utils.lane_expansion import (demote_non_index_symbols,
                                                                           fan_out_tile_gather_index_symbols)
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.fusion_inline import FuseStates
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


def _index_lane_stride(window_size, tile_width) -> int:
    """Recover the per-lane index stride ``c`` from a widened index window.

    The gather fan-out widens a strided index access ``idx[c*i]`` into a
    contiguous bounding window of ``c*(W-1)+1`` elements. This inverts that:
    ``c = (window_size - 1) // (W - 1)``. A window equal to ``W`` (or a
    degenerate/symbolic size) is the unit-stride contiguous case (``c = 1``).

    :param window_size: The widened index tile's element count.
    :param tile_width: The tile width ``W`` along the gathered dim.
    :returns: The per-lane index stride ``c`` (``1`` when contiguous).
    """
    try:
        window_size, tile_width = int(window_size), int(tile_width)
    except (TypeError, ValueError):
        return 1
    if tile_width <= 1 or window_size == tile_width:
        return 1
    return (window_size - 1) // (tile_width - 1)


# A branch-normalized same-write-set ``if/else`` lowers to a per-target
# ``_o = merge(_c, _t, _e)`` tasklet (see :mod:`dace.runtime.include.dace.merge`);
# recognised here so the descent can lower it to a :class:`TileMerge` per-lane select.
_MERGE_RE = re.compile(r"^\s*(?P<out>\w+)\s*=\s*merge\(\s*(?P<c>\w+)\s*,\s*(?P<t>\w+)\s*,\s*(?P<e>\w+)\s*\)\s*;?\s*$")


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

    def __init__(self, widths: Tuple[int, ...] = (8, )):
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

    def _wire_mask(self, istate: SDFGState, mask_acc: Optional[dace.nodes.AccessNode], node: dace.nodes.Node,
                   subset: str) -> None:
        """Wire the iteration mask into ``node``'s ``_mask`` connector, when the
        map is masked.

        A ``None`` ``mask_acc`` is the unmasked (``has_mask=False``) fast path
        a ``masked_tail`` split's divisible interior takes — no mask edge is
        added and the lib node was built with ``has_mask=False``.

        :param istate: Inner state being rewritten.
        :param mask_acc: This state's mask access node, or ``None`` if unmasked.
        :param node: The lib node to wire the mask into.
        :param subset: The tile subset string (``"0:W0, 0:W1"``).
        """
        if mask_acc is not None:
            istate.add_edge(mask_acc, None, node, TileConnectors.MASK, dace.Memlet(f"{_INNER_MASK}[{subset}]"))

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
            inner.add_array(_INNER_MASK,
                            list(widths),
                            dace.bool_,
                            storage=dace.dtypes.StorageType.Register,
                            transient=False)
        nsdfg_node.add_in_connector(_INNER_MASK)
        subset = ", ".join(f"0:{w}" for w in widths)
        mask_access = self._mask_access(parent_state, mask_name)
        parent_state.add_edge(mask_access, None, nsdfg_node, _INNER_MASK, dace.Memlet(f"{mask_name}[{subset}]"))

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
                inner.add_array(name, list(widths), dtype, storage=dace.dtypes.StorageType.Register, transient=True)
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

    def _inner_access_is_boundary_point(self, inner: dace.SDFG, conn: str, tile_var_set: Set[str]) -> bool:
        """Whether every inner access to ``conn`` is a tile-var-free single point.

        A boundary connector carries its per-tile offset in the NSDFG's *outer*
        edge and is read/written inside at a fixed point (``[0]``). This covers
        both the frontend's length-1 ``(1,)`` connector and the
        ``RefineNestedAccess``-canonicalised form (outer ``e[i]``, inner array
        still ``(N,)`` but accessed at ``[0]``). It deliberately excludes:

        * a ``vbor``-style full-array connector accessed inside at ``a[i]`` (the
          tile var lives in the inner subset) — left for :meth:`_box_classification`;
        * an already-widened ``(W,)`` index/gather tile (read as the range
          ``[0:W]``, not a point).

        :param inner: The body SDFG.
        :param conn: Connector array name.
        :param tile_var_set: Tile iter-var names.
        :returns: ``True`` iff ``conn`` is a boundary connector to widen.
        """
        seen = False
        for istate in inner.states():
            for ed in istate.edges():
                if ed.data is None or ed.data.data != conn or ed.data.subset is None:
                    continue
                seen = True
                for (b, e, _s) in ed.data.subset:
                    if self._tilevar_in(b, tile_var_set) is not None or self._tilevar_in(e, tile_var_set) is not None:
                        return False
                    try:
                        if dace.symbolic.simplify(e - b) != 0:
                            return False
                    except Exception:
                        return False
        return seen

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

        K-general. The widened connector carries the *source array's*
        strides for the tiled dims, so it is a correct strided view of the
        ``(W_0, ..., W_{K-1})`` block. For K=1 the single tiled dim is the
        contiguous (stride-1) axis so the view is contiguous; for K>=2 the
        outer tiled dim has a non-unit array stride (e.g. ``A[1, i:i+W, j:j+W]``
        has row stride ``N``), which the contiguous-tile pure expansions
        cannot address — so :meth:`_materialize_connector_reads` copies the
        strided view into a contiguous register tile via a stride-aware
        ``TileLoad`` before any binop reads it (outputs route back through a
        ``TileStore``).

        :param parent_state: State holding the map + NSDFG.
        :param nsdfg_node: The body NestedSDFG node.
        :param spec: Tile spec.
        """
        inner = nsdfg_node.sdfg
        W = tuple(spec.widths)
        tile_var_set = set(spec.iter_vars)
        tile_var_to_width = dict(zip(spec.iter_vars, W))
        conn_edges = ([(e.dst_conn, e)
                       for e in parent_state.in_edges(nsdfg_node)] + [(e.src_conn, e)
                                                                      for e in parent_state.out_edges(nsdfg_node)])
        for conn, oe in conn_edges:
            if conn is None or conn not in inner.arrays or conn == _INNER_MASK:
                continue
            arr = inner.arrays[conn]
            if arr.transient or oe.data is None or oe.data.subset is None:
                continue
            if not any(self._tilevar_in(b, tile_var_set) is not None for (b, _e, _s) in oe.data.subset):
                continue
            # Widen genuine boundary connectors only: the per-tile offset lives
            # in this outer edge and the inner access is a fixed point. An
            # already-widened ``(W,)`` tile (range inner access) or a vbor-style
            # full-array connector accessed inside at ``a[i]`` is left alone.
            if not self._inner_access_is_boundary_point(inner, conn, tile_var_set):
                continue
            dtype = arr.dtype
            # The source array (in the parent SDFG) supplies the per-dim strides;
            # index them by the *tiled* subset dims, in subset order, so the
            # widened connector is a faithful strided view of the tile block.
            src_arr = parent_state.sdfg.arrays[oe.data.data]
            tiled_widths: List[int] = []
            tiled_strides: List = []
            new_ranges = []
            for d, (b, e, s) in enumerate(oe.data.subset):
                tvar = self._tilevar_in(b, tile_var_set)
                if tvar is not None:
                    w = tile_var_to_width[tvar]
                    # Access stride = coefficient of the tile var in ``b``: lane
                    # ``l`` reads ``begin(tvar -> tvar + l) = b + c*l``. A
                    # contiguous access (``src[i]``) has ``c == 1``; a strided
                    # access (``src[2*i]``) has ``c == 2`` and the widened view
                    # must step by ``c`` over a ``c*(w-1)+1`` source window
                    # (``2*i .. 2*(i+w)``) — else it reads a too-narrow
                    # contiguous block. Recover ``c`` as the per-unit-increment
                    # difference of the affine begin; a non-affine begin
                    # (``i//2`` structured / data gather) collapses to ``c == 1``
                    # here and is handled by the gather path, not this box widen.
                    tvar_sym = dace.symbolic.pystr_to_symbolic(tvar)
                    c = dace.symbolic.simplify(b.subs(tvar_sym, tvar_sym + 1) - b)
                    # ``c`` is the access stride; it may be a constant (``2``) or
                    # a free symbol (``src[S*i]`` -> ``c = S``). A non-affine
                    # begin (``i//2``) leaves a ``c`` that still depends on a tile
                    # var — that is a structured/gather access, not a strided box,
                    # so fall back to the contiguous widen (handled by the gather
                    # path, not here).
                    c_syms = {str(x) for x in getattr(c, "free_symbols", set())}
                    if c_syms & tile_var_set:
                        c = 1
                    new_ranges.append((b, b + c * (w - 1), c))
                    tiled_widths.append(w)
                    tiled_strides.append(c * src_arr.strides[d])
                else:
                    new_ranges.append((b, e, s))
            inner.remove_data(conn, validate=False)
            inner.add_array(conn,
                            tiled_widths,
                            dtype,
                            strides=tiled_strides,
                            storage=dace.dtypes.StorageType.Register,
                            transient=False)
            # A symbolic source stride (e.g. ``N``) the connector view now
            # references must be defined inside the NSDFG; thread it through the
            # symbol mapping (identity) and the inner symbol table.
            for stride in tiled_strides:
                for fs in dace.symbolic.pystr_to_symbolic(str(stride)).free_symbols:
                    sname = str(fs)
                    if sname not in nsdfg_node.symbol_mapping:
                        nsdfg_node.symbol_mapping[sname] = dace.symbolic.pystr_to_symbolic(sname)
                    if sname not in inner.symbols:
                        inner.add_symbol(sname, parent_state.sdfg.symbols.get(sname, dace.int64))
            oe.data.subset = subsets.Range(new_ranges)
            full = subsets.Range([(0, w - 1, 1) for w in tiled_widths])
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
                                            dim_strides=(1, ) * K,
                                            match_dims=tuple(range(K)))
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
                # Resolve each gather index tile + its per-lane stride. The
                # fan-out widened the index connector to a contiguous bounding
                # window of ``c*(W-1)+1`` elements (``idx[c*i:c*i+c*(W-1)]``);
                # recover ``c`` from that window size so the gather reads
                # ``idx[c*l]`` per lane (a ``c``-strided index pick DaCe's
                # base-pointer NSDFG args cannot carry in the memlet itself).
                idx_info = []  # (k, sym, idx_arr, window_size, lane_stride)
                for k, (_dim, sym) in enumerate(gather_syms):
                    idx_arr = self._index_array_for_symbol(inner, sym)
                    if idx_arr is None:
                        raise NotImplementedError(
                            f"PromoteNSDFGBodyToTiles: gather index {sym!r} on {src_name!r} did not resolve "
                            f"to a widened index tile; run fan_out_tile_gather_index_symbols first")
                    window = inner.arrays[idx_arr].shape[0]
                    idx_info.append((k, sym, idx_arr, window, _index_lane_stride(window, W[0])))
                gather = TileGather(name=f"gather_{src_name}",
                                    widths=W,
                                    source_ndim=src_ndim,
                                    has_mask=mask_acc is not None,
                                    index_strides=tuple(stride for *_, stride in idx_info))
                istate.add_node(gather)
                istate.add_edge(e.src, None, gather, TileConnectors.SRC, dace.Memlet.from_array(src_name, src_arr))
                for k, sym, idx_arr, window, _stride in idx_info:
                    istate.add_edge(istate.add_access(idx_arr), None, gather, TileConnectors.idx(k),
                                    dace.Memlet(f"{idx_arr}[0:{window}]"))
                    consumed.add(sym)
                self._wire_mask(istate, mask_acc, gather, out_subset)
                tile_name = f"{src_name}_gather"
                suffix = 0
                while tile_name in inner.arrays:
                    suffix += 1
                    tile_name = f"{src_name}_gather_{suffix}"
                inner.add_array(tile_name,
                                list(W),
                                src_arr.dtype,
                                storage=dace.dtypes.StorageType.Register,
                                transient=True)
                tile_acc = istate.add_access(tile_name)
                istate.add_edge(gather, TileConnectors.DST, tile_acc, None, dace.Memlet(f"{tile_name}[{out_subset}]"))
                istate.remove_edge(e)
                istate.add_edge(tile_acc, None, tasklet, e.dst_conn, dace.Memlet(f"{tile_name}[{out_subset}]"))
        return consumed

    def _collapse_tile_scatters(self, istate: SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                                mask_acc: dace.nodes.AccessNode, spec: TileDimSpec) -> set:
        """Collapse a fanned-out data scatter into a masked :class:`TileScatter`.

        The write-side mirror of :meth:`_collapse_tile_gathers`: a ``dst[__sym]``
        store whose ``__sym`` was fanned out (by the same interstate-assignment
        fan-out — ``__sym = idx[i]``) into a widened ``(W,)`` index tile becomes
        ``TileScatter(value_tile, idx_tile, mask) -> dst``. The value reaches the
        store either directly from a ``(W,)`` tile access node or through a
        trivial assign tasklet (folded away). The scatter index symbols are
        returned so the caller can simplify them away.

        :param istate: Inner state being rewritten.
        :param nsdfg_node: The body NestedSDFG node.
        :param mask_acc: This state's inner mask access node.
        :param spec: Tile spec.
        :returns: The consumed scatter base-symbol names.
        :raises NotImplementedError: For scatter shapes this slice does not
            handle (mixed affine + data-scatter dest dims, strided index,
            unresolved index, non-tile value source).
        """
        inner = istate.sdfg
        W = tuple(spec.widths)
        tile_var_set = set(spec.iter_vars)
        out_subset = ", ".join(f"0:{w}" for w in W)
        consumed: set = set()
        for e in list(istate.edges()):
            if not (isinstance(e.dst, dace.nodes.AccessNode) and e.dst.data in nsdfg_node.out_connectors):
                continue
            if (e.data is None or e.data.subset is None or e.dst.data == _INNER_MASK
                    or inner.arrays[e.dst.data].transient):
                continue
            dst_name = e.dst.data
            scatter_syms = self._gather_index_symbols(e.data.subset, tile_var_set)
            if not scatter_syms:
                continue  # a perfect-box store; _promote_stores handles it
            # The value to scatter is a ``(W, ...)`` tile, reached directly or
            # through a trivial ``_out = _in`` assign tasklet (folded away).
            if isinstance(e.src, dace.nodes.Tasklet) and _is_assign_tasklet(e.src):
                assign = e.src
                value_access = istate.in_edges(assign)[0].src
                to_remove = assign
            elif isinstance(e.src, dace.nodes.AccessNode):
                value_access = e.src
                to_remove = None
            else:
                raise NotImplementedError(
                    f"PromoteNSDFGBodyToTiles: scatter into {dst_name!r} from {type(e.src).__name__} "
                    f"unsupported (expected a tile value access node or assign tasklet)")
            if not isinstance(value_access, dace.nodes.AccessNode):
                raise NotImplementedError(
                    f"PromoteNSDFGBodyToTiles: scatter value into {dst_name!r} is not an access node")
            dst_arr = inner.arrays[dst_name]
            dst_ndim = len(dst_arr.shape)
            if len(scatter_syms) != dst_ndim:
                raise NotImplementedError(
                    f"PromoteNSDFGBodyToTiles: scatter on {dst_name!r} indexes {len(scatter_syms)} of "
                    f"{dst_ndim} dest dims; mixed affine + data-scatter dest dims not yet supported")
            idx_info = []  # (k, sym, idx_arr, window)
            for k, (_dim, sym) in enumerate(scatter_syms):
                idx_arr = self._index_array_for_symbol(inner, sym)
                if idx_arr is None:
                    raise NotImplementedError(
                        f"PromoteNSDFGBodyToTiles: scatter index {sym!r} on {dst_name!r} did not resolve "
                        f"to a widened index tile; run fan_out_tile_gather_index_symbols first")
                window = inner.arrays[idx_arr].shape[0]
                if _index_lane_stride(window, W[0]) != 1:
                    raise NotImplementedError(
                        f"PromoteNSDFGBodyToTiles: strided scatter index on {dst_name!r} (window {window}, "
                        f"W {W[0]}) not yet supported")
                idx_info.append((k, sym, idx_arr, window))
            scatter = TileScatter(name=f"scatter_{dst_name}",
                                  widths=W,
                                  dest_ndim=dst_ndim,
                                  has_mask=mask_acc is not None)
            istate.add_node(scatter)
            istate.add_edge(value_access, None, scatter, TileConnectors.SRC,
                            dace.Memlet(f"{value_access.data}[{out_subset}]"))
            for k, sym, idx_arr, window in idx_info:
                istate.add_edge(istate.add_access(idx_arr), None, scatter, TileConnectors.idx(k),
                                dace.Memlet(f"{idx_arr}[0:{window}]"))
                consumed.add(sym)
            self._wire_mask(istate, mask_acc, scatter, out_subset)
            istate.add_edge(scatter, TileConnectors.DST, e.dst, e.dst_conn, dace.Memlet.from_array(dst_name, dst_arr))
            istate.remove_edge(e)
            if to_remove is not None and istate.degree(to_remove) == 0:
                istate.remove_node(to_remove)
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

    def _promote_loads(self, istate: SDFGState, nsdfg_node: dace.nodes.NestedSDFG, mask_acc: dace.nodes.AccessNode,
                       spec: TileDimSpec) -> None:
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
            load = TileLoad(name=f"load_{src_name}",
                            widths=W,
                            dim_strides=tuple(cls.dim_strides),
                            src_dims=tuple(cls.match_dims),
                            has_mask=mask_acc is not None)
            istate.add_node(load)
            istate.add_edge(ed.src, ed.src_conn, load, TileConnectors.SRC, dace.Memlet(data=src_name, subset=promoted))
            self._wire_mask(istate, mask_acc, load, subset)
            istate.add_edge(load, TileConnectors.DST, ed.dst, ed.dst_conn, dace.Memlet(f"{ed.dst.data}[{subset}]"))
            istate.remove_edge(ed)

    def _promote_binops(self, istate: SDFGState, nsdfg_node: dace.nodes.NestedSDFG, mask_acc: dace.nodes.AccessNode,
                        spec: TileDimSpec) -> None:
        """Replace split binop tasklets with :class:`TileBinop`.

        An operand reading a length-1 / ``dace.data.Scalar`` connector (a
        loop-invariant value such as a non-tile scalar kernel arg, e.g.
        ``c`` in ``B[i, j] > c``) is wired as a ``Scalar`` operand that
        ``TileBinop`` broadcasts to every lane; a numeric literal is a
        ``Symbol``; everything else is a ``Tile``.

        :param istate: Inner state being rewritten.
        :param nsdfg_node: The body NestedSDFG node (for connector names).
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
                src = ie[0].src
                src_desc = inner.arrays[src.data]
                # A length-1 / Scalar connector (not a reshaped tile) is a
                # loop-invariant broadcast operand.
                is_scalar = (src.data in nsdfg_node.in_connectors and not src_desc.transient
                             and (isinstance(src_desc, dace.data.Scalar) or tuple(src_desc.shape) == (1, )))
                return ("Scalar", src) if is_scalar else ("Tile", src)

            kind_a, info_a = _operand(a_tok)
            kind_b, info_b = _operand(b_tok)
            if kind_a != "Tile" and kind_b != "Tile":
                raise NotImplementedError(
                    f"PromoteNSDFGBodyToTiles: binop {t.label!r} has no tile operand ({kind_a}/{kind_b})")
            kwargs = dict(name=f"{t.label}_binop",
                          widths=W,
                          op=op,
                          has_mask=mask_acc is not None,
                          kind_a=kind_a,
                          kind_b=kind_b)
            if kind_a == "Symbol":
                kwargs["expr_a"] = info_a
            if kind_b == "Symbol":
                kwargs["expr_b"] = info_b
            binop = TileBinop(**kwargs)
            istate.add_node(binop)

            def _wire(kind, info, conn):
                if kind == "Tile":
                    istate.add_edge(info, None, binop, conn, dace.Memlet(f"{info.data}[{subset}]"))
                elif kind == "Scalar":
                    istate.add_edge(info, None, binop, conn, dace.Memlet(f"{info.data}[0]"))
                # Symbol: nothing to wire (embedded inline).

            _wire(kind_a, info_a, TileConnectors.A)
            _wire(kind_b, info_b, TileConnectors.B)
            self._wire_mask(istate, mask_acc, binop, subset)
            write_access, write_conn = self._route_output(istate, out_access, out_edge.dst_conn, nsdfg_node, spec)
            istate.add_edge(binop, TileConnectors.C, write_access, write_conn,
                            dace.Memlet(f"{write_access.data}[{subset}]"))
            for e in list(istate.in_edges(t)) + list(istate.out_edges(t)):
                istate.remove_edge(e)
            istate.remove_node(t)

    def _promote_unops(self, istate: SDFGState, nsdfg_node: dace.nodes.NestedSDFG, mask_acc: dace.nodes.AccessNode,
                       spec: TileDimSpec) -> None:
        """Replace split unary tasklets with :class:`TileUnop`.

        The single-operand mirror of :meth:`_promote_binops`: recognises the
        function forms (``abs``/``exp``/``log``/``sqrt``/``sin``/``cos``/
        ``floor``/``ceil``/``tanh`` + dace-mangled shims) and unary minus via
        ``_classify_unop_tasklet_body``. Tasklets that already classify as a
        binop are left to :meth:`_promote_binops`. The operand is wired with the
        same kind logic — a numeric literal is an inline ``Symbol``, a length-1 /
        ``Scalar`` connector a broadcast ``Scalar``, everything else a ``Tile``.

        :param istate: Inner state being rewritten.
        :param nsdfg_node: The body NestedSDFG node (for connector names).
        :param mask_acc: This state's inner mask access node.
        :param spec: Tile spec.
        """
        inner = istate.sdfg
        W = tuple(spec.widths)
        subset = ", ".join(f"0:{w}" for w in W)
        unops = []
        for t in istate.nodes():
            if not isinstance(t, dace.nodes.Tasklet) or _is_assign_tasklet(t):
                continue
            # A binop is _promote_binops' job; only classify genuine unops here.
            if _classify_binop_tasklet_body(t) is not None:
                continue
            parsed = _classify_unop_tasklet_body(t)
            if parsed is not None:
                unops.append((t, parsed))
        for t, (out_conn, op, a_tok) in unops:
            out_edge = istate.out_edges(t)[0]
            out_access = out_edge.dst
            if _is_numeric_literal(a_tok):
                kind_a, info_a = "Symbol", a_tok
            else:
                ie = [e for e in istate.in_edges(t) if e.dst_conn == a_tok]
                if len(ie) != 1 or not isinstance(ie[0].src, dace.nodes.AccessNode):
                    raise NotImplementedError(
                        f"PromoteNSDFGBodyToTiles: unop {t.label!r} operand {a_tok!r} not a single tile read")
                src = ie[0].src
                src_desc = inner.arrays[src.data]
                is_scalar = (src.data in nsdfg_node.in_connectors and not src_desc.transient
                             and (isinstance(src_desc, dace.data.Scalar) or tuple(src_desc.shape) == (1, )))
                kind_a, info_a = ("Scalar", src) if is_scalar else ("Tile", src)
            kwargs = dict(name=f"{t.label}_unop",
                          widths=W,
                          op=op,
                          has_mask=mask_acc is not None,
                          kind_a=kind_a)
            if kind_a == "Symbol":
                kwargs["expr_a"] = info_a
            unop = TileUnop(**kwargs)
            istate.add_node(unop)
            if kind_a == "Tile":
                istate.add_edge(info_a, None, unop, TileConnectors.A, dace.Memlet(f"{info_a.data}[{subset}]"))
            elif kind_a == "Scalar":
                istate.add_edge(info_a, None, unop, TileConnectors.A, dace.Memlet(f"{info_a.data}[0]"))
            # Symbol: nothing to wire (embedded inline).
            self._wire_mask(istate, mask_acc, unop, subset)
            write_access, write_conn = self._route_output(istate, out_access, out_edge.dst_conn, nsdfg_node, spec)
            istate.add_edge(unop, TileConnectors.C, write_access, write_conn,
                            dace.Memlet(f"{write_access.data}[{subset}]"))
            for e in list(istate.in_edges(t)) + list(istate.out_edges(t)):
                istate.remove_edge(e)
            istate.remove_node(t)

    def _materialize_connector_reads(self, istate: SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                                     mask_acc: dace.nodes.AccessNode, spec: TileDimSpec) -> None:
        """Copy each widened (strided-view) boundary connector read directly by a
        compute tasklet into a contiguous register tile via a stride-aware masked
        :class:`TileLoad`, then reroute the reads to that tile.

        A K>=2 connector view (``A[1, i:i+W, j:j+W]``) has a non-unit outer array
        stride that the contiguous-offset binop / merge pure expansions cannot
        address; the ``TileLoad`` reads the view through its descriptor strides and
        writes a contiguous tile, which the binops then consume. A ``(1,)`` scalar
        connector is left alone (it stays a per-lane broadcast operand).

        :param istate: Inner state being rewritten.
        :param nsdfg_node: The body NestedSDFG node (for connector names).
        :param mask_acc: This state's inner mask access node.
        :param spec: Tile spec.
        """
        inner = istate.sdfg
        W = tuple(spec.widths)
        subset = ", ".join(f"0:{w}" for w in W)
        materialized: Dict[str, dace.nodes.AccessNode] = {}
        for tasklet in [n for n in istate.nodes() if isinstance(n, dace.nodes.Tasklet)]:
            for e in list(istate.in_edges(tasklet)):
                if not isinstance(e.src, dace.nodes.AccessNode):
                    continue
                cname = e.src.data
                desc = inner.arrays.get(cname)
                if desc is None or desc.transient or cname == _INNER_MASK:
                    continue
                if cname not in nsdfg_node.in_connectors and cname not in nsdfg_node.out_connectors:
                    continue
                if tuple(desc.shape) != W:
                    continue
                if cname not in materialized:
                    cls = self._box_classification(e.data.subset, desc, spec.iter_vars)
                    tile_name = f"{cname}_ld"
                    k = 0
                    while tile_name in inner.arrays:
                        k += 1
                        tile_name = f"{cname}_ld_{k}"
                    inner.add_array(tile_name,
                                    list(W),
                                    desc.dtype,
                                    storage=dace.dtypes.StorageType.Register,
                                    transient=True)
                    load = TileLoad(name=f"load_{cname}",
                                    widths=W,
                                    dim_strides=tuple(cls.dim_strides),
                                    src_dims=tuple(cls.match_dims),
                                    has_mask=mask_acc is not None)
                    istate.add_node(load)
                    istate.add_edge(istate.add_access(cname), None, load, TileConnectors.SRC,
                                    dace.Memlet(f"{cname}[{subset}]"))
                    self._wire_mask(istate, mask_acc, load, subset)
                    tile_acc = istate.add_access(tile_name)
                    istate.add_edge(load, TileConnectors.DST, tile_acc, None, dace.Memlet(f"{tile_name}[{subset}]"))
                    materialized[cname] = tile_acc
                tile_acc = materialized[cname]
                istate.remove_edge(e)
                istate.add_edge(tile_acc, None, tasklet, e.dst_conn, dace.Memlet(f"{tile_acc.data}[{subset}]"))

    def _route_output(self, istate: SDFGState, out_access: dace.nodes.AccessNode, out_conn,
                      nsdfg_node: dace.nodes.NestedSDFG, spec: TileDimSpec):
        """Route a compute node's output to a widened out-connector through a
        contiguous register tile, so the (contiguous-addressing) lib node never
        writes a strided connector view directly.

        Returns the access node + connector the producer should write to; when the
        target is not a widened connector (a plain transient tile), it is returned
        unchanged. Otherwise a fresh contiguous tile is returned and a
        ``contig -> connector`` copy edge is left for :meth:`_promote_stores` to
        lower to a stride-aware :class:`TileStore`.

        :param istate: Inner state being rewritten.
        :param out_access: The producer's original output access node.
        :param out_conn: The producer's original output dst-connector.
        :param nsdfg_node: The body NestedSDFG node (for connector names).
        :param spec: Tile spec.
        :returns: ``(write_access, write_conn)`` the producer should target.
        """
        inner = istate.sdfg
        W = tuple(spec.widths)
        subset = ", ".join(f"0:{w}" for w in W)
        desc = inner.arrays[out_access.data]
        is_widened_conn = (out_access.data in nsdfg_node.out_connectors and not desc.transient
                           and tuple(desc.shape) == W)
        if not is_widened_conn:
            return out_access, out_conn
        tile_name = f"{out_access.data}_st"
        k = 0
        while tile_name in inner.arrays:
            k += 1
            tile_name = f"{out_access.data}_st_{k}"
        inner.add_array(tile_name, list(W), desc.dtype, storage=dace.dtypes.StorageType.Register, transient=True)
        contig = istate.add_access(tile_name)
        istate.add_edge(contig, None, out_access, out_conn, dace.Memlet(f"{out_access.data}[{subset}]"))
        return contig, None

    def _promote_const_stores(self, istate: SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                              mask_acc: dace.nodes.AccessNode, spec: TileDimSpec) -> None:
        """Replace a constant-store tasklet ``__out = <literal>`` with a tile fill.

        Two shapes:

        * **Into a transient tile** (a branch arm-local ``_else_<arr>`` operand a
          :class:`TileMerge` consumes): a mask-gated nested-loop fill
          ``_out[lane] = mask[lane] ? <literal> : 0`` straight into the reshaped
          ``(W, ...)`` transient (the merge store discards OOB lanes, so the gate
          is just defensive).
        * **Into an output-connector array** (``a[i] = 3.0`` — the widened
          out-connector): fill a fresh contiguous transient tile UNMASKED, then
          route it through :meth:`_route_output` so :meth:`_promote_stores` lowers
          the connector write to a *masked* :class:`TileStore`. The mask must gate
          the memory write here — filling ``0`` into OOB lanes of a widened
          out-connector would write past the array's valid region.

        :param istate: Inner state being rewritten.
        :param nsdfg_node: The body NestedSDFG node (for connector names).
        :param mask_acc: This state's inner mask access node.
        :param spec: Tile spec.
        """
        inner = istate.sdfg
        W = tuple(spec.widths)
        subset = ", ".join(f"0:{w}" for w in W)
        off = tile_offset(list(W))
        const_stores = []
        for t in istate.nodes():
            if not isinstance(t, dace.nodes.Tasklet) or _is_assign_tasklet(t):
                continue
            if _classify_binop_tasklet_body(t) is not None:
                continue
            val = _constant_store_value(t)
            if val is None:
                continue
            out_edges = istate.out_edges(t)
            if len(out_edges) != 1 or not isinstance(out_edges[0].dst, dace.nodes.AccessNode):
                continue
            dst_desc = inner.arrays[out_edges[0].dst.data]
            is_transient = dst_desc.transient
            is_widened_conn = (out_edges[0].dst.data in nsdfg_node.out_connectors and not dst_desc.transient
                               and tuple(dst_desc.shape) == W)
            if not (is_transient or is_widened_conn):
                continue
            const_stores.append((t, val, out_edges[0], is_transient))
        for t, val, out_edge, is_transient in const_stores:
            out_access = out_edge.dst
            if is_transient:
                out_dtype = inner.arrays[out_access.data].dtype.ctype
                if mask_acc is not None:
                    body = f"_out[{off}] = _mask[{off}] ? {val} : {out_dtype}(0);"
                    inputs = {"_mask"}
                else:
                    body = f"_out[{off}] = {val};"
                    inputs = set()
                fill = istate.add_tasklet(name=f"const_{out_access.data}", inputs=inputs, outputs={"_out"},
                                          code=nested_loops(list(W), body), language=dace.dtypes.Language.CPP)
                self._wire_mask(istate, mask_acc, fill, subset)
                istate.add_edge(fill, "_out", out_access, out_edge.dst_conn,
                                dace.Memlet(f"{out_access.data}[{subset}]"))
            else:
                # Output-connector const store: fill a fresh contiguous transient
                # tile (every lane) and let _promote_stores mask the store.
                write_access, write_conn = self._route_output(istate, out_access, out_edge.dst_conn, nsdfg_node, spec)
                fill = istate.add_tasklet(name=f"const_{out_access.data}", inputs=set(), outputs={"_out"},
                                          code=nested_loops(list(W), f"_out[{off}] = {val};"),
                                          language=dace.dtypes.Language.CPP)
                istate.add_edge(fill, "_out", write_access, write_conn, dace.Memlet(f"{write_access.data}[{subset}]"))
            for e in list(istate.in_edges(t)) + list(istate.out_edges(t)):
                istate.remove_edge(e)
            istate.remove_node(t)

    def _promote_merges(self, istate: SDFGState, nsdfg_node: dace.nodes.NestedSDFG, mask_acc: dace.nodes.AccessNode,
                        spec: TileDimSpec) -> None:
        """Replace branch-normalized ``_o = merge(_c, _t, _e)`` tasklets with
        a masked :class:`TileMerge` per-lane select.

        The three operands are tile transients (the lifted condition tile and
        the two arm-local ``_then_<arr>`` / ``_else_<arr>`` tiles). The
        iteration mask gates the write so an out-of-bounds lane keeps the
        destination's zero-fill — the branch select itself is unconditional
        per lane (both arm values are already materialised). A write straight
        to a widened out-connector is routed through a contiguous tile +
        ``TileStore`` (see :meth:`_route_output`).

        :param istate: Inner state being rewritten.
        :param nsdfg_node: The body NestedSDFG node (for connector names).
        :param mask_acc: This state's inner mask access node.
        :param spec: Tile spec.
        """
        W = tuple(spec.widths)
        subset = ", ".join(f"0:{w}" for w in W)
        merges = []
        for t in istate.nodes():
            if not isinstance(t, dace.nodes.Tasklet):
                continue
            m = _MERGE_RE.match(t.code.as_string.strip())
            if m is not None:
                merges.append((t, m))
        for t, m in merges:
            out_edge = istate.out_edges(t)[0]
            out_access = out_edge.dst

            def _src_of(conn):
                ie = [e for e in istate.in_edges(t) if e.dst_conn == conn]
                if len(ie) != 1 or not isinstance(ie[0].src, dace.nodes.AccessNode):
                    raise NotImplementedError(
                        f"PromoteNSDFGBodyToTiles: merge {t.label!r} operand {conn!r} not a single tile read")
                return ie[0].src

            cond_src = _src_of(m.group("c"))
            then_src = _src_of(m.group("t"))
            else_src = _src_of(m.group("e"))
            merge = TileMerge(name=f"{t.label}_merge", widths=W, has_mask=mask_acc is not None)
            istate.add_node(merge)
            istate.add_edge(cond_src, None, merge, "_cond", dace.Memlet(f"{cond_src.data}[{subset}]"))
            istate.add_edge(then_src, None, merge, "_t", dace.Memlet(f"{then_src.data}[{subset}]"))
            istate.add_edge(else_src, None, merge, "_e", dace.Memlet(f"{else_src.data}[{subset}]"))
            self._wire_mask(istate, mask_acc, merge, subset)
            write_access, write_conn = self._route_output(istate, out_access, out_edge.dst_conn, nsdfg_node, spec)
            istate.add_edge(merge, "_o", write_access, write_conn, dace.Memlet(f"{write_access.data}[{subset}]"))
            for e in list(istate.in_edges(t)) + list(istate.out_edges(t)):
                istate.remove_edge(e)
            istate.remove_node(t)

    def _promote_stores(self, istate: SDFGState, nsdfg_node: dace.nodes.NestedSDFG, mask_acc: dace.nodes.AccessNode,
                        spec: TileDimSpec) -> None:
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
            e for e in istate.edges() if isinstance(e.dst, dace.nodes.AccessNode)
            and e.dst.data in nsdfg_node.out_connectors and e.data is not None and e.data.data is not None
        ]
        for ed in store_edges:
            # A tile lib node (TileBinop / TileMerge / TileLoad / TileGather)
            # already wrote the masked ``(W, ...)`` tile straight into the
            # widened out-connector; the NSDFG outer edge stores it to the
            # array region. No separate TileStore is needed (it would be an
            # identity copy of an already-tile-shaped connector).
            if isinstance(ed.src, dace.nodes.LibraryNode):
                continue
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
            store = TileStore(name=f"store_{dst_name}",
                              widths=W,
                              dim_strides=tuple(cls.dim_strides),
                              dst_dims=tuple(cls.match_dims),
                              has_mask=mask_acc is not None)
            istate.add_node(store)
            istate.add_edge(src_access, None, store, TileConnectors.SRC, dace.Memlet(f"{src_access.data}[{subset}]"))
            self._wire_mask(istate, mask_acc, store, subset)
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
            if isinstance(t, dace.nodes.Tasklet) and _is_assign_tasklet(t) and istate.in_edges(t)
            and istate.out_edges(t) and istate.out_edges(t)[0].dst.data not in nsdfg_node.out_connectors
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
                 spec: TileDimSpec, mask_name: Optional[str]) -> None:
        """Promote one flat body NSDFG to tile ops in place.

        :param parent_state: State holding the map + NSDFG.
        :param map_entry: Inner map entry.
        :param nsdfg_node: The body NestedSDFG node.
        :param spec: Tile spec.
        :param mask_name: Map-scope mask transient name, or ``None`` for an
            unmasked (``has_mask=False``) interior region (masked_tail split).
        """
        W = tuple(spec.widths)
        inner = nsdfg_node.sdfg
        # Reverse ScalarToSymbolPromotion on the tile body: demote every
        # interstate-edge symbol back to a scalar dataflow tasklet UNLESS it is
        # a gather/scatter index (used inside a memlet subset). A loop-dependent
        # compute symbol (``sym = A_slice[0] + B_slice[0]``) cannot be broadcast
        # and fanning it per lane explodes in K dims; demoting it lets the
        # standard TileLoad/TileBinop/TileStore promotion lower it tile-natively
        # (no ``_laneid_`` symbols). Runs before the gather fan-out so only the
        # genuine index symbols remain to fan, and before the mask threading so
        # the demoted dataflow is in place when the mask is wired in.
        if demote_non_index_symbols(inner):
            # Demotion appends a ``sym = expr`` assign state before each
            # consumer; fold them back into the body and split the (possibly
            # multi-op) demoted compute into single-op tasklets the binop /
            # store promotion expects.
            FuseStates().apply_pass(inner, {})
            SplitTasklets().apply_pass(inner, {})
        if mask_name is not None:
            self._thread_mask(parent_state, map_entry, nsdfg_node, mask_name, W)
        # Gather descent (mirrors 1D expand -> DetectGather): fan a per-lane
        # gather index ``__sym = idx[i]`` into a widened ``(W,)`` index tile,
        # then collapse the ``src[__sym]`` reads into a TileGather and drop the
        # now-unused index symbols. K=1 only for now (the separable / K-D index
        # cases land in a later slice).
        if len(W) == 1:
            fan_out_tile_gather_index_symbols(inner, nsdfg_node, parent_state, W[0], spec.iter_vars[0])
        # Boundary-connector widening is K-general: a length-1 connector whose
        # tile-var offset lives in the outer edge (``A[1, i, j]`` -> inner
        # ``[0]``) grows to a ``(W_0, ..., W_{K-1})`` tile, the outer edge to the
        # per-dim tile region, and the inner ``[0]`` memlets to the full tile.
        self._widen_boundary_connectors(parent_state, nsdfg_node, spec)
        self._reshape_transients(inner, W)
        consumed: set = set()
        for istate in inner.states():
            mask_acc = istate.add_access(_INNER_MASK) if mask_name is not None else None
            consumed |= self._collapse_tile_gathers(istate, nsdfg_node, mask_acc, spec)
            self._promote_loads(istate, nsdfg_node, mask_acc, spec)
            # Materialize strided-view connector reads into contiguous register
            # tiles before any binop/merge consumes them (K>=2 boundary loads).
            self._materialize_connector_reads(istate, nsdfg_node, mask_acc, spec)
            self._promote_const_stores(istate, nsdfg_node, mask_acc, spec)
            self._promote_binops(istate, nsdfg_node, mask_acc, spec)
            self._promote_unops(istate, nsdfg_node, mask_acc, spec)
            self._promote_merges(istate, nsdfg_node, mask_acc, spec)
            # Route a data-scatter store (``dst[idx[i]]`` — value tile already
            # produced by the binop/merge above) through a TileScatter before the
            # box-store promotion, which would otherwise reject the non-box dst.
            consumed |= self._collapse_tile_scatters(istate, nsdfg_node, mask_acc, spec)
            self._promote_stores(istate, nsdfg_node, mask_acc, spec)
            self._promote_internal_assigns(istate, nsdfg_node, spec)
            if mask_acc is not None and istate.degree(mask_acc) == 0:
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
            if n.map.label.endswith(SCALAR_TAIL_MARKER):  # scalar_postamble tail: keep scalar body
                continue
            if specs is not None and n not in specs:
                continue
            if len(n.map.params) < K:
                continue
            nsdfg = self._flat_body_nsdfg(g, n)
            if nsdfg is None:
                continue
            spec = specs[n] if specs is not None and n in specs else self._spec_for(n)
            mask_name = _mask_name_for_map(g, n)
            # A no-mask map is only legitimately unmasked when it is the
            # masked_tail split's provably-divisible interior (``__tile_main``).
            # An unmarked no-mask map means GenerateTileIterationMask was not
            # run; emitting has_mask=False there could be OOB-unsafe — refuse.
            if mask_name is None and not n.map.label.endswith(TILE_MAIN_MARKER):
                raise NotImplementedError(
                    f"PromoteNSDFGBodyToTiles: map {n.map.label!r} has no TileMaskGen in scope and "
                    f"is not a masked_tail interior (__tile_main); run GenerateTileIterationMask first.")
            self._promote(g, n, nsdfg, spec, mask_name)
            handled.add(n)
        return handled or None
