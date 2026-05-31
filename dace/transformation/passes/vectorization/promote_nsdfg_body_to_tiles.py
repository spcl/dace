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
import copy
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
    _lane_index_expr,
    _mask_name_for_map,
    _tile_region_subset,
)
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER, TILE_MAIN_MARKER)
from dace.transformation.passes.vectorization.utils.lane_expansion import (demote_non_index_symbols,
                                                                           fan_out_tile_gather_index_symbols,
                                                                           fan_out_tile_gather_index_symbols_kd)
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
        no compute tasklets, and the inner SDFG's top-level CFG either:

        - is fully flat (only :class:`SDFGState` nodes — the canonical
          NSDFG-body shape), OR
        - contains :class:`LoopRegion` nodes whose own bodies are
          themselves flat (only states). The outer map's tile vars stay
          in scope across the sequential inner loop, so the descent runs
          on the inner LoopRegion's body states transparently (each
          ``inner.states()`` walk is already recursive into LoopRegions).
          This is the TSVC carried-dep shape ``for i: for j: aa[j,i] =
          aa[j-1,i] + bb[j,i]`` after :class:`LoopToMap` (outer ``i``
          becomes a map, inner ``j`` stays a LoopRegion).

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
        from dace.sdfg.state import ConditionalBlock, LoopRegion
        for cfg in nsdfg.sdfg.nodes():
            if isinstance(cfg, SDFGState):
                continue
            if isinstance(cfg, LoopRegion) and all(isinstance(inner_cfg, SDFGState) for inner_cfg in cfg.nodes()):
                continue
            if isinstance(cfg, ConditionalBlock):
                # A ConditionalBlock at vectorisation time is a BUG in the
                # branch-normalisation upstream: ``EliminateBranches``
                # (fp_factor) and the merge-tasklet branch normalisation
                # must lower every conditional into straight-line arithmetic
                # (a ``c*x + (1-c)*y`` factor or a ``TileMerge`` select)
                # before the descent runs. Refuse loudly so the upstream
                # gap surfaces rather than silently skipping vectorisation.
                raise NotImplementedError(f"PromoteNSDFGBodyToTiles: NSDFG body still carries a "
                                          f"ConditionalBlock {cfg.label!r} after branch "
                                          f"normalisation. Every conditional must be lowered to "
                                          f"merge tasklets (branch_mode='merge') or FP-factor "
                                          f"arithmetic (branch_mode='fp_factor') before the "
                                          f"descent — surface this kernel as a branch-norm gap.")
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

        Two reshape sources:

        1. **Native length-1**: a ``(1,)``-shape transient minted by branch
           normalization / ScalarToSymbolPromotion (e.g. ``_cond___tmp0``);
           directly resize to ``(W, ...)``.
        2. **Oversized, single-point-access**: a transient whose descriptor
           is full-length (e.g. ``_then_b`` shape ``(LEN_1D,)``) but every
           memlet access is a single-element subset (post-``RefineNested
           Access`` ``[0]`` window-relative read). s273's
           branch normalization leaves the ``_then_<arr>`` arm-local
           transient at the source array's shape; the TileMerge expansion
           then takes it as a scalar (``double _t = _then_b[0]``) and the
           compiler rejects the mismatch with the pointer expected by
           ``tile_merge<T, T, W, ...>(_o, _cond, _t, _e, ...)``. Narrow to
           ``(1,)`` first so the rest of this method widens it to ``(W,)``.

        Rewrites the memlets of the reshaped arrays to the full tile region.

        :param inner: The body SDFG.
        :param widths: Tile widths.
        :returns: The set of reshaped array names.
        """
        # Phase 1: narrow oversized transients with single-point-only access
        # to ``(1,)`` so they fall into the widening path below.
        candidates: Set[str] = set()
        for name, arr in inner.arrays.items():
            if not arr.transient or tuple(arr.shape) == (1, ):
                continue
            if isinstance(arr, dace.data.View):
                continue
            candidates.add(name)
        if candidates:
            access_counts: Dict[str, int] = {n: 0 for n in candidates}
            all_point_access: Dict[str, bool] = {n: True for n in candidates}
            for istate in inner.states():
                for ed in istate.edges():
                    if ed.data is None or ed.data.data not in candidates:
                        continue
                    access_counts[ed.data.data] += 1
                    sub = ed.data.subset
                    try:
                        ne = dace.symbolic.simplify(sub.num_elements()) if sub is not None else None
                        if ne != 1:
                            all_point_access[ed.data.data] = False
                    except (TypeError, AttributeError):
                        all_point_access[ed.data.data] = False
            for name in list(candidates):
                if not all_point_access[name] or access_counts[name] == 0:
                    continue
                arr = inner.arrays[name]
                dtype = arr.dtype
                inner.remove_data(name, validate=False)
                inner.add_scalar(name, dtype, storage=dace.dtypes.StorageType.Register, transient=True)
        # Phase 2: native widening for every ``(1,)`` transient (now also
        # covers the narrowed ones from Phase 1).
        reshaped: Set[str] = set()
        for name, arr in list(inner.arrays.items()):
            if arr.transient and (isinstance(arr, dace.data.Scalar) or tuple(arr.shape) == (1, )):
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
        # Per widened connector, the tile-lane -> connector-dim permutation. The
        # connector is built in SUBSET (array) dim order, but the tile lanes
        # iterate in iter_vars order; for a transposed map (Fortran ``A[i,j]``
        # with iter_vars ``[j, i]``, the unit-stride dim ``i`` is the inner lane
        # but subset dim 0) these disagree. The tile ops read it back via
        # ``_box_classification`` (-> TileLoad/TileStore ``src_dims``/``dst_dims``)
        # so lane ``p`` addresses the right connector dim (no transpose).
        self._conn_match_dims: Dict[str, Tuple[int, ...]] = {}
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
            # A pure-affine non-box gather (diagonal / structured) was already
            # consumed by _collapse_affine_gathers (its outer edge is now the
            # tile-var-free full-array subset, so it never reaches here). A
            # non-box access still carrying a tile var is therefore a
            # DATA-dependent / mixed gather (``zqx[jo-1, 0, i]``, jo = iorder[i])
            # the descent does not yet handle — refuse cleanly instead of
            # box-widening it into an invalid (cross-product) connector.
            if classify_tile_access(oe.data.subset, tuple(src_arr.strides), spec.iter_vars).kind not in _BOX_KINDS:
                raise NotImplementedError(
                    f"PromoteNSDFGBodyToTiles: boundary connector {conn!r} access {oe.data.subset} is a "
                    f"data-dependent / mixed gather (not pure affine); not supported in the descent yet")
            tiled_widths: List[int] = []
            tiled_strides: List = []
            tile_w_per_dim: List[int] = []
            conn_itervar: List[str] = []
            new_ranges = []
            for d, (b, e, s) in enumerate(oe.data.subset):
                tvar = self._tilevar_in(b, tile_var_set)
                if tvar is not None:
                    w = tile_var_to_width[tvar]
                    conn_itervar.append(tvar)
                    # A tiled dim is either a POINT read (``src[i]`` / ``src[2*i]``,
                    # ``e == b``) or a per-iteration WINDOW (``A[i:i+3]`` stencil
                    # neighbour fan, ``e > b``). Both widen to a register tile the
                    # consumer reads as a (possibly shifted) W-subset:
                    #   * POINT: lane ``l`` reads ``begin(tvar -> tvar+l) = b+c*l``,
                    #     ``c`` = the access stride (per-unit-increment diff of the
                    #     affine begin: 1 contiguous, 2 strided, ``S`` symbolic; a
                    #     non-affine ``i//2`` keeps a tile var in ``c`` -> falls to
                    #     ``c = 1``, the gather path handled it). Window
                    #     ``[b : b+c*(w-1) : c]``, shape ``w``.
                    #   * WINDOW of extent ``E+1`` (``E = e-b``): output lane ``l``
                    #     reads ``[b+l : b+l+E]``, so the bounding window over all W
                    #     lanes is ``[b : b+E+(w-1)]``, shape ``E+w``; each inner
                    #     read at offset ``k`` becomes the W-subset ``[k : k+w-1]``
                    #     (the offset-preserving rewrite below). Stencils are
                    #     stride-1, so the window step is 1.
                    e_minus_b = dace.symbolic.simplify(e - b)
                    if e_minus_b != 0:
                        new_ranges.append((b, b + e_minus_b + (w - 1), 1))
                        tiled_widths.append(int(e_minus_b) + w)
                        tiled_strides.append(src_arr.strides[d])
                    else:
                        tvar_sym = dace.symbolic.pystr_to_symbolic(tvar)
                        c = dace.symbolic.simplify(b.subs(tvar_sym, tvar_sym + 1) - b)
                        c_syms = {str(x) for x in getattr(c, "free_symbols", set())}
                        if c_syms & tile_var_set:
                            c = 1
                        new_ranges.append((b, b + c * (w - 1), c))
                        tiled_widths.append(w)
                        tiled_strides.append(c * src_arr.strides[d])
                    tile_w_per_dim.append(w)
                else:
                    # A non-tiled dim must be a fixed point (extent 1). The
                    # multi-slice case (extent > 1) is split by
                    # :class:`SplitMultiSliceBoundaryConnectors` before Promote
                    # runs; if a multi-slice dim survives here, refuse — that
                    # split missed a case and a silent volume mismatch would
                    # corrupt reads.
                    try:
                        degenerate = bool(dace.symbolic.simplify(e - b) == 0)
                    except Exception:
                        degenerate = False
                    if not degenerate:
                        raise NotImplementedError(
                            f"PromoteNSDFGBodyToTiles: boundary connector {conn!r} has a non-tiled dim {d} of "
                            f"extent > 1 ({b}:{e}) — a multi-slice access; not supported in the descent yet")
                    new_ranges.append((b, e, s))
            inner.remove_data(conn, validate=False)
            inner.add_array(conn,
                            tiled_widths,
                            dtype,
                            strides=tiled_strides,
                            storage=dace.dtypes.StorageType.Register,
                            transient=False)
            # Record the tile-lane -> connector-dim permutation (only when the
            # connector covers every tile dim, i.e. conn_itervar is exactly the
            # iter-var set). ``match_dims[p]`` = the connector dim holding
            # ``iter_vars[p]``; identity for a C-order map, ``[1, 0]`` for a
            # transposed (Fortran) ``A[i, j]`` with iter_vars ``[j, i]``.
            if len(conn_itervar) == len(spec.iter_vars) and set(conn_itervar) == tile_var_set:
                self._conn_match_dims[conn] = tuple(conn_itervar.index(iv) for iv in spec.iter_vars)
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
                    if ed.data is None or ed.data.data != conn:
                        continue
                    old = ed.data.subset
                    # Offset-preserving rewrite: an inner read at point ``k`` in a
                    # tiled dim becomes the W-subset ``[k : k+w-1]`` of the widened
                    # window (so a stencil's ``A[0]``/``A[1]``/``A[2]`` reads become
                    # the shifted tiles ``A_win[0:W]``/``[1:W+1]``/``[2:W+2]``). Only
                    # when the inner subset already has exactly the tiled-dim count
                    # (no squeezed non-tiled dim to realign) — otherwise fall back
                    # to the full tile (the point-read shape, unchanged behaviour).
                    if old is not None and len(old) == len(tile_w_per_dim):
                        shifted = [(kb, kb + tile_w_per_dim[dp] - 1, 1) for dp, (kb, _ke, _ks) in enumerate(old)]
                        ed.data = dace.Memlet(data=conn, subset=subsets.Range(shifted))
                    else:
                        ed.data = dace.Memlet(data=conn, subset=subsets.Range(list(full.ranges)))

    def _box_classification(self,
                            subset: subsets.Range,
                            arr: dace.data.Data,
                            iter_vars: Tuple[str, ...],
                            conn_name: Optional[str] = None):
        """Classify a connector access and require a perfect box.

        A register-tile boundary connector (shape exactly the tile widths,
        widened from a length-1 per-lane connector by
        :meth:`_widen_boundary_connectors`) carries its tile-var offset in
        the NSDFG's *outer* edge, so the inner access is the full tile and
        :func:`classify_tile_access` would see a tile-var-free subset
        (``BROADCAST_SYMBOL``). Treat it directly as a contiguous full-tile
        load/store — using the widen's recorded tile-lane -> connector-dim
        permutation (``match_dims``) so a transposed (Fortran) map addresses the
        right dim per lane instead of assuming connector-order == lane-order.

        :param subset: Per-iteration subset on the connector array.
        :param arr: The connector array descriptor.
        :param iter_vars: Tile iter-vars.
        :param conn_name: Connector name, to look up the widen's recorded
            ``match_dims`` permutation (identity when absent).
        :returns: The :class:`TileAccessClassification`.
        :raises NotImplementedError: For non-box (gather/structured) access.
        """
        if tuple(arr.shape) == tuple(self.widths):
            K = len(self.widths)
            match = self._conn_match_dims.get(conn_name, tuple(range(K)))
            return TileAccessClassification(kind=TileAccessKind.CONTIGUOUS, dim_strides=(1, ) * K, match_dims=match)
        cls = classify_tile_access(subset, tuple(arr.strides), iter_vars)
        if cls.kind not in _BOX_KINDS:
            raise NotImplementedError(
                f"PromoteNSDFGBodyToTiles: connector access {subset} is {cls.kind.value}; "
                f"only perfect-box (contiguous / strided) loads/stores are supported in this slice")
        return cls

    def _fanned_symbols(self, inner: dace.SDFG) -> Set[str]:
        """Return base names of symbols whose ``_laneid_<i>`` fan exists in interstate edges.

        The fan-out pass mints ``<base>_laneid_<l>`` companions for every
        gather-index symbol it widens. The presence of even one such
        companion identifies ``<base>`` as a fanned-out gather index.

        :param inner: Body NSDFG to scan.
        :returns: Set of base symbol names (no ``_laneid_*`` suffix).
        """
        fanned: Set[str] = set()
        for ie in inner.all_interstate_edges():
            for name in ie.data.assignments.keys():
                parsed = LaneIdScheme.parse(name)
                if parsed is not None:
                    fanned.add(parsed[0])
        return fanned

    def _gather_index_symbols(self,
                              subset: subsets.Range,
                              tile_var_set: set,
                              inner: Optional[dace.SDFG] = None) -> List[Tuple[int, str]]:
        """Find source dims indexed by a (non-tile-var) gather symbol.

        After the fan-out pass, a data gather reads ``src[__sym]`` where
        ``__sym`` is a point-access symbol that is NOT a tile iter-var (it is
        bound, via an interstate assignment, to a widened index tile element).
        Two index-expression shapes are accepted:

        - **bare** ``src[k]`` — the begin expression is exactly the gather
          symbol.
        - **affine** ``src[affine(k, invariants...)]`` (s4114:
          ``c[LEN_1D - k - 1]``) — the begin expression contains EXACTLY ONE
          fanned-out symbol plus any number of loop-invariants. ``inner``
          is required to recognise the affine form (it carries the
          ``_laneid_`` fan that identifies which symbols are gather indices).

        :param subset: The source-access subset.
        :param tile_var_set: The tile iter-var names.
        :param inner: The body NSDFG (used to identify fanned symbols).
        :returns: ``[(source_dim, symbol_name), ...]`` for the gather dims.
        """
        fanned = self._fanned_symbols(inner) if inner is not None else set()
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
            if not is_point:
                continue
            # Bare-symbol shape (back-compat): ``c[k]``.
            if len(fs) == 1 and str(b) in fs:
                out.append((d, next(iter(fs))))
                continue
            # Affine shape: ``c[LEN_1D - k - 1]``. Accept iff EXACTLY ONE
            # symbol in the begin is a fanned-out gather symbol; the rest
            # are loop invariants the caller threads through to the fill
            # tasklet.
            if fanned:
                gather_in_fs = fs & fanned
                if len(gather_in_fs) == 1:
                    out.append((d, next(iter(gather_in_fs))))
        return out

    def _unrefine_minmax_connectors(self, parent_state: SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                                    spec: TileDimSpec) -> None:
        """Undo ``RefineNestedAccess``'s union-bound on a multi-read connector.

        A connector read at multiple distinct indices (s115's ``a`` read at
        ``[i]`` and ``[j]``) gets its outer subset refined to the minimum
        bounding range ``[Min(i, j) : Max(i, j) + 1]`` and the inner subsets
        offset by the same ``Min``. The descent's classify sees ``Min/Max`` as
        ``Gather`` and refuses; the body itself is fine — the refinement is a
        tight-allocation artifact. Restore the full source view: outer subset
        becomes ``[0 : src.shape[d] - 1]`` per dim, inner array reshapes to the
        full source shape, and each inner edge's subset has the original outer
        begin added back so a refined ``a[i - Min(i, j)]`` becomes the absolute
        ``a[i]`` (a per-lane tile read) and ``a[j - Min(i, j)]`` becomes
        ``a[j]`` (a loop-invariant scalar broadcast). The downstream widen +
        operand classification then handle both naturally.

        :param parent_state: State holding the map + NSDFG.
        :param nsdfg_node: The body NestedSDFG node.
        """
        inner = nsdfg_node.sdfg

        def _has_minmax(expr) -> bool:
            s = str(expr)
            return ("Min(" in s) or ("Max(" in s) or ("min(" in s) or ("max(" in s)

        conn_edges = ([(e.dst_conn, e)
                       for e in parent_state.in_edges(nsdfg_node)] + [(e.src_conn, e)
                                                                      for e in parent_state.out_edges(nsdfg_node)])
        seen: Set[str] = set()
        for conn, oe in conn_edges:
            if conn is None or conn in seen or conn == _INNER_MASK:
                continue
            if conn not in inner.arrays:
                continue
            if oe.data is None or oe.data.subset is None or oe.data.data is None:
                continue
            if not any(_has_minmax(b) or _has_minmax(e_) for (b, e_, _s) in oe.data.subset):
                continue
            src_arr = parent_state.sdfg.arrays[oe.data.data]
            # Capture the per-dim offset (the Min(...) value to add back).
            offsets = [b for (b, _e, _s) in oe.data.subset]
            old_desc = inner.arrays[conn]
            inner.remove_data(conn, validate=False)
            inner.add_array(conn,
                            list(src_arr.shape),
                            src_arr.dtype,
                            strides=src_arr.strides,
                            storage=old_desc.storage,
                            transient=False)
            for sym_expr in list(src_arr.shape) + list(src_arr.strides):
                for fs in dace.symbolic.pystr_to_symbolic(str(sym_expr)).free_symbols:
                    sname = str(fs)
                    if sname not in nsdfg_node.symbol_mapping:
                        nsdfg_node.symbol_mapping[sname] = dace.symbolic.pystr_to_symbolic(sname)
                    if sname not in inner.symbols:
                        inner.add_symbol(sname, parent_state.sdfg.symbols.get(sname, dace.int64))
            # Rebuild both in / out outer edges for this connector.
            for _conn, _oe in conn_edges:
                if _conn == conn and _oe.data is not None and _oe.data.data is not None:
                    _oe.data = dace.Memlet(data=_oe.data.data,
                                           subset=subsets.Range([(0, s - 1, 1) for s in src_arr.shape]))
            # Add the offset back to every inner edge subset for this connector
            # (cancel the refinement: ``[expr - Min(...)]`` -> ``[expr]``); for
            # a per-dim subset that is a single-element TILE-VAR-bearing point
            # (e.g. ``a[i]`` for the i-tile) ALSO expand to a per-tile W-window
            # ``[i : i + W - 1]`` so downstream materialisation lowers it as a
            # tile read; a tile-var-FREE single-element point (``a[j]``) stays a
            # point and lowers as a Scalar broadcast.
            tile_w = dict(zip(spec.iter_vars, spec.widths))
            for istate in inner.states():
                for ed in istate.edges():
                    if ed.data is None or ed.data.data != conn or ed.data.subset is None:
                        continue
                    new_sub = []
                    for d, (b, e_, s) in enumerate(ed.data.subset):
                        off = offsets[d]
                        nb, ne = b + off, e_ + off
                        # Per-dim W-window expansion: only when this dim is a
                        # single-element point (nb == ne) AND its begin has a
                        # tile var (non-tile-var points stay as broadcasts).
                        is_point = bool(dace.symbolic.simplify(ne - nb) == 0)
                        if is_point:
                            fs = {str(x) for x in getattr(nb, "free_symbols", set())}
                            tvars_in = fs & set(spec.iter_vars)
                            if len(tvars_in) == 1:
                                tv = next(iter(tvars_in))
                                w = tile_w[tv]
                                ne = nb + (w - 1)
                                s = 1
                        new_sub.append((nb, ne, s))
                    ed.data = dace.Memlet(data=conn, subset=subsets.Range(new_sub))
            seen.add(conn)

    def _collapse_affine_gathers(self, parent_state: SDFGState, nsdfg_node: dace.nodes.NestedSDFG, spec: TileDimSpec,
                                 mask_name: Optional[str]) -> None:
        """Route a non-box affine/structured boundary read to a :class:`TileGather`.

        A deterministic but non-box read — the diagonal ``A[i, i]`` (one tile var
        in two dims), a correlated affine ``A[2*i, i]``, or a structured
        ``c[i//2]`` (``int_floor``) — classifies as ``Gather`` / ``Structured``;
        box-widening it would read the wrong cross-product (an 8x8 block for the
        diagonal). Instead read the FULL source array through a ``TileGather``
        whose per-dim index tile is the affine lane index
        ``_gidx_k[l] = begin_k(tvar -> tvar + l)`` (the same affine substitution
        the flat :class:`EmitTileOps` gather path uses, via ``_lane_index_expr``).
        Runs before :meth:`_widen_boundary_connectors`, which then skips the
        connector (its outer edge is the tile-var-free full-array subset).

        :param parent_state: State holding the map + NSDFG.
        :param nsdfg_node: The body NestedSDFG node.
        :param spec: Tile spec.
        :param mask_name: Map-scope mask name, or ``None`` for an unmasked region.
        """
        inner = nsdfg_node.sdfg
        W = tuple(spec.widths)
        out_subset = ", ".join(f"0:{w}" for w in W)
        off = tile_offset(list(W))
        for oe in list(parent_state.in_edges(nsdfg_node)):
            conn = oe.dst_conn
            if conn is None or conn not in inner.arrays or conn == _INNER_MASK:
                continue
            arr = inner.arrays[conn]
            if arr.transient or oe.data is None or oe.data.subset is None or oe.data.data is None:
                continue
            src_name = oe.data.data
            src_arr = parent_state.sdfg.arrays[src_name]
            cls = classify_tile_access(oe.data.subset, tuple(src_arr.strides), spec.iter_vars)
            if cls.kind not in (TileAccessKind.GATHER, TileAccessKind.STRUCTURED):
                continue
            # Only a PURE affine/structured gather is handled here: every
            # non-constant index dim must be a function of the tile vars (the
            # lane substitution tvar -> tvar + l varies it). A tile-var-free
            # non-constant dim (``zqx[jo-1, 0, i]`` with ``jo = iorder[i]``) is a
            # DATA-dependent / mixed gather — leave it for the data-gather path /
            # a clean downstream refusal rather than emit a wrong per-lane-constant
            # index (which would build an invalid SDFG).
            if any(
                    getattr(b, "free_symbols", set()) and not ({str(x)
                                                                for x in b.free_symbols} & set(spec.iter_vars))
                    for (b, _e, _s) in oe.data.subset):
                continue
            src_ndim = len(src_arr.shape)
            # Per-dim affine lane-index expression (tvar -> tvar + __l_p).
            idx_exprs = []
            for k in range(src_ndim):
                expr = _lane_index_expr(str(oe.data.subset.ranges[k][0]), spec.iter_vars)
                if expr is None:
                    raise NotImplementedError(f"PromoteNSDFGBodyToTiles: affine gather on {src_name!r} dim {k} "
                                              f"({oe.data.subset.ranges[k][0]!r}) is not affine in the tile vars")
                idx_exprs.append(expr)
            # Expand the connector to the FULL source array so the gather can
            # index it; the outer edge becomes the tile-var-free full subset
            # (so _widen_boundary_connectors skips it).
            storage = arr.storage
            inner.remove_data(conn, validate=False)
            inner.add_array(conn,
                            list(src_arr.shape),
                            src_arr.dtype,
                            strides=src_arr.strides,
                            storage=storage,
                            transient=False)
            # The index-fill tasklets reference the tile vars (``i``), the source
            # shape/stride symbols (``N``), and any symbol in the index expression
            # itself (a symbolic divisor ``src[i // DV]``); EmitTileOps builds
            # these in the map scope where they are defined, but the descent
            # builds them inside the NSDFG body, so thread every such symbol into
            # the NSDFG's symbol mapping (identity) and inner symbol table. The
            # per-lane loop vars ``__l*`` are the only non-symbol tokens and are
            # already in scope (the nested-loop counters).
            shape_syms = set()
            for sym_expr in list(src_arr.shape) + list(src_arr.strides):
                shape_syms |= {str(x) for x in dace.symbolic.pystr_to_symbolic(str(sym_expr)).free_symbols}
            idx_syms = set()
            for expr in idx_exprs:
                idx_syms |= {str(x) for x in dace.symbolic.pystr_to_symbolic(expr).free_symbols}
            idx_syms = {s for s in idx_syms if not s.startswith("__l")}
            for sname in shape_syms | idx_syms | set(spec.iter_vars):
                if sname not in nsdfg_node.symbol_mapping:
                    nsdfg_node.symbol_mapping[sname] = dace.symbolic.pystr_to_symbolic(sname)
                if sname not in inner.symbols:
                    inner.add_symbol(sname, parent_state.sdfg.symbols.get(sname, dace.int64))
            oe.data = dace.Memlet(data=src_name, subset=subsets.Range([(0, s - 1, 1) for s in src_arr.shape]))
            full = ", ".join(f"0:{s}" for s in src_arr.shape)
            for istate in inner.states():
                for cnode in [n for n in istate.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == conn]:
                    consumers = list(istate.out_edges(cnode))
                    if not consumers:
                        continue
                    mask_acc = istate.add_access(_INNER_MASK) if mask_name is not None else None
                    idx_accs = []
                    for k, expr in enumerate(idx_exprs):
                        iname = f"_agidx_{conn}_{k}"
                        suffix = 0
                        while iname in inner.arrays:
                            suffix += 1
                            iname = f"_agidx_{conn}_{k}_{suffix}"
                        inner.add_array(iname,
                                        list(W),
                                        dace.int64,
                                        storage=dace.dtypes.StorageType.Register,
                                        transient=True)
                        fill = istate.add_tasklet(name=f"agidx_{iname}",
                                                  inputs=set(),
                                                  outputs={"_out"},
                                                  code=nested_loops(list(W), f"_out[{off}] = {expr};"),
                                                  language=dace.dtypes.Language.CPP)
                        iacc = istate.add_access(iname)
                        istate.add_edge(fill, "_out", iacc, None, dace.Memlet(f"{iname}[{out_subset}]"))
                        idx_accs.append(iacc)
                    gather = TileGather(name=f"agather_{conn}",
                                        widths=W,
                                        source_ndim=src_ndim,
                                        has_mask=mask_acc is not None)
                    istate.add_node(gather)
                    istate.add_edge(cnode, None, gather, TileConnectors.SRC,
                                    dace.Memlet(data=conn, subset=subsets.Range.from_string(full)))
                    for k, iacc in enumerate(idx_accs):
                        istate.add_edge(iacc, None, gather, TileConnectors.idx(k),
                                        dace.Memlet(f"{iacc.data}[{out_subset}]"))
                    if mask_acc is not None:
                        istate.add_edge(mask_acc, None, gather, TileConnectors.MASK,
                                        dace.Memlet(f"{_INNER_MASK}[{out_subset}]"))
                    gtile = f"{conn}_agather"
                    suffix = 0
                    while gtile in inner.arrays:
                        suffix += 1
                        gtile = f"{conn}_agather_{suffix}"
                    inner.add_array(gtile,
                                    list(W),
                                    src_arr.dtype,
                                    storage=dace.dtypes.StorageType.Register,
                                    transient=True)
                    gtacc = istate.add_access(gtile)
                    istate.add_edge(gather, TileConnectors.DST, gtacc, None, dace.Memlet(f"{gtile}[{out_subset}]"))
                    for ce in consumers:
                        istate.add_edge(gtacc, None, ce.dst, ce.dst_conn, dace.Memlet(f"{gtile}[{out_subset}]"))
                        istate.remove_edge(ce)

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

    def _index_expr_for_symbol(self, inner: dace.SDFG, sym: str) -> Optional[str]:
        """Return the right-hand-side string of ``sym``'s interstate assignment.

        Companion to :meth:`_index_array_for_symbol` — that returns the array
        head, this returns the verbatim ``ie.data.assignments[sym]`` string so
        the caller can build per-lane access expressions (e.g., the multi-dim
        source-array gather builds ``edge_blk[0, __l0, 0]`` from the
        ``edge_blk[0, 0, 0]`` binding by substituting the tile-var-direction
        index).

        :param inner: The body SDFG.
        :param sym: The gather index symbol name.
        :returns: The raw assignment RHS, or ``None`` if not found.
        """
        for ie in inner.all_interstate_edges():
            if sym in ie.data.assignments:
                return ie.data.assignments[sym]
        return None

    def _materialize_loop_invariant_idx(self, inner: dace.SDFG, istate: SDFGState, idx_arr: str,
                                        nsdfg_node: dace.nodes.NestedSDFG) -> str:
        """Copy a const non-transient ``(1,)`` boundary connector into a fresh transient.

        A loop-invariant scalar gather/scatter index sourced from an outer
        ``const T* __restrict__`` connector triggers a duplicate
        ``__restrict__`` qualifier in the generated tasklet body when wired
        directly as a ``TileGather`` / ``TileScatter`` ``_idx_<k>`` input
        (the inner tasklet codegen redeclares the pointer with another
        ``__restrict__``, which gcc rejects). Materialize it into a fresh
        ``(1,)`` transient via a CPP copy tasklet so the lib-node connector
        is fed by a non-const transient and codegen emits a single
        ``__restrict__``. Idempotent: if ``idx_arr`` is already transient
        or not a ``(1,)`` const, returns it unchanged.

        :param inner: The body SDFG.
        :param istate: The state in which the gather/scatter is being emitted.
        :param idx_arr: Candidate index source array name.
        :param nsdfg_node: The body NestedSDFG node (to check connector-ness).
        :returns: Either ``idx_arr`` (no materialization needed) or the name
            of a fresh transient holding the same scalar value.
        """
        desc = inner.arrays.get(idx_arr)
        if desc is None or desc.transient or tuple(desc.shape) != (1, ):
            return idx_arr
        if idx_arr not in nsdfg_node.in_connectors:
            return idx_arr
        # Mint a fresh transient of the same dtype + (1,) shape.
        mat_name = f"{idx_arr}_lc"
        suffix = 0
        while mat_name in inner.arrays:
            suffix += 1
            mat_name = f"{idx_arr}_lc_{suffix}"
        inner.add_array(mat_name, [1], desc.dtype, storage=dace.dtypes.StorageType.Register, transient=True)
        # Emit a CPP copy tasklet (Language.CPP needed because the connector
        # is a non-trivial pointer-typed input; Python tasklet codegen would
        # not be able to type-infer the connector).
        copy_t = istate.add_tasklet(
            name=f"copy_lc_{mat_name}",
            inputs={"_in"},
            outputs={"_out"},
            code="_out[0] = _in[0];",
            language=dace.dtypes.Language.CPP,
        )
        in_acc = istate.add_access(idx_arr)
        out_acc = istate.add_access(mat_name)
        istate.add_edge(in_acc, None, copy_t, "_in", dace.Memlet(f"{idx_arr}[0]"))
        istate.add_edge(copy_t, "_out", out_acc, None, dace.Memlet(f"{mat_name}[0]"))
        return mat_name

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
                gather_syms = self._gather_index_symbols(e.data.subset, tile_var_set, inner)
                if not gather_syms:
                    continue
                src_arr = inner.arrays[src_name]
                src_ndim = len(src_arr.shape)
                # K=1 mixed: resolve every source dim — data-gather dims via the
                # fanned-out widened index tile, affine dims (tile-var-only) via
                # a fresh affine ``_agidx`` tile built per-tile. The common
                # CloudSC pattern ``zsolqb[i, k, idx[j]]`` (two affine + one
                # data-gather) lands here at K=1 after collapse.
                sym_by_dim = {d: sym for (d, sym) in gather_syms}
                idx_info = []  # (k, sym_or_None, idx_arr, window_size, lane_stride)
                unresolved = False
                affine_dims = []  # (k, begin_expr) for tile-var affine fills
                # (k, sym, idx_arr, begin_expr_with_sym_placeholder) for
                # affine-of-gather-symbol fills. Substitute the gather symbol
                # for the ``__l0``-indexed read of the widened index tile.
                gather_affine_dims = []
                # (k, sym, idx_arr, idx_expr_str, tile_var_dim_in_inner_array)
                # for multi-dim source-array gather indices (icon's
                # ``edge_blk_index = edge_blk[jb, jc, 0]`` where ``edge_blk``
                # is the full 3D source, not a widened 1D index tile). The
                # inner read is window-relative (``edge_blk[0, 0, 0]`` is the
                # window origin); the fill substitutes ``0`` -> ``__l0`` in
                # the dim that the OUTER memlet binds to the tile var.
                multidim_gather_dims = []
                parent_state = nsdfg_node.sdfg.parent
                outer_subsets_by_conn = {}
                for oe in parent_state.in_edges(nsdfg_node):
                    if oe.dst_conn is not None and oe.data is not None and oe.data.subset is not None:
                        outer_subsets_by_conn[oe.dst_conn] = oe.data.subset
                for k in range(src_ndim):
                    if k in sym_by_dim:
                        sym = sym_by_dim[k]
                        idx_arr = self._index_array_for_symbol(inner, sym)
                        if idx_arr is None:
                            # ``sym`` is not a fanned-out data-dependent gather
                            # index (no widened index tile bound via interstate
                            # assignment) — it is a loop-invariant outer symbol
                            # (s115's ``a[j]``). Skip this edge; the downstream
                            # operand classifier handles it as Scalar broadcast.
                            unresolved = True
                            break
                        b, _e, _s = e.data.subset.ranges[k]
                        idx_arr_shape = inner.arrays[idx_arr].shape
                        idx_arr_ndim = len(idx_arr_shape)
                        if idx_arr_ndim > 1:
                            # Multi-dim source array (icon ``edge_blk`` shape
                            # ``(NB, NPROMA, 3)``; ``B[i, j]`` 2-D index for
                            # ``A[i, B[i,j]]``): for each tile iter-var, find
                            # which inner-array dim the outer memlet binds it
                            # to, then substitute that dim's value in the
                            # assignment expression with ``__l<p>``. K-aware:
                            # builds a per-iter-var map so K>=2 lanes resolve
                            # to the right inner-array dim per tile var.
                            outer_sub = outer_subsets_by_conn.get(idx_arr)
                            tile_var_dim_per_iter: Dict[int, int] = {}
                            if outer_sub is not None and len(outer_sub.ranges) == idx_arr_ndim:
                                for p, tv_name in enumerate(spec.iter_vars):
                                    tv_sym = dace.symbolic.pystr_to_symbolic(tv_name)
                                    for d, (ob, _oe, _os) in enumerate(outer_sub.ranges):
                                        ob_syms = ob.free_symbols if hasattr(ob, 'free_symbols') else set()
                                        if tv_sym in ob_syms:
                                            tile_var_dim_per_iter[p] = d
                                            break
                            if not tile_var_dim_per_iter:
                                unresolved = True
                                break
                            idx_expr_str = self._index_expr_for_symbol(inner, sym)
                            if idx_expr_str is None:
                                unresolved = True
                                break
                            multidim_gather_dims.append((k, sym, idx_arr, idx_expr_str, tile_var_dim_per_iter))
                            idx_info.append((k, None, None, W[0], 1))
                            continue
                        # 1-D index source: at K=1, ``fan_out_tile_gather_index_symbols``
                        # has widened ``idx_arr`` to a ``(W,)`` tile, so the per-
                        # lane read is ``idx_arr[l]``. At K>=2 the fan-out does
                        # NOT widen a ``(1,)`` const boundary connector (it widens
                        # the K-aware fan-out's multi-dim case instead). When the
                        # symbol's interstate-edge assignment carries NO tile var
                        # (``__sym = c + 1`` with ``c`` a loop-invariant outer
                        # scalar param), the index value is the SAME across all
                        # lanes — route through a constant-fill ``_agidx`` tile
                        # (``_agidx[off] = sym_expr``). This is the K-aware
                        # generalisation of the K=1 "Scalar broadcast" fall-back.
                        if len(W) > 1:
                            raise NotImplementedError(
                                f"PromoteNSDFGBodyToTiles: gather on {src_name!r} dim {k} reads 1-D index "
                                f"tile {idx_arr!r} but K={len(W)}; the K-shape index fan-out is a separate "
                                f"slice — use a multi-dim index source ({idx_arr}[i, j, ...]) at K>=2.")
                        if str(b) == sym:
                            # Bare-symbol form ``src[k]`` — use idx_arr directly.
                            window = idx_arr_shape[0]
                            idx_info.append((k, sym, idx_arr, window, _index_lane_stride(window, W[0])))
                        else:
                            # Affine-of-gather form ``src[affine(k, invariants...)]``
                            # (s4114: ``c[LEN_1D - k - 1]``). Build a fresh
                            # ``_agidx`` tile whose lane ``l`` holds
                            # ``begin.subs(sym, idx_arr[l])``.
                            gather_affine_dims.append((k, sym, idx_arr, b))
                            idx_info.append((k, None, None, W[0], 1))
                    else:
                        b, _e, _s = e.data.subset.ranges[k]
                        expr = _lane_index_expr(str(b), spec.iter_vars)
                        if expr is None:
                            unresolved = True
                            break
                        affine_dims.append((k, expr))
                        # placeholder; filled below once the fresh tile lands in
                        # the inner SDFG (contiguous stride=1).
                        idx_info.append((k, None, None, W[0], 1))
                if unresolved:
                    continue
                # Per-dim agidx access nodes produced by the fill tasklets.
                # The gather wiring loop reuses these instead of minting a
                # fresh ``add_access(idx_arr)`` so the fill -> agidx -> gather
                # dataflow is one connected path; otherwise the topological
                # sort can place the gather BEFORE the fill (the gather reads
                # uninitialised memory — segfault on the gathered ``c[...]``).
                agidx_access_by_dim: Dict[int, dace.nodes.AccessNode] = {}
                if affine_dims or gather_affine_dims or multidim_gather_dims:
                    off = tile_offset(list(W))
                    # Thread every symbol the affine expressions reference
                    # into the NSDFG (the lane fill tasklets reference the
                    # tile vars plus any shape / divisor symbols, and for the
                    # gather-affine case, any loop-invariant symbols inside
                    # the affine transform — e.g. ``LEN_1D`` in s4114's
                    # ``LEN_1D - k - 1``).
                    needed_syms = set(spec.iter_vars)
                    for (_k, expr) in affine_dims:
                        needed_syms |= {
                            s
                            for s in (str(x) for x in dace.symbolic.pystr_to_symbolic(expr).free_symbols)
                            if not s.startswith("__l")
                        }
                    for (_k, sym, _idx, begin_expr) in gather_affine_dims:
                        needed_syms |= {
                            s
                            for s in (str(x) for x in begin_expr.free_symbols) if not s.startswith("__l") and s != sym
                        }
                    for sname in needed_syms:
                        if sname not in nsdfg_node.symbol_mapping:
                            nsdfg_node.symbol_mapping[sname] = dace.symbolic.pystr_to_symbolic(sname)
                        if sname not in inner.symbols:
                            inner.add_symbol(
                                sname,
                                nsdfg_node.sdfg.parent_sdfg.symbols.get(sname, dace.int64)
                                if nsdfg_node.sdfg.parent_sdfg is not None else dace.int64)
                    for (k, expr) in affine_dims:
                        iname = f"_agidx_{src_name}_{k}"
                        suffix = 0
                        while iname in inner.arrays:
                            suffix += 1
                            iname = f"_agidx_{src_name}_{k}_{suffix}"
                        inner.add_array(iname,
                                        list(W),
                                        dace.int64,
                                        storage=dace.dtypes.StorageType.Register,
                                        transient=True)
                        fill = istate.add_tasklet(name=f"agidx_{iname}",
                                                  inputs=set(),
                                                  outputs={"_out"},
                                                  code=nested_loops(list(W), f"_out[{off}] = {expr};"),
                                                  language=dace.dtypes.Language.CPP)
                        iacc = istate.add_access(iname)
                        istate.add_edge(fill, "_out", iacc, None, dace.Memlet(f"{iname}[{out_subset}]"))
                        agidx_access_by_dim[k] = iacc
                        # Replace the placeholder with the fresh tile.
                        idx_info[k] = (k, None, iname, W[0], 1)
                    for (k, sym, idx_arr, begin_expr) in gather_affine_dims:
                        iname = f"_agidx_{src_name}_{k}"
                        suffix = 0
                        while iname in inner.arrays:
                            suffix += 1
                            iname = f"_agidx_{src_name}_{k}_{suffix}"
                        inner.add_array(iname,
                                        list(W),
                                        dace.int64,
                                        storage=dace.dtypes.StorageType.Register,
                                        transient=True)
                        # Substitute ``sym`` -> ``_idx[__l0]`` (a per-lane read
                        # of the widened index tile) inside the affine expr.
                        # K=1 here (gather_affine_dims is built only when
                        # ``len(W) == 1`` reaches the gather collapse), so
                        # ``__l0`` is the sole lane variable.
                        sym_expr = dace.symbolic.pystr_to_symbolic(sym)
                        placeholder = dace.symbolic.pystr_to_symbolic("__GATHER_LANE_READ__")
                        substituted = begin_expr.subs(sym_expr, placeholder)
                        expr_str = dace.symbolic.symstr(substituted).replace("__GATHER_LANE_READ__", "_idx[__l0]")
                        idx_window = inner.arrays[idx_arr].shape[0]
                        fill = istate.add_tasklet(name=f"agidx_{iname}",
                                                  inputs={"_idx"},
                                                  outputs={"_out"},
                                                  code=nested_loops(list(W), f"_out[{off}] = {expr_str};"),
                                                  language=dace.dtypes.Language.CPP)
                        iread = istate.add_access(idx_arr)
                        istate.add_edge(iread, None, fill, "_idx", dace.Memlet(f"{idx_arr}[0:{idx_window}]"))
                        iacc = istate.add_access(iname)
                        istate.add_edge(fill, "_out", iacc, None, dace.Memlet(f"{iname}[{out_subset}]"))
                        agidx_access_by_dim[k] = iacc
                        # Mark ``sym`` as consumed so the post-collapse
                        # ``_drop_gather_symbols`` simplifies its fan away.
                        consumed.add(sym)
                        idx_info[k] = (k, None, iname, W[0], 1)
                    for (k, sym, idx_arr, idx_expr_str, tile_var_dim_per_iter) in multidim_gather_dims:
                        iname = f"_agidx_{src_name}_{k}"
                        suffix = 0
                        while iname in inner.arrays:
                            suffix += 1
                            iname = f"_agidx_{src_name}_{k}_{suffix}"
                        inner.add_array(iname,
                                        list(W),
                                        dace.int64,
                                        storage=dace.dtypes.StorageType.Register,
                                        transient=True)
                        # Substitute each tile iter-var's bound inner-array
                        # dim with ``__l<p>`` so lane ``(l0, ..., l(K-1))``
                        # reads the right per-lane value:
                        #   K=1: ``edge_blk[0, 0, 0]`` -> ``edge_blk[0, __l0, 0]``
                        #   K=2: ``B[0, 0]``           -> ``B[__l0, __l1]``
                        # Use sympy Subscript surgery so we don't accidentally
                        # rewrite a numeric ``0`` in a non-index position.
                        idx_expr_sympy = dace.symbolic.pystr_to_symbolic(idx_expr_str)
                        if not isinstance(idx_expr_sympy, dace.symbolic.Subscript):
                            unresolved = True
                            break
                        new_indices = list(idx_expr_sympy.args[1:])
                        any_out_of_range = any(d >= len(new_indices) for d in tile_var_dim_per_iter.values())
                        if any_out_of_range:
                            unresolved = True
                            break
                        for p, d in tile_var_dim_per_iter.items():
                            new_indices[d] = dace.symbolic.pystr_to_symbolic(f"__l{p}")
                        # Connector names cannot collide with inner array
                        # names (validation rejects ``edge_blk`` as a
                        # connector when an ``edge_blk`` array exists). Pass
                        # the source via a connector ``_src`` (a 1D pointer
                        # in the CPP body), and compute the per-lane access
                        # as a FLAT offset using the source array's strides:
                        # ``edge_blk[0, __l0, 0]`` is invalid C++ syntax
                        # (comma operator collapses it to ``_src[0]``), so
                        # emit ``_src[0 * strides[0] + __l0 * strides[1] +
                        # 0 * strides[2]]``. The pointer ``_src`` points at
                        # the boundary memlet's begin element, so flat-offset
                        # indexing with the source strides addresses the
                        # right absolute position.
                        src_conn = "_src"
                        src_strides = inner.arrays[idx_arr].strides
                        flat_offset_terms = []
                        for d, idx_expr in enumerate(new_indices):
                            term = f"({dace.symbolic.symstr(idx_expr)}) * ({dace.symbolic.symstr(src_strides[d])})"
                            flat_offset_terms.append(term)
                        flat_offset = " + ".join(flat_offset_terms)
                        expr_str = f"{src_conn}[{flat_offset}]"
                        full_src_subset = ", ".join(f"0:{s}" for s in inner.arrays[idx_arr].shape)
                        fill = istate.add_tasklet(name=f"agidx_{iname}",
                                                  inputs={src_conn},
                                                  outputs={"_out"},
                                                  code=nested_loops(list(W), f"_out[{off}] = {expr_str};"),
                                                  language=dace.dtypes.Language.CPP)
                        iread = istate.add_access(idx_arr)
                        istate.add_edge(iread, None, fill, src_conn, dace.Memlet(f"{idx_arr}[{full_src_subset}]"))
                        iacc = istate.add_access(iname)
                        istate.add_edge(fill, "_out", iacc, None, dace.Memlet(f"{iname}[{out_subset}]"))
                        agidx_access_by_dim[k] = iacc
                        consumed.add(sym)
                        idx_info[k] = (k, None, iname, W[0], 1)
                if unresolved:
                    continue
                gather = TileGather(name=f"gather_{src_name}",
                                    widths=W,
                                    source_ndim=src_ndim,
                                    has_mask=mask_acc is not None,
                                    index_strides=tuple(stride for *_, stride in idx_info))
                istate.add_node(gather)
                istate.add_edge(e.src, None, gather, TileConnectors.SRC, dace.Memlet.from_array(src_name, src_arr))
                for k, sym, idx_arr, window, _stride in idx_info:
                    # Data-gather index tiles are widened 1D (``_idx[0:c*(W-1)+1]``);
                    # affine-fill index tiles are K-dim (one element per lane in the
                    # K-shape register tile) — pick the matching memlet subset from
                    # the descriptor's actual rank.
                    idx_subset = (out_subset if len(inner.arrays[idx_arr].shape) == len(W) else f"0:{window}")
                    # For affine fill dims (tile-var affine OR affine-of-
                    # gather-symbol), the fill tasklet already produced an
                    # AccessNode for the agidx; reuse it as the gather's input
                    # so the dataflow ``fill -> agidx -> gather`` is connected
                    # and the topological sort runs the fill first.
                    if k in agidx_access_by_dim:
                        src_acc = agidx_access_by_dim[k]
                        idx_arr_wire = idx_arr
                    else:
                        idx_arr_wire = self._materialize_loop_invariant_idx(inner, istate, idx_arr, nsdfg_node)
                        src_acc = istate.add_access(idx_arr_wire)
                    istate.add_edge(src_acc, None, gather, TileConnectors.idx(k),
                                    dace.Memlet(f"{idx_arr_wire}[{idx_subset}]"))
                    if sym is not None:
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
            scatter_syms = self._gather_index_symbols(e.data.subset, tile_var_set, inner)
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
            # Mixed-dim scatter at K=1 stays refused: the data-scatter symbol
            # may resolve to a length-1 LOOP-INVARIANT boundary connector
            # whose ``_idx_k[l]`` per-lane read is OOB at runtime (the
            # broadcast / unify-then-scatter lowering is a separate slice).
            # K>=2 paths route through the widened K-shape tile from the
            # K-aware fan-out and are unaffected.
            if len(scatter_syms) != dst_ndim and len(W) == 1:
                raise NotImplementedError(
                    f"PromoteNSDFGBodyToTiles: scatter on {dst_name!r} indexes {len(scatter_syms)} of "
                    f"{dst_ndim} dest dims; mixed affine + data-scatter dest dims at K=1 are not "
                    f"supported in this slice (K>=2 mixed scatter is supported)")
            # Mixed-dim classification: per dest dim, route data-scatter
            # (symbol in subset) to the widened index tile; route affine
            # (tile-var-only) to a fresh per-tile ``_agidx`` fill the same
            # way the gather collapse handles its affine_dims path.
            sym_by_dim = {d: sym for (d, sym) in scatter_syms}
            idx_info: list = []  # (k, sym, idx_arr_or_None, window)
            affine_dims: list = []  # (k, lane_expr)
            multidim_scatter_dims: list = []  # (k, sym, idx_arr, idx_expr_str, tile_var_dim_per_iter)
            unresolved = False
            outer_subsets_by_conn = {}
            parent_state = nsdfg_node.sdfg.parent
            for oe in parent_state.in_edges(nsdfg_node):
                if oe.dst_conn is not None and oe.data is not None and oe.data.subset is not None:
                    outer_subsets_by_conn[oe.dst_conn] = oe.data.subset
            for k in range(dst_ndim):
                if k in sym_by_dim:
                    sym = sym_by_dim[k]
                    idx_arr = self._index_array_for_symbol(inner, sym)
                    if idx_arr is None:
                        unresolved = True
                        break
                    idx_arr_shape = inner.arrays[idx_arr].shape
                    idx_arr_ndim = len(idx_arr_shape)
                    if idx_arr_ndim > 1:
                        # K-shape index tile (from the K-aware fan-out or a
                        # multi-dim index source — icon edge_blk-style):
                        # do Subscript surgery on the assignment RHS to
                        # substitute each iter-var-bound inner-array dim
                        # with its ``__l<p>`` per lane.
                        outer_sub = outer_subsets_by_conn.get(idx_arr)
                        tile_var_dim_per_iter: Dict[int, int] = {}
                        if outer_sub is not None and len(outer_sub.ranges) == idx_arr_ndim:
                            for p, tv_name in enumerate(spec.iter_vars):
                                tv_sym = dace.symbolic.pystr_to_symbolic(tv_name)
                                for d, (ob, _oe, _os) in enumerate(outer_sub.ranges):
                                    ob_syms = ob.free_symbols if hasattr(ob, 'free_symbols') else set()
                                    if tv_sym in ob_syms:
                                        tile_var_dim_per_iter[p] = d
                                        break
                        if not tile_var_dim_per_iter:
                            unresolved = True
                            break
                        idx_expr_str = self._index_expr_for_symbol(inner, sym)
                        if idx_expr_str is None:
                            unresolved = True
                            break
                        multidim_scatter_dims.append((k, sym, idx_arr, idx_expr_str, tile_var_dim_per_iter))
                        idx_info.append((k, None, None, W[0]))
                        continue
                    window = idx_arr_shape[0]
                    if _index_lane_stride(window, W[0]) != 1:
                        raise NotImplementedError(
                            f"PromoteNSDFGBodyToTiles: strided scatter index on {dst_name!r} (window {window}, "
                            f"W {W[0]}) not yet supported")
                    idx_info.append((k, sym, idx_arr, window))
                else:
                    b, _e, _s = e.data.subset.ranges[k]
                    expr = _lane_index_expr(str(b), spec.iter_vars)
                    if expr is None:
                        unresolved = True
                        break
                    affine_dims.append((k, expr))
                    idx_info.append((k, None, None, W[0]))
            if unresolved:
                continue
            # Build the agidx tiles for affine + multi-dim-scatter dest dims.
            off = tile_offset(list(W))
            agidx_access_by_dim: Dict[int, dace.nodes.AccessNode] = {}
            if affine_dims or multidim_scatter_dims:
                # Thread every symbol the affine + multidim expressions
                # reference through the NSDFG's symbol mapping + inner symbol
                # table (identity binding).
                needed_syms = set(spec.iter_vars)
                for (_k, expr) in affine_dims:
                    needed_syms |= {
                        s
                        for s in (str(x) for x in dace.symbolic.pystr_to_symbolic(expr).free_symbols)
                        if not s.startswith("__l")
                    }
                for sname in needed_syms:
                    if sname not in nsdfg_node.symbol_mapping:
                        nsdfg_node.symbol_mapping[sname] = dace.symbolic.pystr_to_symbolic(sname)
                    if sname not in inner.symbols:
                        inner.add_symbol(
                            sname,
                            nsdfg_node.sdfg.parent_sdfg.symbols.get(sname, dace.int64)
                            if nsdfg_node.sdfg.parent_sdfg is not None else dace.int64)
                for (k, expr) in affine_dims:
                    iname = f"_agidx_{dst_name}_{k}"
                    suffix = 0
                    while iname in inner.arrays:
                        suffix += 1
                        iname = f"_agidx_{dst_name}_{k}_{suffix}"
                    inner.add_array(iname,
                                    list(W),
                                    dace.int64,
                                    storage=dace.dtypes.StorageType.Register,
                                    transient=True)
                    fill = istate.add_tasklet(name=f"agidx_{iname}",
                                              inputs=set(),
                                              outputs={"_out"},
                                              code=nested_loops(list(W), f"_out[{off}] = {expr};"),
                                              language=dace.dtypes.Language.CPP)
                    iacc = istate.add_access(iname)
                    istate.add_edge(fill, "_out", iacc, None, dace.Memlet(f"{iname}[{out_subset}]"))
                    agidx_access_by_dim[k] = iacc
                    idx_info[k] = (k, None, iname, W[0])
                for (k, sym, idx_arr, idx_expr_str, tile_var_dim_per_iter) in multidim_scatter_dims:
                    iname = f"_agidx_{dst_name}_{k}"
                    suffix = 0
                    while iname in inner.arrays:
                        suffix += 1
                        iname = f"_agidx_{dst_name}_{k}_{suffix}"
                    inner.add_array(iname,
                                    list(W),
                                    dace.int64,
                                    storage=dace.dtypes.StorageType.Register,
                                    transient=True)
                    idx_expr_sympy = dace.symbolic.pystr_to_symbolic(idx_expr_str)
                    if not isinstance(idx_expr_sympy, dace.symbolic.Subscript):
                        unresolved = True
                        break
                    new_indices = list(idx_expr_sympy.args[1:])
                    if any(d >= len(new_indices) for d in tile_var_dim_per_iter.values()):
                        unresolved = True
                        break
                    for p, d in tile_var_dim_per_iter.items():
                        new_indices[d] = dace.symbolic.pystr_to_symbolic(f"__l{p}")
                    src_conn = "_src"
                    src_strides = inner.arrays[idx_arr].strides
                    flat_offset_terms = []
                    for d, idx_expr in enumerate(new_indices):
                        term = f"({dace.symbolic.symstr(idx_expr)}) * ({dace.symbolic.symstr(src_strides[d])})"
                        flat_offset_terms.append(term)
                    flat_offset = " + ".join(flat_offset_terms)
                    expr_str = f"{src_conn}[{flat_offset}]"
                    full_src_subset = ", ".join(f"0:{s}" for s in inner.arrays[idx_arr].shape)
                    fill = istate.add_tasklet(name=f"agidx_{iname}",
                                              inputs={src_conn},
                                              outputs={"_out"},
                                              code=nested_loops(list(W), f"_out[{off}] = {expr_str};"),
                                              language=dace.dtypes.Language.CPP)
                    iread = istate.add_access(idx_arr)
                    istate.add_edge(iread, None, fill, src_conn, dace.Memlet(f"{idx_arr}[{full_src_subset}]"))
                    iacc = istate.add_access(iname)
                    istate.add_edge(fill, "_out", iacc, None, dace.Memlet(f"{iname}[{out_subset}]"))
                    agidx_access_by_dim[k] = iacc
                    consumed.add(sym)
                    idx_info[k] = (k, None, iname, W[0])
                if unresolved:
                    continue
            scatter = TileScatter(name=f"scatter_{dst_name}",
                                  widths=W,
                                  dest_ndim=dst_ndim,
                                  has_mask=mask_acc is not None)
            istate.add_node(scatter)
            istate.add_edge(value_access, None, scatter, TileConnectors.SRC,
                            dace.Memlet(f"{value_access.data}[{out_subset}]"))
            for k, sym, idx_arr, window in idx_info:
                idx_shape = inner.arrays[idx_arr].shape
                if len(idx_shape) == 1:
                    idx_subset = f"0:{window}"
                else:
                    idx_subset = ", ".join(f"0:{s}" for s in idx_shape)
                if k in agidx_access_by_dim:
                    src_acc = agidx_access_by_dim[k]
                    idx_arr_wire = idx_arr
                else:
                    idx_arr_wire = self._materialize_loop_invariant_idx(inner, istate, idx_arr, nsdfg_node)
                    src_acc = istate.add_access(idx_arr_wire)
                istate.add_edge(src_acc, None, scatter, TileConnectors.idx(k),
                                dace.Memlet(f"{idx_arr_wire}[{idx_subset}]"))
                if sym is not None:
                    consumed.add(sym)
            self._wire_mask(istate, mask_acc, scatter, out_subset)
            istate.add_edge(scatter, TileConnectors.DST, e.dst, e.dst_conn, dace.Memlet.from_array(dst_name, dst_arr))
            istate.remove_edge(e)
            if to_remove is not None:
                # The assign tasklet's value source already wires into the
                # new ``TileScatter`` (``value_access`` above), so the dangling
                # in-edges are dead — drop them with the tasklet so its
                # connectors don't leak into the validated SDFG.
                for ie in list(istate.in_edges(to_remove)) + list(istate.out_edges(to_remove)):
                    istate.remove_edge(ie)
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
            cls = self._box_classification(ed.data.subset, inner.arrays[src_name], spec.iter_vars, src_name)
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

            # Refuse binops whose output transient binds an interstate-edge
            # symbol (``__sym_<tname>`` convention from ScalarToSymbol
            # promotion). ``_reshape_transients`` has already widened the
            # ``(1,)`` transient to a ``(W, ...)`` tile, so the downstream
            # assignment ``__sym = transient`` becomes a tile-from-scalar bind
            # that the validator rejects. Surface as a ScalarToSymbol-
            # promotion gap — the binop should have been lifted to an
            # interstate-edge assignment BEFORE the descent ran.
            _out_tname = out_access.data
            if f"__sym_{_out_tname}" in inner.symbols:
                raise NotImplementedError(f"PromoteNSDFGBodyToTiles: binop {t.label!r} writes to "
                                          f"{_out_tname!r} which is bound to interstate-edge symbol "
                                          f"__sym_{_out_tname}; lift via ScalarToSymbol or refuse the descent.")

            def _operand(token):
                if _is_numeric_literal(token):
                    return "Symbol", token, None
                ie = [e for e in istate.in_edges(t) if e.dst_conn == token]
                # A tile-iter-var used as a *value* (``c3 = (i > c0)``) has NO
                # in-edge — the symbol comes from the surrounding scope.
                # TileBinop's ``kind="Symbol"`` path drops the operand
                # connector entirely (``inputs.discard("_a")`` /
                # ``"_b"`` depending on which side is Symbol; see
                # ``tile_binop.py`` lines ~149-152) and embeds the per-lane
                # expression inline via ``(out_dtype)(expr)``. Lower the
                # tile-var with per-lane substitution ``v -> v + __l_p`` via
                # :func:`_lane_index_expr` so each lane sees the right value.
                # Extended to all K: a tile-iter-var read as a value (``c1 = i``,
                # ``i + offset``) is embedded inline via ``_lane_index_expr`` →
                # ``i + __l0``, ``j + __l1``, etc. The TileBinop's K-nested-loop
                # pure expansion writes one value per lane and ``_reshape_transients``
                # has already grown the destination transient to ``(W_0, ..., W_K)``
                # so the (subset, target) shapes match. Downstream scatter-index
                # consumers that hand-roll a 1D index tile are handled where they
                # are emitted, not preempted here.
                if len(ie) == 0 and token in spec.iter_vars:
                    expr = _lane_index_expr(token, spec.iter_vars) or token
                    return "Symbol", expr, None
                # A connector-less operand that is NOT a tile iter-var: an
                # outer-scope symbol used directly (outer map iter-var, a
                # parent-NSDFG scalar parameter, etc.). Same value across all
                # lanes, so wire it as a Symbol broadcast — the per-lane
                # tile loop closes over the symbol and reads it uniformly,
                # which is exactly what the lib-node's Symbol kind embeds
                # inline (``inputs.discard``-ed connector, expression baked
                # into the body string).
                if len(ie) == 0:
                    is_known_symbol = (token in inner.symbols or (nsdfg_node.sdfg.parent_sdfg is not None
                                                                  and token in nsdfg_node.sdfg.parent_sdfg.symbols)
                                       or token in nsdfg_node.symbol_mapping)
                    if is_known_symbol:
                        return "Symbol", token, None
                if len(ie) != 1 or not isinstance(ie[0].src, dace.nodes.AccessNode):
                    raise NotImplementedError(
                        f"PromoteNSDFGBodyToTiles: binop {t.label!r} operand {token!r} not a single tile read")
                src = ie[0].src
                src_desc = inner.arrays[src.data]
                isub = ie[0].data.subset
                # A length-1 transient intermediate (the frontend's
                # ``aa[j-1, i] -> aa_index (1,) -> tasklet[0]`` pattern) hides
                # an N-D-source per-lane tile read behind a scalar-shaped
                # operand. Walk one hop upstream: if the source is a length-1
                # transient with exactly ONE in-edge from a non-transient
                # in-connector whose subset carries a tile-var, treat the
                # tasklet read as a Tile reading from the ORIGINAL upstream
                # source at its tile-var-carrying subset.
                if src_desc.transient and tuple(src_desc.shape) == (1, ):
                    up_edges = list(istate.in_edges(src))
                    if len(up_edges) == 1 and isinstance(up_edges[0].src, dace.nodes.AccessNode):
                        up = up_edges[0].src
                        up_desc = inner.arrays[up.data]
                        up_sub = up_edges[0].data.subset
                        if (up.data in nsdfg_node.in_connectors and not up_desc.transient and up_sub is not None):
                            fs_begin = set()
                            for (b, _e, _s) in up_sub:
                                fs_begin |= {str(x) for x in getattr(b, "free_symbols", set())}
                            tile_dims_in_sub = sorted(d for d, (b, _e, _s) in enumerate(up_sub)
                                                      if {str(x)
                                                          for x in getattr(b, "free_symbols", set())}
                                                      & set(spec.iter_vars))
                            # Exactly K tile-var dims in the source subset match
                            # the K-dim tile cleanly: the wiring emits a TileLoad
                            # with ``src_dims=(tile_dims_in_sub,)`` reading the
                            # K-dim widened subset (each tile-var dim grows from
                            # 1 to its W_k) into a (W,)-shape register tile.
                            if len(tile_dims_in_sub) == len(W):
                                return "NDTile", (up, up_sub, tuple(tile_dims_in_sub)), None
                # A broadcast (Scalar) operand: a true ``dace.data.Scalar``, a
                # length-1 ``(1,)`` array, OR a single-element TILE-VAR-FREE read
                # of a larger in-connector array (``a[j]`` where ``j`` is
                # non-tiled — every lane reads the same element, a loop-invariant
                # scalar). A single-element read WITH a tile-var in the begin
                # (``a[i]`` for tiled ``i``) is per-lane SHIFTED (lane ``l`` reads
                # ``a[i+l]``) — that is a Tile read, not a broadcast.
                is_single_invariant = False
                if isub is not None and bool(dace.symbolic.simplify(isub.num_elements() - 1) == 0):
                    fs_begin = set()
                    for (b, _e, _s) in isub:
                        fs_begin |= {str(x) for x in getattr(b, "free_symbols", set())}
                    is_single_invariant = not (fs_begin & set(spec.iter_vars))
                is_scalar = (src.data in nsdfg_node.in_connectors and not src_desc.transient
                             and (isinstance(src_desc, dace.data.Scalar) or tuple(src_desc.shape) == (1, )
                                  or is_single_invariant))
                if is_scalar:
                    return "Scalar", src, isub
                # A Tile operand is read as a (W, ...) tile — but the connector
                # ARRAY may be larger: a stencil bounding window (``A_win`` of
                # shape ``E+W``) read at a shifted W-subset ``A_win[k:k+W]``. So
                # validate the READ SUBSET size, not the array shape, and pass the
                # subset through to the wiring. A non-W subset (an unhandled
                # multi-dim / gather residue, ``zqx[jo-1, 0, i]``) is refused.
                sizes = isub.size()
                ok = (len(sizes) == len(W)
                      and all(bool(dace.symbolic.simplify(sz - w) == 0) for sz, w in zip(sizes, W)))
                if not ok:
                    raise NotImplementedError(
                        f"PromoteNSDFGBodyToTiles: binop {t.label!r} operand {token!r} reads {src.data!r} "
                        f"subset size {tuple(sizes)} != tile {W} — an unhandled data-dependent / mixed gather")
                return "Tile", src, isub

            kind_a, info_a, sub_a = _operand(a_tok)
            kind_b, info_b, sub_b = _operand(b_tok)

            # ``NDTile`` (from the ``Array[tile_var_subset] -> length-1
            # transient -> tasklet[0]`` walk-back in :func:`_operand`) counts as
            # a Tile operand for the "must have one tile" check — the wiring
            # below emits a TileLoad and then routes the resulting tile
            # transient into the binop connector, identical to the plain Tile
            # case from the lib node's perspective. A ``Symbol`` operand whose
            # per-lane expression carries ``__l`` (the tile-var-as-value
            # path, ``c3 = (i > c0)``) also provides per-lane variation; the
            # TileBinop drops the operand connector for Symbol and embeds the
            # expression inline, so the lib node still validates.
            def _is_tile_kind(k, info):
                return k in ("Tile", "NDTile") or (k == "Symbol" and isinstance(info, str) and "__l" in info)

            if not _is_tile_kind(kind_a, info_a) and not _is_tile_kind(kind_b, info_b):
                # Two operands are non-tile (Scalar / Symbol / loop-invariant
                # combinations). The output decides whether we can emit a
                # broadcast TileBinop:
                # (a) OUTPUT is tile-shaped AND not used as an interstate-
                #     edge assignment RHS — the lib node's pure expansion
                #     fills every lane with the (Scalar op Scalar) value, so
                #     a real ``(W, ...)`` tile is produced. TileBinop already
                #     supports kind_a / kind_b ``Scalar`` and ``Symbol`` with
                #     this broadcast semantics. We exclude interstate-bound
                #     transients because an assignment ``__sym = transient``
                #     reads a scalar value and is incompatible with the
                #     broadcast tile (every lane reads element ``[0]``,
                #     which for a broadcast tile is correct semantically
                #     but the downstream pipeline that emptied the
                #     assignment relies on the original ``(1,)`` shape).
                # (b) OUTPUT is also a length-1 / Scalar transient — the
                #     binop's result is a single scalar value (typically an
                #     index computation like ``i + offset1`` bound to a
                #     symbol via a downstream interstate-edge assignment).
                #     Refuse loudly: the descent's downstream walks expect
                #     the scalar to either be (i) already lifted to an
                #     interstate-edge symbol via ScalarToSymbol promotion or
                #     (ii) a tile transient. Leaving the binop as a Python
                #     tasklet breaks the symbol-propagation contract.
                out_desc = inner.arrays.get(out_access.data)
                out_is_tile_shape = (out_desc is not None and not isinstance(out_desc, dace.data.Scalar)
                                     and tuple(out_desc.shape) != (1, ) and tuple(out_desc.shape) == tuple(W))
                # The transient may have been bound to an interstate-edge
                # symbol via ScalarToSymbol promotion BEFORE the descent
                # ran — once ``_reshape_transients`` widens it from ``(1,)``
                # to ``(W, ...)``, the assignment ``__sym = transient`` no
                # longer reads a single scalar. Detect via two channels:
                #  - the transient is still the RHS of an interstate-edge
                #    assignment in the inner SDFG, or
                #  - a symbol whose name is ``__sym_<transient>`` exists in
                #    ``inner.symbols`` (the convention ScalarToSymbol
                #    promotion uses — the assignment may have been
                #    transiently emptied earlier in the descent but the
                #    symbol still binds the transient's logical value).
                tname = out_access.data
                out_is_iedge_src = (any(tname == v for ie in inner.all_interstate_edges()
                                        for v in ie.data.assignments.values()) or f"__sym_{tname}" in inner.symbols)
                if not out_is_tile_shape or out_is_iedge_src:
                    raise NotImplementedError(f"PromoteNSDFGBodyToTiles: binop {t.label!r} has no tile operand "
                                              f"({kind_a}/{kind_b}) and its output {tname!r} is not a "
                                              f"broadcast-safe tile — surface this as a ScalarToSymbol-promotion "
                                              f"gap (the binop should have been lifted to an interstate-edge "
                                              f"assignment before the descent).")
                # Fall through: kind_a / kind_b stay as Scalar / Symbol; the
                # TileBinop construction below uses them directly and the
                # pure expansion broadcasts each operand across all lanes.
            # The TileBinop ``kind_X`` property only accepts ``"Tile" | "Scalar"
            # | "Symbol"`` (lib-node validation); NDTile is wired as Tile via
            # an inserted TileLoad in :func:`_wire`. Stamp the property as
            # ``"Tile"`` here so the lib node validates.
            binop_kind_a = "Tile" if kind_a == "NDTile" else kind_a
            binop_kind_b = "Tile" if kind_b == "NDTile" else kind_b
            kwargs = dict(name=f"{t.label}_binop",
                          widths=W,
                          op=op,
                          has_mask=mask_acc is not None,
                          kind_a=binop_kind_a,
                          kind_b=binop_kind_b)
            if binop_kind_a == "Symbol":
                kwargs["expr_a"] = info_a
            if binop_kind_b == "Symbol":
                kwargs["expr_b"] = info_b
            binop = TileBinop(**kwargs)
            istate.add_node(binop)

            def _wire(kind, info, isub, conn):
                if kind == "Tile":
                    # Reuse the in-edge subset (a full ``[0:W]`` tile or a shifted
                    # ``[k:k+W]`` window subset) so a stencil neighbour reads the
                    # right shifted slice of the bounding-window tile. Build a
                    # FRESH Range — when both operands are the same tile (``b*b``
                    # from ``b**2``) the two edges must not share one subset object
                    # (DaCe rejects a duplicate memlet reference).
                    istate.add_edge(info, None, binop, conn,
                                    dace.Memlet(data=info.data, subset=subsets.Range(list(isub.ranges))))
                elif kind == "NDTile":
                    # ``info = (up_access, up_sub, tile_dims_tuple)`` from the
                    # ``Array[tile_var_subset] -> length-1 transient ->
                    # tasklet[0]`` walk-back. Emit a TileLoad reading the
                    # upstream N-D source at the per-lane subset (each
                    # tile-var dim widened from 1 to its W_k), producing a
                    # ``widths``-shape tile transient, and wire that tile
                    # into the binop connector.
                    up_acc, up_sub, tile_dims = info
                    up_arr = inner.arrays[up_acc.data]
                    # Map each tile-var dim in the source to its tile width.
                    # The K-dim ordering follows ``spec.iter_vars`` (innermost-
                    # last); ``tile_dims`` is in source-dim order, so we walk
                    # both together by matching the iter-var that appears in
                    # each source dim's begin expression.
                    tile_dim_to_w = {}
                    for k_idx, src_dim in enumerate(tile_dims):
                        b, _e, _s = up_sub.ranges[src_dim]
                        fs = {str(x) for x in getattr(b, "free_symbols", set())}
                        iv_match = fs & set(spec.iter_vars)
                        if len(iv_match) != 1:
                            return None  # unhandled — multiple tile vars in one dim
                        iv = next(iter(iv_match))
                        iv_idx = spec.iter_vars.index(iv)
                        tile_dim_to_w[src_dim] = W[iv_idx]
                    promoted_ranges = []
                    for d, (b, e_, s) in enumerate(up_sub):
                        if d in tile_dim_to_w:
                            promoted_ranges.append((b, b + (tile_dim_to_w[d] - 1), 1))
                        else:
                            promoted_ranges.append((b, e_, s))
                    promoted_sub = subsets.Range(promoted_ranges)
                    tile_name = f"{up_acc.data}_ndtile"
                    suffix = 0
                    while tile_name in inner.arrays:
                        suffix += 1
                        tile_name = f"{up_acc.data}_ndtile_{suffix}"
                    inner.add_array(tile_name,
                                    list(W),
                                    up_arr.dtype,
                                    storage=dace.dtypes.StorageType.Register,
                                    transient=True)
                    tile_acc = istate.add_access(tile_name)
                    load = TileLoad(name=f"load_{up_acc.data}_ndtile",
                                    widths=W,
                                    dim_strides=tuple(1 for _ in W),
                                    src_dims=tile_dims,
                                    has_mask=mask_acc is not None)
                    istate.add_node(load)
                    istate.add_edge(up_acc, None, load, TileConnectors.SRC,
                                    dace.Memlet(data=up_acc.data, subset=promoted_sub))
                    self._wire_mask(istate, mask_acc, load, subset)
                    istate.add_edge(load, TileConnectors.DST, tile_acc, None, dace.Memlet(f"{tile_name}[{subset}]"))
                    istate.add_edge(tile_acc, None, binop, conn, dace.Memlet(f"{tile_name}[{subset}]"))
                elif kind == "Scalar":
                    # Use the in-edge subset (``[0]`` for a length-1 array, the
                    # actual point ``[j]`` for a single-element read of a bigger
                    # array) so DaCe reads the right element. Fresh Range.
                    sub = subsets.Range(list(isub.ranges)) if isub is not None else None
                    sub_memlet = dace.Memlet(data=info.data,
                                             subset=sub) if sub is not None else dace.Memlet(f"{info.data}[0]")
                    istate.add_edge(info, None, binop, conn, sub_memlet)
                # Symbol: nothing to wire (embedded inline).

            _wire(kind_a, info_a, sub_a, TileConnectors.A)
            _wire(kind_b, info_b, sub_b, TileConnectors.B)
            self._wire_mask(istate, mask_acc, binop, subset)
            write_access, write_conn = self._route_output(istate, out_access, out_edge.dst_conn, nsdfg_node, spec)
            istate.add_edge(binop, TileConnectors.C, write_access, write_conn,
                            dace.Memlet(f"{write_access.data}[{subset}]"))
            # Preserve dependency-only in-edges (the frontend's
            # ``Memlet(None)`` pattern that orders a tasklet after a
            # producer without consuming data — e.g. an init tasklet
            # ``_out = 0`` reading nothing but sequenced after a prior
            # access). The original tasklet is removed below; re-route
            # those dep edges onto the new binop so the dataflow ordering
            # invariant survives. ``dst_conn=None`` on the binop side is
            # the canonical no-data dep-edge form.
            for e in list(istate.in_edges(t)):
                if (e.data is None or e.data.data is None) and e.dst_conn is None:
                    istate.add_nedge(e.src, binop, dace.Memlet())
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
            # Mirror :meth:`_promote_binops`: refuse outputs bound to an
            # interstate-edge symbol (``__sym_<tname>``).
            _out_tname = out_access.data
            if f"__sym_{_out_tname}" in inner.symbols:
                raise NotImplementedError(f"PromoteNSDFGBodyToTiles: unop {t.label!r} writes to "
                                          f"{_out_tname!r} which is bound to interstate-edge symbol "
                                          f"__sym_{_out_tname}; lift via ScalarToSymbol or refuse the descent.")
            isub_a = None
            if _is_numeric_literal(a_tok):
                kind_a, info_a = "Symbol", a_tok
            else:
                ie = [e for e in istate.in_edges(t) if e.dst_conn == a_tok]
                # A connector-less operand: per-lane tile-iter-var (``i + __l0``)
                # OR an outer-scope known symbol (broadcast — same value across
                # all lanes). Mirror :meth:`_promote_binops._operand`.
                if len(ie) == 0 and a_tok in spec.iter_vars:
                    expr = _lane_index_expr(a_tok, spec.iter_vars) or a_tok
                    kind_a, info_a = "Symbol", expr
                elif len(ie) == 0:
                    is_known_symbol = (a_tok in inner.symbols or (nsdfg_node.sdfg.parent_sdfg is not None
                                                                  and a_tok in nsdfg_node.sdfg.parent_sdfg.symbols)
                                       or a_tok in nsdfg_node.symbol_mapping)
                    if not is_known_symbol:
                        raise NotImplementedError(f"PromoteNSDFGBodyToTiles: unop {t.label!r} operand {a_tok!r} "
                                                  f"not a single tile read")
                    kind_a, info_a = "Symbol", a_tok
                else:
                    if len(ie) != 1 or not isinstance(ie[0].src, dace.nodes.AccessNode):
                        raise NotImplementedError(
                            f"PromoteNSDFGBodyToTiles: unop {t.label!r} operand {a_tok!r} not a single tile read")
                    src = ie[0].src
                    src_desc = inner.arrays[src.data]
                    isub_a = ie[0].data.subset
                    # See _promote_binops._operand: single-element read counts as a
                    # broadcast only when the begin is tile-var-FREE (else lane ``l``
                    # reads a different element per lane -- a Tile read, not a broadcast).
                    is_single_invariant = False
                    if isub_a is not None and bool(dace.symbolic.simplify(isub_a.num_elements() - 1) == 0):
                        fs_begin = set()
                        for (b, _e, _s) in isub_a:
                            fs_begin |= {str(x) for x in getattr(b, "free_symbols", set())}
                        is_single_invariant = not (fs_begin & set(spec.iter_vars))
                    is_scalar = (src.data in nsdfg_node.in_connectors and not src_desc.transient
                                 and (isinstance(src_desc, dace.data.Scalar) or tuple(src_desc.shape) == (1, )
                                      or is_single_invariant))
                    kind_a, info_a = ("Scalar", src) if is_scalar else ("Tile", src)
            # When the only operand is a Symbol/Scalar broadcast, allow the
            # TileUnop only when the OUTPUT is tile-shaped AND not used as
            # an interstate-edge assignment RHS (the broadcast tile's
            # ``[0]`` read returns the correct value, but downstream walks
            # depend on the original ``(1,)`` shape — see the binop
            # rationale above). Otherwise refuse so the gap surfaces
            # cleanly as a ScalarToSymbol-promotion issue.
            if kind_a in ("Scalar", "Symbol"):
                out_desc = inner.arrays.get(out_access.data)
                out_is_tile_shape = (out_desc is not None and not isinstance(out_desc, dace.data.Scalar)
                                     and tuple(out_desc.shape) != (1, ) and tuple(out_desc.shape) == tuple(W))
                tname = out_access.data
                out_is_iedge_src = (any(tname == v for ie in inner.all_interstate_edges()
                                        for v in ie.data.assignments.values()) or f"__sym_{tname}" in inner.symbols)
                if not out_is_tile_shape or out_is_iedge_src:
                    raise NotImplementedError(f"PromoteNSDFGBodyToTiles: unop {t.label!r} has only a {kind_a} operand "
                                              f"and its output {tname!r} is not a broadcast-safe tile — "
                                              f"surface this as a ScalarToSymbol-promotion gap (the unop should have "
                                              f"been lifted to an interstate-edge assignment before the descent).")
            kwargs = dict(name=f"{t.label}_unop", widths=W, op=op, has_mask=mask_acc is not None, kind_a=kind_a)
            if kind_a == "Symbol":
                kwargs["expr_a"] = info_a
            unop = TileUnop(**kwargs)
            istate.add_node(unop)
            if kind_a == "Tile":
                # Use the in-edge subset (full [0:W] or a shifted [k:k+W] window
                # sub-block) — same subset-preservation as _promote_binops so a
                # stencil neighbour through a unop reads the right shifted slice.
                tile_sub = subsets.Range(list(isub_a.ranges)) if isub_a is not None else None
                tile_memlet = dace.Memlet(
                    data=info_a.data,
                    subset=tile_sub) if tile_sub is not None else dace.Memlet(f"{info_a.data}[{subset}]")
                istate.add_edge(info_a, None, unop, TileConnectors.A, tile_memlet)
            elif kind_a == "Scalar":
                # In-edge subset ([0] for length-1 array, [j] for single-element
                # read of bigger array). Fresh Range to avoid duplicate-reference.
                scl_sub = subsets.Range(list(isub_a.ranges)) if isub_a is not None else None
                scl_memlet = dace.Memlet(data=info_a.data,
                                         subset=scl_sub) if scl_sub is not None else dace.Memlet(f"{info_a.data}[0]")
                istate.add_edge(info_a, None, unop, TileConnectors.A, scl_memlet)
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
        K = len(W)
        full_subset = ", ".join(f"0:{w}" for w in W)
        materialized: Dict[object, dace.nodes.AccessNode] = {}
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
                # K>=2 fan-out widened this connector as a strided VIEW of the
                # source array (descriptor strides match the source). The
                # ``_collapse_tile_gathers`` fill emits a flat-offset read using
                # those source strides directly; routing the read through a
                # contiguous materialized tile would change the strides under
                # the already-emitted fill body and produce wrong gather
                # indices. The strided-view read is already correct for the
                # gather fill, so skip the materialization for these.
                if cname in getattr(self, "_kd_idx_connectors", set()):
                    continue
                isub = e.data.subset
                # Only a W-sized read is a tile operand. A whole-tile connector
                # (shape == W, read at ``[0:W]``) is materialized once per
                # connector; a stencil bounding-window connector (shape ``E+W``
                # per dim, read at a SHIFTED sub-block ``[k:k+W]``) is a strided
                # sub-block (its row stride is the window width, not W) that the
                # contiguous-offset binop cannot address directly, so materialize
                # each distinct sub-block into a contiguous tile (keyed by the
                # offset). 1-D windows are contiguous so this is a no-op copy; K>=2
                # windows genuinely need the stride-aware load.
                sizes = isub.size()
                if not (len(sizes) == K and all(bool(dace.symbolic.simplify(sz - w) == 0) for sz, w in zip(sizes, W))):
                    continue
                is_whole = (tuple(desc.shape) == W
                            and all(bool(dace.symbolic.simplify(b) == 0) for (b, _e, _s) in isub))
                if is_whole:
                    cls = self._box_classification(isub, desc, spec.iter_vars, cname)
                    dim_strides = tuple(cls.dim_strides)
                    src_dims = tuple(cls.match_dims)
                    src_memlet = dace.Memlet(f"{cname}[{full_subset}]")
                    key: object = cname
                else:
                    # Window sub-block: the connector's own (window) strides + the
                    # recorded tile-lane permutation address it; coeff is 1.
                    dim_strides = (1, ) * K
                    src_dims = self._conn_match_dims.get(cname, tuple(range(K)))
                    src_memlet = dace.Memlet(data=cname, subset=subsets.Range(list(isub.ranges)))
                    key = (cname, tuple(str(b) for (b, _e, _s) in isub))
                if key not in materialized:
                    tile_name = f"{cname}_ld"
                    suffix = 0
                    while tile_name in inner.arrays:
                        suffix += 1
                        tile_name = f"{cname}_ld_{suffix}"
                    inner.add_array(tile_name,
                                    list(W),
                                    desc.dtype,
                                    storage=dace.dtypes.StorageType.Register,
                                    transient=True)
                    load = TileLoad(name=f"load_{cname}",
                                    widths=W,
                                    dim_strides=dim_strides,
                                    src_dims=src_dims,
                                    has_mask=mask_acc is not None)
                    istate.add_node(load)
                    istate.add_edge(istate.add_access(cname), None, load, TileConnectors.SRC, src_memlet)
                    self._wire_mask(istate, mask_acc, load, full_subset)
                    tile_acc = istate.add_access(tile_name)
                    istate.add_edge(load, TileConnectors.DST, tile_acc, None,
                                    dace.Memlet(f"{tile_name}[{full_subset}]"))
                    materialized[key] = tile_acc
                tile_acc = materialized[key]
                istate.remove_edge(e)
                istate.add_edge(tile_acc, None, tasklet, e.dst_conn, dace.Memlet(f"{tile_acc.data}[{full_subset}]"))

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
                fill = istate.add_tasklet(name=f"const_{out_access.data}",
                                          inputs=inputs,
                                          outputs={"_out"},
                                          code=nested_loops(list(W), body),
                                          language=dace.dtypes.Language.CPP)
                self._wire_mask(istate, mask_acc, fill, subset)
                istate.add_edge(fill, "_out", out_access, out_edge.dst_conn,
                                dace.Memlet(f"{out_access.data}[{subset}]"))
            else:
                # Output-connector const store: fill a fresh contiguous transient
                # tile (every lane) and let _promote_stores mask the store.
                write_access, write_conn = self._route_output(istate, out_access, out_edge.dst_conn, nsdfg_node, spec)
                fill = istate.add_tasklet(name=f"const_{out_access.data}",
                                          inputs=set(),
                                          outputs={"_out"},
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
                return ie[0].src, ie[0].data.subset

            cond_src, cond_sub = _src_of(m.group("c"))
            then_src, then_sub = _src_of(m.group("t"))
            else_src, else_sub = _src_of(m.group("e"))
            merge = TileMerge(name=f"{t.label}_merge", widths=W, has_mask=mask_acc is not None)
            istate.add_node(merge)

            def _op_memlet(src, sub):
                """In-edge subset (full ``[0:W]`` or a shifted ``[k:k+W]`` window
                sub-block, fresh Range so two operands of the same tile don't
                share a memlet object)."""
                if sub is None:
                    return dace.Memlet(f"{src.data}[{subset}]")
                return dace.Memlet(data=src.data, subset=subsets.Range(list(sub.ranges)))

            istate.add_edge(cond_src, None, merge, "_cond", _op_memlet(cond_src, cond_sub))
            istate.add_edge(then_src, None, merge, "_t", _op_memlet(then_src, then_sub))
            istate.add_edge(else_src, None, merge, "_e", _op_memlet(else_src, else_sub))
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
            cls = self._box_classification(ed.data.subset, inner.arrays[dst_name], spec.iter_vars, dst_name)
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
        # Reset per-promote bookkeeping for the K-aware fan-out's exempt list.
        self._kd_idx_connectors: Set[str] = set()
        if len(W) == 1:
            fan_out_tile_gather_index_symbols(inner, nsdfg_node, parent_state, W[0], spec.iter_vars[0])
        else:
            # K>=2: widen each length-1 boundary connector that an interstate-
            # edge assignment binds to a gather index. The widened K-shape tile
            # carries the per-lane gather index across all K dims and routes
            # through :meth:`_collapse_tile_gathers`'s ``multidim_gather_dims``
            # path (Subscript surgery substitutes ``__l<p>`` per iter-var-bound
            # dim) — exactly what the K=1 fan does at K=1, generalised to K.
            self._kd_idx_connectors = fan_out_tile_gather_index_symbols_kd(inner, nsdfg_node, parent_state, list(W),
                                                                           list(spec.iter_vars))
        # Un-refine ``RefineNestedAccess`` Min/Max union-bounded body connectors
        # (s115 ``a[Min(i,j):Max(i,j)+1]`` from a triangular forward-substitution
        # with two distinct reads ``a[i]`` + ``a[j]``). Restore the full source
        # view and add the original outer begin back to each inner subset so the
        # downstream widen / classify / broadcast paths see clean absolute
        # indices instead of Min/Max in the connector subset. Also expand any
        # tile-var-bearing single-element inner subset (``a[i]`` for the i-tile)
        # to the per-tile W-window ``a[i : i+W-1]`` so
        # :meth:`_materialize_connector_reads` materialises it into a contiguous
        # ``(W,)`` tile; a tile-var-free single-element subset (``a[j]``) stays a
        # point and lowers as a Scalar broadcast through
        # :meth:`_promote_binops._operand`.
        self._unrefine_minmax_connectors(parent_state, nsdfg_node, spec)
        # Affine / structured non-box reads (diagonal A[i,i], correlated A[2*i,i],
        # structured c[i//2]) -> TileGather over the full source with affine
        # index tiles. Runs before the box widening, which then skips these
        # connectors (their outer edge is the tile-var-free full-array subset).
        self._collapse_affine_gathers(parent_state, nsdfg_node, spec, mask_name)
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
