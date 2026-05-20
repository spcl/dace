# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``EmitTileOps`` — rewrite every K-dim eligible inner-map body to use
:mod:`dace.libraries.tileops` lib nodes.

T5 MVP shape (matches the v2 plan): each inner map's body contains a
single binop tasklet ``__output = __rhs1 OP __rhs2`` (post-
``SplitTasklets``) reading scalar accesses ``A[i, j]`` and writing
``C[i, j]``. The pass:

1. Promotes each per-iteration scalar memlet ``X[i, j]`` to its tile
   region ``X[i:i+W_0, j:j+W_1]``.
2. Inserts :class:`TileLoad` lib nodes producing tile-shape transients.
3. Replaces the binop tasklet with :class:`TileBinop`.
4. Emits :class:`TileStore` for the output edge.

All lib nodes are placed in the parent state, inside the inner-map
scope, with mask wiring threaded through.
"""
import re
from typing import Dict, List, Optional, Tuple

import dace
from dace import properties, subsets, symbolic
from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl
from dace.libraries.tileops import TileBinop, TileLoad, TileStore
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.name_schemes import TileNameScheme
from dace.transformation.passes.vectorization.utils.tile_dims import (
    TileAccessKind,
    TileDimSpec,
    classify_tile_access,
)

_BINOP_RE = re.compile(
    r"^\s*(?P<out>\w+)\s*=\s*"
    r"(?:\(\s*)?(?P<a>\w+)\s*(?P<op>\+|-|\*|/|<=|>=|==|!=|<|>|and|or)\s*(?P<b>\w+)\s*\)?\s*;?\s*$"
)
_PY_TO_TILEBINOP_OP = {
    "+": "+", "-": "-", "*": "*", "/": "/",
    "<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "==", "!=": "!=",
    "and": "&&", "or": "||",
}


def _tile_region_subset(orig_subset: subsets.Range,
                        iter_vars: Tuple[str, ...],
                        widths: Tuple[int, ...]) -> subsets.Range:
    """Promote a per-iteration scalar subset to its tile-region slice.

    :param orig_subset: The per-iteration subset (one ``(b, e, s)`` per
        array dim).
    :param iter_vars: Tile iter-var names, innermost-last.
    :param widths: Tile widths matching ``iter_vars``.
    :returns: A new :class:`subsets.Range` covering the tile region.
    """
    tile_var_set = set(iter_vars)
    tile_var_to_width = dict(zip(iter_vars, widths))
    new_ranges = []
    for (b, e, s) in orig_subset.ranges:
        b_sym = symbolic.pystr_to_symbolic(str(b))
        free_in_b = {str(sym) for sym in b_sym.free_symbols}
        deps = free_in_b & tile_var_set
        if len(deps) == 1:
            tvar = next(iter(deps))
            w = tile_var_to_width[tvar]
            new_ranges.append((b, b + (w - 1), 1))
        else:
            new_ranges.append((b, e, s))
    return subsets.Range(new_ranges)


@properties.make_properties
class EmitTileOps(ppl.Pass):
    """Replace per-iteration scalar tasklets with tile-op lib node chains.

    Preconditions enforced via loud failure:

    * Inner map's body is a single binop tasklet
      ``out = lhs OP rhs`` with ``OP`` in
      ``{+, -, *, /, <, <=, >, >=, ==, !=, and, or}``.
    * The outer scope has a ``_tile_iter_mask`` produced by
      :class:`GenerateTileIterationMask`.

    All other shapes (multi-statement bodies, gather, multi-tasklet
    fusion chains) raise ``NotImplementedError``.
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
            raise ValueError(f"EmitTileOps: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = list(widths)

    def modifies(self) -> ppl.Modifies:
        """Pass rewrites map bodies.

        :returns: ``ppl.Modifies.Everything``.
        """
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Idempotent — runs once.

        :param modified: Modifications produced by earlier passes (unused).
        :returns: ``False``.
        """
        return False

    def _spec_for(self, map_entry: MapEntry) -> TileDimSpec:
        """Rebuild a :class:`TileDimSpec` from a map's last K params.

        :param map_entry: Inner map entry.
        :returns: A fresh :class:`TileDimSpec` covering the K innermost
            dims.
        """
        K = len(self.widths)
        params = list(map_entry.map.params)
        ranges = list(map_entry.map.range.ranges)
        return TileDimSpec(
            iter_vars=tuple(params[-K:]),
            widths=tuple(self.widths),
            global_ubs=tuple(str(r[1] + 1) for r in ranges[-K:]),
        )

    def _find_body_tasklets(self,
                            state: dace.SDFGState,
                            map_entry: MapEntry) -> List[dace.nodes.Tasklet]:
        """Return the tasklets sitting in ``map_entry``'s scope.

        :param state: Parent state.
        :param map_entry: Inner map entry.
        :returns: List of tasklet nodes strictly inside the map scope.
        """
        nodes_in_scope = state.all_nodes_between(map_entry, state.exit_node(map_entry))
        return [n for n in nodes_in_scope if isinstance(n, dace.nodes.Tasklet)]

    def _classify_binop_tasklet(self, tasklet: dace.nodes.Tasklet) -> Tuple[str, str, str, str]:
        """Parse a single-line Python binop tasklet body into its parts.

        :param tasklet: Compute tasklet to inspect.
        :returns: ``(out_conn, lhs_conn, op, rhs_conn)``.
        :raises NotImplementedError: When the body does not match.
        """
        body = tasklet.code.as_string.strip().rstrip(";").strip()
        m = _BINOP_RE.match(body)
        if m is None:
            raise NotImplementedError(
                f"EmitTileOps: tasklet {tasklet.label!r} body {body!r} is not 'out = lhs OP rhs'; "
                f"run SplitTasklets first."
            )
        out = m.group("out")
        if out not in tasklet.out_connectors:
            raise NotImplementedError(
                f"EmitTileOps: tasklet {tasklet.label!r} writes to {out!r} not in out_connectors"
            )
        return out, m.group("a"), _PY_TO_TILEBINOP_OP[m.group("op")], m.group("b")

    def _operand_kind(self,
                      state: dace.SDFGState,
                      tasklet: dace.nodes.Tasklet,
                      conn_name: str,
                      spec: TileDimSpec) -> Tuple[str, Optional[Tuple]]:
        """Decide whether ``conn_name`` reads a tile region or a symbol.

        :param state: Parent state containing the tasklet.
        :param tasklet: The compute tasklet.
        :param conn_name: Input connector name.
        :param spec: Tile spec for the surrounding map.
        :returns: ``("Tile", (path_edge, dim_strides))`` for Tile,
            ``("Symbol", (expr_str,))`` for Symbol.
        :raises NotImplementedError: For shapes T5 MVP doesn't handle.
        """
        in_e = [e for e in state.in_edges(tasklet) if e.dst_conn == conn_name]
        if len(in_e) != 1:
            raise NotImplementedError(
                f"EmitTileOps: tasklet {tasklet.label!r} input {conn_name!r} has "
                f"{len(in_e)} in-edges; expected exactly 1"
            )
        edge = in_e[0]
        if edge.data is None or edge.data.data is None:
            raise NotImplementedError(
                f"EmitTileOps: tasklet {tasklet.label!r} input {conn_name!r} has no memlet"
            )
        src_data_name = edge.data.data
        src_arr = state.sdfg.arrays[src_data_name]
        cls = classify_tile_access(
            edge.data.subset,
            tuple(src_arr.strides),
            spec.iter_vars,
        )
        path = state.memlet_path(edge)
        source_edge = path[0]
        if cls.kind in (TileAccessKind.CONTIGUOUS, TileAccessKind.STRIDED):
            return "Tile", (source_edge, edge, edge.data.subset, cls.dim_strides)
        if cls.kind == TileAccessKind.BROADCAST_SYMBOL:
            return "Symbol", (src_data_name,)
        raise NotImplementedError(
            f"EmitTileOps: tasklet {tasklet.label!r} input {conn_name!r} access "
            f"{source_edge.data!r} is {cls.kind.value}; T5 MVP only handles Tile / Symbol"
        )

    def _add_tile_transient(self,
                            sdfg: dace.SDFG,
                            base: str,
                            dtype: dace.dtypes.typeclass,
                            widths: Tuple[int, ...]) -> str:
        """Add a fresh tile-shape register transient and return its name.

        :param sdfg: SDFG receiving the new array.
        :param base: Base name; ``"_<idx>"`` is appended on collision.
        :param dtype: Element type.
        :param widths: Tile shape.
        :returns: The added array name.
        """
        name = base
        idx = 0
        while name in sdfg.arrays:
            idx += 1
            name = f"{base}_{idx}"
        sdfg.add_array(
            name,
            list(widths),
            dtype,
            storage=dace.dtypes.StorageType.Register,
            transient=True,
        )
        return name

    def _emit_tile_load(self,
                        state: dace.SDFGState,
                        tasklet: dace.nodes.Tasklet,
                        conn: str,
                        source_edge,
                        in_edge,
                        per_iter_subset: subsets.Range,
                        dim_strides: Tuple[int, ...],
                        spec: TileDimSpec,
                        mask_name: str) -> str:
        """Insert a :class:`TileLoad` materializing one tile transient.

        :param state: Parent state.
        :param tasklet: Original tasklet (for label scoping).
        :param conn: Original input connector name.
        :param source_edge: Outermost memlet-path edge (its source is
            the original ``AccessNode``).
        :param per_iter_subset: Per-iteration subset (``A[i, j]``) used
            as the base for the tile-region expansion.
        :param dim_strides: Per-tile-dim stride coefficients.
        :param spec: Surrounding tile spec.
        :param mask_name: Name of the iteration-mask transient.
        :returns: Name of the produced tile transient.
        """
        src_name = source_edge.data.data
        sdfg = state.sdfg
        src_arr = sdfg.arrays[src_name]
        tile_name = self._add_tile_transient(
            sdfg, f"_tile_{conn.lstrip('_')}", src_arr.dtype, spec.widths,
        )
        load = TileLoad(
            name=f"{tasklet.label}_load_{conn.lstrip('_')}",
            widths=spec.widths,
            dim_strides=tuple(dim_strides),
            has_mask=True,
        )
        state.add_node(load)
        promoted_subset = _tile_region_subset(per_iter_subset, spec.iter_vars, spec.widths)
        state.add_edge(in_edge.src, in_edge.src_conn, load, "_src",
                       dace.Memlet(data=src_name, subset=promoted_subset))
        mask_access = self._find_mask_access(state, mask_name) or state.add_access(mask_name)
        subset = ", ".join(f"0:{w}" for w in spec.widths)
        state.add_edge(mask_access, None, load, "_mask",
                       dace.Memlet(f"{mask_name}[{subset}]"))
        tile_access = state.add_access(tile_name)
        state.add_edge(load, "_dst", tile_access, None, dace.Memlet(f"{tile_name}[{subset}]"))
        return tile_name, tile_access

    def _find_mask_access(self, state: dace.SDFGState, mask_name: str) -> Optional[dace.nodes.AccessNode]:
        """Return the AccessNode for ``mask_name`` placed by
        :class:`GenerateTileIterationMask`, if present.

        :param state: Parent state.
        :param mask_name: Name of the iteration-mask transient.
        :returns: The producer-fed access node, or ``None``.
        """
        producers = [
            n for n in state.data_nodes()
            if n.data == mask_name and state.in_edges(n)
        ]
        return producers[0] if producers else None

    def _drop_mask_placeholder_edge(self, state: dace.SDFGState, mask_name: str, map_entry: MapEntry) -> None:
        """Remove the placeholder empty-memlet edge
        ``mask_access -> MapExit`` that
        :class:`GenerateTileIterationMask` added to keep the
        intermediate SDFG scope-valid.

        :param state: Parent state.
        :param mask_name: Name of the iteration-mask transient.
        :param map_entry: Inner map entry whose exit holds the placeholder.
        """
        mask_access = self._find_mask_access(state, mask_name)
        if mask_access is None:
            return
        map_exit = state.exit_node(map_entry)
        for e in list(state.out_edges(mask_access)):
            if e.dst is map_exit and (e.data is None or e.data.data is None):
                state.remove_edge(e)

    def _rewrite_one_map(self,
                         state: dace.SDFGState,
                         map_entry: MapEntry,
                         spec: TileDimSpec) -> None:
        """Replace the body of ``map_entry`` with a tile-op chain.

        :param state: Parent state.
        :param map_entry: Inner map being rewritten.
        :param spec: Per-dim tile specification.
        :raises NotImplementedError: For shapes T5 MVP doesn't handle.
        """
        tasklets = self._find_body_tasklets(state, map_entry)
        if len(tasklets) != 1:
            raise NotImplementedError(
                f"EmitTileOps: map {map_entry.label!r} body has {len(tasklets)} tasklets; "
                f"T5 MVP requires exactly 1. Run SplitTasklets first."
            )
        tasklet = tasklets[0]
        out_conn, a_conn, op, b_conn = self._classify_binop_tasklet(tasklet)
        kind_a, info_a = self._operand_kind(state, tasklet, a_conn, spec)
        kind_b, info_b = self._operand_kind(state, tasklet, b_conn, spec)
        if kind_a == "Symbol" and kind_b == "Symbol":
            raise NotImplementedError(
                f"EmitTileOps: tasklet {tasklet.label!r} is Symbol/Symbol; "
                f"belongs outside the tile path."
            )

        out_e = [e for e in state.out_edges(tasklet) if e.src_conn == out_conn]
        if len(out_e) != 1:
            raise NotImplementedError(
                f"EmitTileOps: tasklet {tasklet.label!r} out {out_conn!r} has "
                f"{len(out_e)} out-edges; expected exactly 1"
            )
        out_edge = out_e[0]
        out_path = state.memlet_path(out_edge)
        out_sink_edge = out_path[-1]
        out_dst_name = out_edge.data.data
        out_dst_arr = state.sdfg.arrays[out_dst_name]
        out_promoted_subset = _tile_region_subset(out_edge.data.subset, spec.iter_vars, spec.widths)

        mask_name = TileNameScheme.ITER_MASK

        a_tile = (
            self._emit_tile_load(state, tasklet, a_conn, info_a[0], info_a[1], info_a[2], info_a[3],
                                 spec, mask_name)
            if kind_a == "Tile" else (None, None)
        )
        b_tile = (
            self._emit_tile_load(state, tasklet, b_conn, info_b[0], info_b[1], info_b[2], info_b[3],
                                 spec, mask_name)
            if kind_b == "Tile" else (None, None)
        )
        a_tile_name, a_tile_access = a_tile
        b_tile_name, b_tile_access = b_tile

        out_tile_name = self._add_tile_transient(
            state.sdfg, "_tile_out", out_dst_arr.dtype, spec.widths,
        )
        binop_kwargs = dict(
            name=f"{tasklet.label}_binop",
            widths=spec.widths,
            op=op,
            has_mask=True,
            kind_a=kind_a,
            kind_b=kind_b,
        )
        if kind_a == "Symbol":
            binop_kwargs["expr_a"] = info_a[0]
        if kind_b == "Symbol":
            binop_kwargs["expr_b"] = info_b[0]
        binop = TileBinop(**binop_kwargs)
        state.add_node(binop)

        subset = ", ".join(f"0:{w}" for w in spec.widths)
        if kind_a == "Tile":
            state.add_edge(a_tile_access, None, binop, "_a", dace.Memlet(f"{a_tile_name}[{subset}]"))
        if kind_b == "Tile":
            state.add_edge(b_tile_access, None, binop, "_b", dace.Memlet(f"{b_tile_name}[{subset}]"))
        mask_access = self._find_mask_access(state, mask_name) or state.add_access(mask_name)
        state.add_edge(mask_access, None, binop, "_mask",
                       dace.Memlet(f"{mask_name}[{subset}]"))

        out_access = state.add_access(out_tile_name)
        state.add_edge(binop, "_c", out_access, None, dace.Memlet(f"{out_tile_name}[{subset}]"))

        store = TileStore(name=f"{tasklet.label}_store", widths=spec.widths, has_mask=True)
        state.add_node(store)
        state.add_edge(out_access, None, store, "_src", dace.Memlet(f"{out_tile_name}[{subset}]"))
        state.add_edge(mask_access, None, store, "_mask",
                       dace.Memlet(f"{mask_name}[{subset}]"))
        state.add_edge(store, "_dst", out_edge.dst, out_edge.dst_conn,
                       dace.Memlet(data=out_dst_name, subset=out_promoted_subset))

        for e in list(state.in_edges(tasklet)) + list(state.out_edges(tasklet)):
            state.remove_edge(e)
        state.remove_node(tasklet)
        self._drop_mask_placeholder_edge(state, mask_name, map_entry)

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Optional[Dict]) -> Optional[int]:
        """Walk every K-dim eligible inner map and emit the tile-op chain.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Reads ``"MarkTileDims"`` when present.
        :returns: Number of inner maps rewritten, or ``None`` if none.
        """
        specs: Optional[Dict[MapEntry, TileDimSpec]] = None
        if pipeline_results and "MarkTileDims" in pipeline_results:
            specs = pipeline_results["MarkTileDims"]
        K = len(self.widths)
        rewritten = 0
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, MapEntry) or not isinstance(g, dace.SDFGState):
                continue
            if not is_innermost_map(g, n):
                continue
            if specs is not None and n not in specs:
                continue
            if len(n.map.params) < K:
                continue
            if TileNameScheme.ITER_MASK not in g.sdfg.arrays:
                raise NotImplementedError(
                    f"EmitTileOps: map {n.label!r} parent SDFG lacks "
                    f"{TileNameScheme.ITER_MASK!r}; run GenerateTileIterationMask first."
                )
            spec = specs[n] if specs is not None and n in specs else self._spec_for(n)
            self._rewrite_one_map(g, n, spec)
            rewritten += 1
        return rewritten or None
