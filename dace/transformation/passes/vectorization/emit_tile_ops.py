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
import copy
import re
from typing import Dict, List, Optional, Tuple

import dace
from dace import properties, subsets, symbolic
from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl
from dace.libraries.tileops import TileBinop, TileLoad, TileStore
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.tile_dims import (
    TileAccessKind,
    TileDimSpec,
    classify_tile_access,
)

_OPERAND = r"(?:[A-Za-z_]\w*|\d+\.?\d*(?:[eE][+-]?\d+)?)"
_BINOP_RE = re.compile(
    rf"^\s*(?P<out>\w+)\s*=\s*"
    rf"(?:\(\s*)?(?P<a>{_OPERAND})\s*(?P<op>\+|-|\*|/|<=|>=|==|!=|<|>|and|or)\s*"
    rf"(?P<b>{_OPERAND})\s*\)?\s*;?\s*$"
)


def _is_numeric_literal(token: str) -> bool:
    """Return True iff ``token`` is a numeric literal (not an identifier).

    :param token: A regex-captured operand token.
    :returns: True for ``2``, ``0.2``, ``1e-5`` etc.; False for
        identifier-shaped tokens.
    """
    try:
        float(token)
        return True
    except ValueError:
        return False


_PY_TO_TILEBINOP_OP = {
    "+": "+", "-": "-", "*": "*", "/": "/",
    "<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "==", "!=": "!=",
    "and": "&&", "or": "||",
}
_ASSIGN_RE = re.compile(r"^\s*(?P<out>\w+)\s*=\s*(?P<inp>\w+)\s*;?\s*$")


def _is_assign_tasklet(tasklet: dace.nodes.Tasklet) -> bool:
    """Return True iff the tasklet body is a trivial ``out = inp`` copy.

    The frontend emits these at the end of multi-tasklet chains (post-
    ``SplitTasklets``) to route an intermediate transient to the outer
    access node; they are pass-throughs for the tile-op rewrite.

    :param tasklet: Tasklet to inspect.
    :returns: True iff ``tasklet.code`` matches ``out = inp``.
    """
    body = tasklet.code.as_string.strip().rstrip(";").strip()
    return _ASSIGN_RE.match(body) is not None


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
        :returns: ``("Tile", (source_edge, edge, subset, dim_strides))``
            for a tile-region access; ``("Scalar", (source_edge, edge))``
            for a length-1 / ``dace.data.Scalar`` array read (broadcast);
            ``("Symbol", (expr_str,))`` for a true free-symbol read.
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
            # Both lower to a strided ``TileLoad``: a perfect-box affine
            # access (each tile dim maps to one distinct array dim) addressed
            # through ``match_dims`` so a transposed / non-last mapping
            # (``cc[j, i]``) steps along the correct axis. Non-box accesses
            # (indirect / diagonal) fall through to GATHER below.
            return "Tile", (source_edge, edge, edge.data.subset, cls.dim_strides, cls.match_dims)
        if cls.kind == TileAccessKind.BROADCAST_SYMBOL:
            # A no-tile-dependency access reads either a length-1 / Scalar
            # array (route through a connector and broadcast) or a true
            # free symbol (embed inline). The data is an array name in
            # both cases, but only a genuine Scalar / 1-element array
            # should be wired; otherwise treat as an inline symbol.
            return "Scalar", (source_edge, edge)
        raise NotImplementedError(
            f"EmitTileOps: tasklet {tasklet.label!r} input {conn_name!r} access "
            f"{source_edge.data!r} is {cls.kind.value}; T5 MVP only handles Tile / Scalar / Symbol"
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
                        src_dims: Tuple[int, ...],
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
            src_dims=tuple(src_dims),
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

    def _mask_name_for_map(self, state: dace.SDFGState, map_entry: MapEntry) -> str:
        """Return the iteration-mask transient name produced inside
        ``map_entry``'s scope.

        Multiple maps in one state each have their own per-map mask
        (``_tile_iter_mask``, ``_tile_iter_mask_1``, ...), so the name
        must be read from THIS map's scope, not assumed global.

        :param state: Parent state.
        :param map_entry: Inner map entry.
        :returns: The mask array name produced inside the map scope.
        :raises NotImplementedError: If the scope has no mask producer.
        """
        from dace.libraries.tileops import TileMaskGen
        scope = state.all_nodes_between(map_entry, state.exit_node(map_entry)) or set()
        for node in scope:
            if isinstance(node, TileMaskGen):
                oe = [e for e in state.out_edges(node) if e.src_conn == "_o"]
                if oe:
                    return oe[0].data.data
        raise NotImplementedError(
            f"EmitTileOps: map {map_entry.label!r} has no TileMaskGen in scope; "
            f"run GenerateTileIterationMask first.")

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

    def _walk_through_assigns(self, state: dace.SDFGState, start_edge, assign_tasklets):
        """Walk forward from ``start_edge`` skipping trivial-assign
        tasklets and AccessNode intermediaries until the edge into the
        ``MapExit`` is reached.

        :param state: Parent state.
        :param start_edge: The binop tasklet's out-edge.
        :param assign_tasklets: Body tasklets known to be trivial
            ``out = inp`` copies (the walk may traverse these).
        :returns: ``(final_edge, intermediates)`` — the edge whose
            ``dst`` is the ``MapExit`` (or ``start_edge`` if no chain
            follows), plus the list of intermediate AccessNodes
            traversed (so the caller can remove them and never leave an
            isolated node).
        """
        assign_set = set(assign_tasklets)
        edge = start_edge
        intermediates = []
        seen = set()
        while id(edge) not in seen:
            seen.add(id(edge))
            dst = edge.dst
            if isinstance(dst, dace.nodes.MapExit):
                return edge, intermediates
            if isinstance(dst, dace.nodes.AccessNode):
                nxt = list(state.out_edges(dst))
                if len(nxt) != 1:
                    return edge, intermediates
                intermediates.append(dst)
                edge = nxt[0]
                continue
            if isinstance(dst, dace.nodes.Tasklet) and dst in assign_set:
                nxt = list(state.out_edges(dst))
                if len(nxt) != 1:
                    return edge, intermediates
                edge = nxt[0]
                continue
            return edge, intermediates
        return edge, intermediates

    def _wire_scalar_operand(self,
                             state: dace.SDFGState,
                             binop,
                             conn: str,
                             source_edge,
                             in_edge) -> None:
        """Wire a Scalar (length-1 / ``dace.data.Scalar``) operand into
        ``binop`` via its ``_a`` / ``_b`` connector, reusing the original
        in-edge's source (the MapEntry pass-through) so no connector is
        orphaned.

        :param state: Parent state.
        :param binop: The ``TileBinop`` lib node.
        :param conn: ``"_a"`` or ``"_b"`` — the scalar connector.
        :param source_edge: Outermost memlet-path edge (its source is
            the original AccessNode).
        :param in_edge: The original binop tasklet's in-edge for this
            operand (carries the MapEntry pass-through source + conn).
        """
        state.add_edge(in_edge.src, in_edge.src_conn, binop, conn,
                       copy.deepcopy(in_edge.data))

    def _drop_dangling_scope_connectors(self, state: dace.SDFGState, map_entry: MapEntry) -> None:
        """Remove MapEntry / MapExit pass-through connectors that lost
        their consumer / producer after the body rewrite.

        A ``MapEntry.OUT_<x>`` connector with no outgoing edge (its only
        reader was the removed tasklet) is dropped together with the
        matching ``IN_<x>`` connector and its source in-edge. Symmetric
        for ``MapExit.IN_<x>`` with no incoming edge.

        :param state: Parent state.
        :param map_entry: Inner map whose scope connectors are cleaned.
        """
        map_exit = state.exit_node(map_entry)
        # MapEntry: drop OUT_<x> with no out-edge + its paired IN_<x>.
        for out_conn in list(map_entry.out_connectors):
            if any(e.src_conn == out_conn for e in state.out_edges(map_entry)):
                continue
            map_entry.remove_out_connector(out_conn)
            in_conn = "IN_" + out_conn[len("OUT_"):] if out_conn.startswith("OUT_") else None
            if in_conn and in_conn in map_entry.in_connectors:
                for e in list(state.in_edges(map_entry)):
                    if e.dst_conn == in_conn:
                        state.remove_edge(e)
                map_entry.remove_in_connector(in_conn)
        # MapExit: drop IN_<x> with no in-edge + its paired OUT_<x>.
        for in_conn in list(map_exit.in_connectors):
            if any(e.dst_conn == in_conn for e in state.in_edges(map_exit)):
                continue
            map_exit.remove_in_connector(in_conn)
            out_conn = "OUT_" + in_conn[len("IN_"):] if in_conn.startswith("IN_") else None
            if out_conn and out_conn in map_exit.out_connectors:
                for e in list(state.out_edges(map_exit)):
                    if e.src_conn == out_conn:
                        state.remove_edge(e)
                map_exit.remove_out_connector(out_conn)

    def _binop_output_data(self, state: dace.SDFGState, binop: dace.nodes.Tasklet) -> Optional[str]:
        """Return the data name the binop tasklet writes to (its sole
        out-edge's data).

        :param state: Parent state.
        :param binop: A binop tasklet.
        :returns: The output data name, or ``None`` if no out-edge.
        """
        oe = list(state.out_edges(binop))
        return oe[0].data.data if oe else None

    def _topo_order_binops(self,
                           state: dace.SDFGState,
                           binops: List[dace.nodes.Tasklet]) -> List[dace.nodes.Tasklet]:
        """Order body binop tasklets so each is emitted after the binops
        that produce its intermediate inputs.

        :param state: Parent state.
        :param binops: The non-assign body tasklets.
        :returns: ``binops`` in dataflow (dependency-respecting) order.
        :raises NotImplementedError: If a dependency cycle is detected.
        """
        out_data = {b: self._binop_output_data(state, b) for b in binops}
        produced_by_binop = {d for d in out_data.values() if d is not None}
        produced: set = set()
        remaining = list(binops)
        ordered: List[dace.nodes.Tasklet] = []
        while remaining:
            progressed = False
            for b in list(remaining):
                deps = {
                    e.data.data for e in state.in_edges(b)
                    if isinstance(e.src, dace.nodes.AccessNode) and e.data.data in produced_by_binop
                    and e.data.data != out_data[b]
                }
                if deps <= produced:
                    ordered.append(b)
                    if out_data[b] is not None:
                        produced.add(out_data[b])
                    remaining.remove(b)
                    progressed = True
            if not progressed:
                raise NotImplementedError(
                    "EmitTileOps: cyclic / unresolvable binop dependency in map body"
                )
        return ordered

    def _emit_one_binop(self,
                        state: dace.SDFGState,
                        tasklet: dace.nodes.Tasklet,
                        spec: TileDimSpec,
                        tile_map: Dict[str, Tuple[str, dace.nodes.AccessNode]],
                        mask_name: str) -> Tuple[str, dace.nodes.AccessNode]:
        """Emit a single :class:`TileBinop` for one body tasklet.

        Resolves each operand to a Tile (existing intermediate tile from
        ``tile_map`` or a fresh :class:`TileLoad`), a Scalar (wired
        through), or a Symbol (numeric literal embedded inline), then
        writes the result to a fresh tile transient.

        :param state: Parent state.
        :param tasklet: The binop tasklet.
        :param spec: Per-dim tile specification.
        :param tile_map: Intermediate-transient-data-name → ``(tile_name,
            tile_access)``; updated by the caller with this binop's output.
        :param mask_name: Iteration-mask transient name.
        :returns: ``(out_data_name, out_tile_access)`` — the data name
            this tasklet wrote to, and the result tile's access node.
        :raises NotImplementedError: For operand shapes T5 doesn't handle.
        """
        out_conn, a_tok, op, b_tok = self._classify_binop_tasklet(tasklet)
        subset = ", ".join(f"0:{w}" for w in spec.widths)

        def _resolve(token, conn_label):
            """Return ``(kind, payload)`` for one operand token."""
            if _is_numeric_literal(token):
                return "Symbol", token
            in_edges = [e for e in state.in_edges(tasklet) if e.dst_conn == token]
            if len(in_edges) != 1:
                raise NotImplementedError(
                    f"EmitTileOps: tasklet {tasklet.label!r} operand {token!r} has "
                    f"{len(in_edges)} in-edges")
            edge = in_edges[0]
            src_data = edge.data.data
            if isinstance(edge.src, dace.nodes.AccessNode) and src_data in tile_map:
                return "Tile-existing", tile_map[src_data][1]
            kind, info = self._operand_kind(state, tasklet, token, spec)
            return kind, info

        kind_a, info_a = _resolve(a_tok, "_a")
        kind_b, info_b = _resolve(b_tok, "_b")
        norm_a = "Tile" if kind_a == "Tile-existing" else kind_a
        norm_b = "Tile" if kind_b == "Tile-existing" else kind_b
        if norm_a != "Tile" and norm_b != "Tile":
            raise NotImplementedError(
                f"EmitTileOps: tasklet {tasklet.label!r} has no Tile operand "
                f"({norm_a}/{norm_b})")

        out_dtype = state.sdfg.arrays[self._binop_output_data(state, tasklet)].dtype
        out_tile_name = self._add_tile_transient(state.sdfg, "_tile_t", out_dtype, spec.widths)
        kwargs = dict(name=f"{tasklet.label}_binop", widths=spec.widths, op=op,
                      has_mask=True, kind_a=norm_a, kind_b=norm_b)
        if norm_a == "Symbol":
            kwargs["expr_a"] = info_a
        if norm_b == "Symbol":
            kwargs["expr_b"] = info_b
        binop = TileBinop(**kwargs)
        state.add_node(binop)

        def _wire(kind, info, conn):
            if kind == "Tile-existing":
                state.add_edge(info, None, binop, conn, dace.Memlet(f"{info.data}[{subset}]"))
            elif kind == "Tile":
                tname, tacc = self._emit_tile_load(
                    state, tasklet, conn[-1], info[0], info[1], info[2], info[3], info[4], spec, mask_name)
                state.add_edge(tacc, None, binop, conn, dace.Memlet(f"{tname}[{subset}]"))
            elif kind == "Scalar":
                self._wire_scalar_operand(state, binop, conn, info[0], info[1])
            # Symbol: nothing to wire (embedded inline).

        _wire(kind_a, info_a, "_a")
        _wire(kind_b, info_b, "_b")
        mask_access = self._find_mask_access(state, mask_name) or state.add_access(mask_name)
        state.add_edge(mask_access, None, binop, "_mask", dace.Memlet(f"{mask_name}[{subset}]"))
        out_access = state.add_access(out_tile_name)
        state.add_edge(binop, "_c", out_access, None, dace.Memlet(f"{out_tile_name}[{subset}]"))
        return self._binop_output_data(state, tasklet), out_access

    def _rewrite_one_map(self,
                         state: dace.SDFGState,
                         map_entry: MapEntry,
                         spec: TileDimSpec) -> None:
        """Replace the body of ``map_entry`` with a tile-op chain.

        Walks the body's binop tasklets in dataflow order, emitting one
        :class:`TileBinop` each (operands resolved against a running
        ``tile_map`` of intermediate tiles), then a :class:`TileStore`
        for each binop whose output flows — through trivial assigns — to
        the ``MapExit``.

        :param state: Parent state.
        :param map_entry: Inner map being rewritten.
        :param spec: Per-dim tile specification.
        :raises NotImplementedError: For shapes T5 MVP doesn't handle.
        """
        tasklets = self._find_body_tasklets(state, map_entry)
        binops = [t for t in tasklets if not _is_assign_tasklet(t)]
        assign_tasklets = [t for t in tasklets if _is_assign_tasklet(t)]
        if not binops:
            raise NotImplementedError(
                f"EmitTileOps: map {map_entry.label!r} body has no binop tasklet")

        mask_name = self._mask_name_for_map(state, map_entry)
        subset = ", ".join(f"0:{w}" for w in spec.widths)
        ordered = self._topo_order_binops(state, binops)

        tile_map: Dict[str, Tuple[str, dace.nodes.AccessNode]] = {}
        stores: List[Tuple[dace.nodes.AccessNode, object]] = []
        all_intermediates: set = set()
        for t in ordered:
            out_e = list(state.out_edges(t))
            out_data, out_access = self._emit_one_binop(state, t, spec, tile_map, mask_name)
            tile_map[out_data] = (out_access.data, out_access)
            final_edge, inters = self._walk_through_assigns(state, out_e[0], assign_tasklets)
            all_intermediates |= set(inters)
            if isinstance(final_edge.dst, dace.nodes.MapExit):
                stores.append((out_access, final_edge))

        for out_access, out_edge in stores:
            out_dst_name = out_edge.data.data
            out_arr = state.sdfg.arrays[out_dst_name]
            out_cls = classify_tile_access(out_edge.data.subset, tuple(out_arr.strides), spec.iter_vars)
            if out_cls.kind not in (TileAccessKind.CONTIGUOUS, TileAccessKind.STRIDED):
                # A non-box output (indirect / diagonal scatter) needs a
                # TileScatter, not a strided store — out of scope here.
                raise NotImplementedError(
                    f"EmitTileOps: output {out_dst_name!r} access is {out_cls.kind.value}; "
                    f"only perfect-box (strided) stores are supported (scatter is TODO)")
            promoted = _tile_region_subset(out_edge.data.subset, spec.iter_vars, spec.widths)
            store = TileStore(name=f"{out_access.data}_store", widths=spec.widths,
                              dim_strides=out_cls.dim_strides, dst_dims=out_cls.match_dims, has_mask=True)
            state.add_node(store)
            state.add_edge(out_access, None, store, "_src",
                           dace.Memlet(f"{out_access.data}[{subset}]"))
            mask_access = self._find_mask_access(state, mask_name) or state.add_access(mask_name)
            state.add_edge(mask_access, None, store, "_mask",
                           dace.Memlet(f"{mask_name}[{subset}]"))
            state.add_edge(store, "_dst", out_edge.dst, out_edge.dst_conn,
                           dace.Memlet(data=out_dst_name, subset=promoted))

        # Remove the original body tasklets + the intermediate transients
        # that only connected them — scoped, so no isolated node remains.
        for node in list(binops) + list(assign_tasklets) + list(all_intermediates):
            for e in list(state.in_edges(node)) + list(state.out_edges(node)):
                state.remove_edge(e)
            if node in state.nodes():
                state.remove_node(node)
        self._drop_dangling_scope_connectors(state, map_entry)
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
            # Verify this map's scope has its own mask producer (raises
            # NotImplementedError otherwise).
            self._mask_name_for_map(g, n)
            spec = specs[n] if specs is not None and n in specs else self._spec_for(n)
            self._rewrite_one_map(g, n, spec)
            rewritten += 1
        return rewritten or None
