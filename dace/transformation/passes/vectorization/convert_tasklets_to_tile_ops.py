# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Convert in-body tasklets to ``TileBinop`` / ``TileUnop`` / ``TileITE``.

After :class:`StageInsideBody` walks every tile-tagged body NSDFG and
stages non-transient AccessNode reads through tile transients, the
body still holds raw tasklets that operate on per-lane scalar values.
This pass walks the same body NSDFGs and replaces each tasklet with
the corresponding tile lib node so the post-expansion pure-loop body
operates on tile-shape register transients (design section 5.1 +
section 6.7).

First-slice scope (this commit):

* BINARY tasklets whose body is exactly ``_o = _a <op> _b`` with
  ``<op>`` in :data:`_SUPPORTED_BINOPS`. Both operands must be Tile
  (the walker stages them through ``TileLoad`` bridges).
* Connector wiring: tasklet ``_a``/``_b``/``_o`` -> :class:`TileBinop`
  ``_a``/``_b``/``_c``; memlet on each edge preserved.

Deferred to subsequent slices:

* UNARY tasklets -> :class:`TileUnop` (mechanical follow-up).
* TERNARY / merge tasklets -> :class:`TileITE`.
* Reduction tasklets -> :class:`TileReduce`.
* Scalar / Symbol operand kinds (currently only Tile + Tile binops).
* Tasklets nested inside multi-state bodies / RMW chains.
"""
from typing import Any, Dict, Optional, Tuple

import dace
from dace import properties
from dace.libraries.tileops import TileBinop, TileITE, TileMaskGen, TileReduce, TileUnop
from dace.sdfg import SDFG
from dace.sdfg.nodes import MapEntry, NestedSDFG, Tasklet
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map

#: Subset of binary operators that map directly onto :class:`TileBinop`.
_SUPPORTED_BINOPS = {"+", "-", "*", "/", "min", "max"}

#: Subset of reduction operators that map directly onto :class:`TileReduce`.
#: Subset of :data:`_SUPPORTED_BINOPS` excluding non-associative ``-`` and ``/``.
_SUPPORTED_REDUCE_OPS = {"+", "*", "min", "max"}

#: Mapping ``tasklet-body form`` -> ``TileUnop op label``. Each value names the
#: ``TileUnop.op`` keyword to instantiate; each key form is matched against the
#: tasklet body (after the DaCe paren wrap is stripped). The leading ``-`` form
#: aliases ``neg``.
_SUPPORTED_UNOPS = {
    "neg",
    "abs",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "floor",
    "ceil",
    "tanh",
}


@properties.make_properties
@transformation.explicit_cf_compatible
class ConvertTaskletsToTileOps(ppl.Pass):
    """Convert in-body tasklets to tile lib nodes (first slice: binary Tile+Tile only).

    :ivar widths: Per-tile-dim widths; mirrors :class:`StageInsideBody`.
    """

    CATEGORY: str = "Vectorization"

    widths = properties.Property(
        dtype=tuple,
        default=(8, ),
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )

    def __init__(self, widths: Tuple[int, ...] = (8, )) -> None:
        """Build the pass.

        :param widths: Per-tile-dim widths.
        :raises ValueError: If ``widths`` length is not in ``{1, 2, 3}``.
        """
        super().__init__()
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"ConvertTaskletsToTileOps: widths length {len(widths)} not in {{1, 2, 3}}")
        self.widths = tuple(widths)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _find_mask_an(self, inner_state: SDFGState):
        """Find the AccessNode that :class:`TileMaskGen` WRITES to (its ``_o`` target).

        Every lib-node ``_mask`` consumer reads from this SAME AccessNode so the SDFG
        scheduler orders TileMaskGen before the consumers. Returns ``None`` when no
        iteration mask is in scope (the divisible / unmasked case).
        """
        from dace.sdfg.nodes import AccessNode
        for n in inner_state.nodes():
            if not isinstance(n, TileMaskGen):
                continue
            for out_edge in inner_state.out_edges(n):
                if out_edge.src_conn == "_o" and isinstance(out_edge.dst, AccessNode):
                    return out_edge.dst
        return None

    def _wire_mask(self, inner_state: SDFGState, lib_node, mask_an) -> None:
        """If ``mask_an`` is non-None, wire ``mask_an -> lib_node._mask`` with a full-tile
        subset memlet matching the mask's widths shape."""
        if mask_an is None:
            return
        subset = ", ".join(f"0:{w}" for w in self.widths)
        inner_state.add_edge(mask_an, None, lib_node, "_mask", dace.Memlet(f"{mask_an.data}[{subset}]"))

    def _body_nsdfgs(self, sdfg: SDFG):
        """Yield ``(state, nsdfg_node, map_entry)`` for every tile-tagged body NSDFG.

        Mirror of the walker shape used by :class:`StageInsideBody` and
        :class:`PreparePerLaneIndices`.
        """
        K = len(self.widths)
        for node, parent in sdfg.all_nodes_recursive():
            if not isinstance(node, MapEntry):
                continue
            if not isinstance(parent, SDFGState):
                continue
            try:
                if not is_innermost_map(parent, node):
                    continue
            except (StopIteration, ValueError):
                continue
            if len(node.map.params) < K:
                continue
            try:
                scope_nodes = parent.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
            except (StopIteration, ValueError):
                continue
            nsdfgs = [n for n in scope_nodes if isinstance(n, NestedSDFG)]
            if len(nsdfgs) != 1:
                continue
            yield parent, nsdfgs[0], node

    def _detect_binop(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str, str]]:
        """If ``tasklet`` is a simple binary ``_out = _a <op> _b`` body, return
        ``(out_conn, a_conn, b_conn, op)``. Otherwise ``None``.
        """
        if len(tasklet.in_connectors) != 2 or len(tasklet.out_connectors) != 1:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        in_conns = list(tasklet.in_connectors)
        # Try every op in _SUPPORTED_BINOPS against the two operand orderings. DaCe wraps the
        # RHS of an assignment tasklet in parentheses (``_o = (_a + _b)``); accept both forms.
        for op in _SUPPORTED_BINOPS:
            for a, b in (in_conns, list(reversed(in_conns))):
                if op in ("min", "max"):
                    forms = (f"{out_conn} = {op}({a}, {b})", f"{out_conn} = ({op}({a}, {b}))")
                else:
                    forms = (f"{out_conn} = {a} {op} {b}", f"{out_conn} = ({a} {op} {b})")
                if body in forms:
                    return out_conn, a, b, op
        return None

    def _detect_binop_with_symbol(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str, str, str]]:
        """Detect a binop with ONE Tile/Scalar operand and ONE Symbol operand.

        Matches ``_out = _a <op> <expr>`` or ``_out = <expr> <op> _a`` where ``<expr>``
        is an outer-scope symbol or numeric literal (not a tasklet connector). Returns
        ``(out_conn, tile_or_scalar_conn, op, symbol_side, symbol_expr)`` where
        ``symbol_side`` is "a" (symbol on the LHS) or "b" (symbol on the RHS) of the op.

        This complements the connector-counting detector ``_detect_binop`` which
        requires two in-connectors. The Symbol case has only ONE in-connector but is
        still a binop (the second operand is an inline string).
        """
        if len(tasklet.in_connectors) != 1 or len(tasklet.out_connectors) != 1:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        a_conn = next(iter(tasklet.in_connectors))
        # Strip outer parens (DaCe wraps the RHS).
        rhs = body[len(f"{out_conn} = "):]
        if rhs.startswith("(") and rhs.endswith(")"):
            rhs = rhs[1:-1]
        # Try every op; for each, see whether the rhs splits into ``<a_conn> <op> <expr>``
        # or ``<expr> <op> <a_conn>``. Function-form ops (min/max) handled via the
        # explicit ``min(x, y)`` parsing below.
        for op in _SUPPORTED_BINOPS:
            if op in ("min", "max"):
                # min(_a, expr) / min(expr, _a) -- accept both orderings.
                for sym_side in ("b", "a"):
                    if sym_side == "b":
                        prefix, sep = f"{op}({a_conn}, ", ")"
                    else:
                        prefix, sep = f"{op}(", f", {a_conn})"
                    if rhs.startswith(prefix) and rhs.endswith(sep):
                        expr = rhs[len(prefix):-len(sep)] if sep else rhs[len(prefix):]
                        expr = expr.strip()
                        if expr and a_conn not in expr.split():
                            return out_conn, a_conn, op, sym_side, expr
            else:
                sep = f" {op} "
                if sep not in rhs:
                    continue
                lhs_part, rhs_part = rhs.split(sep, 1)
                if lhs_part.strip() == a_conn:
                    # Form ``_a <op> <expr>``; symbol on side b.
                    expr = rhs_part.strip()
                    if expr and a_conn not in expr.split():
                        return out_conn, a_conn, op, "b", expr
                if rhs_part.strip() == a_conn:
                    # Form ``<expr> <op> _a``; symbol on side a.
                    expr = lhs_part.strip()
                    if expr and a_conn not in expr.split():
                        return out_conn, a_conn, op, "a", expr
        return None

    def _detect_unop(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str]]:
        """If ``tasklet`` is a simple unary ``_out = <op>(_a)`` body (or ``_out = -_a``),
        return ``(out_conn, a_conn, op_label)``. Otherwise ``None``.

        Accepts both the DaCe-emitted parenthesised RHS form and the bare form.
        ``op_label`` is one of :data:`_SUPPORTED_UNOPS`.
        """
        if len(tasklet.in_connectors) != 1 or len(tasklet.out_connectors) != 1:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        a_conn = next(iter(tasklet.in_connectors))
        # Negation: ``_o = -_a`` / ``_o = (-_a)`` / ``_o = (- _a)`` (DaCe sometimes inserts a
        # space between the unary minus and the operand).
        for form in (f"{out_conn} = -{a_conn}", f"{out_conn} = (-{a_conn})", f"{out_conn} = (- {a_conn})"):
            if body == form:
                return out_conn, a_conn, "neg"
        # Function-call ops: ``_o = abs(_a)`` / ``_o = math.exp(_a)`` / etc. Accept both bare
        # name and ``math.`` / ``std::`` prefixes (RemoveMathCall strips ``math.`` upstream;
        # this handles the case where it didn't run or wasn't needed).
        for op in _SUPPORTED_UNOPS:
            if op == "neg":
                continue
            prefixes = (op, f"math.{op}", f"std::{op}")
            for pref in prefixes:
                for form in (f"{out_conn} = {pref}({a_conn})", f"{out_conn} = ({pref}({a_conn}))"):
                    if body == form:
                        return out_conn, a_conn, op
        return None

    def _operand_kind(self, inner_state: SDFGState, edge) -> str:
        """Classify the operand kind for the lib node based on the source descriptor.

        Per design section 6.5: a CONSTANT-only staged source is a Scalar bridge or a
        length-1 Array; the lib node consuming it sees ``kind="Scalar"`` and emits a
        hardware splat to fill the lane register. Tile-shape staged sources (LINEAR /
        AFFINE / REPLICATE / MODULAR / GATHER) wire as ``kind="Tile"``.

        :param inner_state: Inner SDFGState holding the edge.
        :param edge: The wired in-edge to the tasklet / lib node connector.
        :returns: ``"Scalar"`` if the source's descriptor is a Scalar or a length-1
            Array; ``"Tile"`` otherwise.
        """
        import dace.data as dd
        if edge.data is None or edge.data.data is None:
            return "Tile"
        try:
            desc = inner_state.sdfg.arrays[edge.data.data]
        except KeyError:
            return "Tile"
        if isinstance(desc, dd.Scalar):
            return "Scalar"
        if isinstance(desc, dd.Array):
            shape = tuple(desc.shape)
            # Single-element Array (shape == (1,) or all 1s) is treated as Scalar broadcast.
            if all(bool(dace.symbolic.simplify(s - 1) == 0) for s in shape):
                return "Scalar"
        return "Tile"

    def _detect_reduction(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str]]:
        """If ``tasklet`` is an in-place RMW reduction body ``_acc = _acc <op> _val`` (or
        ``_acc = _val <op> _acc`` for commutative ops), return ``(acc_conn, val_conn, op)``.
        Otherwise ``None``.

        Detection criteria:

        * 2 in-connectors, 1 out-connector.
        * The output connector name matches one of the in-connector names (in-place RMW).
        * Body is exactly ``<out> = <out> <op> <other>`` (or the symmetric form) with op
          in :data:`_SUPPORTED_REDUCE_OPS` (associative subset).

        Per the user direction (2026-06-09): a tile -> scalar reduction is the natural
        store-with-reduction case; the in-body accumulator pattern is the entry point. The
        post-walker shape places the tile-shape input on the ``_val`` edge (from a bridge)
        and the scalar accumulator on the ``_acc`` edges. ``TileReduce`` lowers the
        accumulation across the full tile to a single scalar write at the boundary.
        """
        if len(tasklet.in_connectors) != 2 or len(tasklet.out_connectors) != 1:
            return None
        out_conn = next(iter(tasklet.out_connectors))
        if out_conn not in tasklet.in_connectors:
            return None
        # The "other" in-connector is the tile-shape input feeding the reduction.
        other_conns = [c for c in tasklet.in_connectors if c != out_conn]
        if len(other_conns) != 1:
            return None
        other_conn = other_conns[0]
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        for op in _SUPPORTED_REDUCE_OPS:
            if op in ("min", "max"):
                forms = (
                    f"{out_conn} = {op}({out_conn}, {other_conn})",
                    f"{out_conn} = ({op}({out_conn}, {other_conn}))",
                    f"{out_conn} = {op}({other_conn}, {out_conn})",
                    f"{out_conn} = ({op}({other_conn}, {out_conn}))",
                )
            else:
                forms = (
                    f"{out_conn} = {out_conn} {op} {other_conn}",
                    f"{out_conn} = ({out_conn} {op} {other_conn})",
                    f"{out_conn} = {other_conn} {op} {out_conn}",
                    f"{out_conn} = ({other_conn} {op} {out_conn})",
                )
            if body in forms:
                return out_conn, other_conn, op
        return None

    def _detect_ite(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str, str]]:
        """If ``tasklet`` is a ternary if-then-else body, return
        ``(out_conn, cond_conn, t_conn, e_conn)``. Otherwise ``None``.

        Matches the Python ternary form (DaCe parenthesises the RHS):
        ``_o = _t if _cond else _e`` -> ``_o = (_t if _cond else _e)``.

        Three in-connectors are required; their roles are inferred from the
        position in the expression (``_t``, ``_cond``, ``_e``).
        """
        if len(tasklet.in_connectors) != 3 or len(tasklet.out_connectors) != 1:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        in_conns = list(tasklet.in_connectors)
        # Try every (t, cond, e) ordering of the three connectors; accept both the parenthesised
        # and bare forms.
        from itertools import permutations
        for t, cond, e in permutations(in_conns, 3):
            for form in (f"{out_conn} = {t} if {cond} else {e}", f"{out_conn} = ({t} if {cond} else {e})"):
                if body == form:
                    return out_conn, cond, t, e
        return None

    def _detect_assign(self, tasklet: Tasklet) -> Optional[Tuple[str, str]]:
        """If ``tasklet`` body is exactly ``_o = _a`` (a trivial assign), return
        ``(out_conn, a_conn)``. Otherwise ``None``.
        """
        if len(tasklet.in_connectors) != 1 or len(tasklet.out_connectors) != 1:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        a_conn = next(iter(tasklet.in_connectors))
        if body in (f"{out_conn} = {a_conn}", f"{out_conn} = ({a_conn})"):
            return out_conn, a_conn
        return None

    def _convert_assign(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        """Replace a trivial ``_o = _a`` tasklet with a direct AN-to-AN edge.

        Both edges already have matching tile-shape descriptors (the pre-pass ensures
        this); the tasklet is doing nothing semantically. DaCe handles array-to-array
        memlet copies natively.

        The new edge's memlet references the SOURCE bridge (the read side); DaCe handles
        the destination via the edge endpoint. Memlets typed against the destination
        confuse the SDFG validator (which checks data-vs-endpoint consistency).
        """
        out_conn, a_conn = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if a_conn not in in_edges or not out_edges:
            return False
        a_edge = in_edges[a_conn]
        out_edge = out_edges[0]
        # Use the input-side memlet (which references the source bridge / AN) so the new
        # edge's memlet.data matches the new edge's source.
        inner_state.add_edge(a_edge.src, a_edge.src_conn, out_edge.dst, out_edge.dst_conn,
                             dace.Memlet.from_memlet(a_edge.data))
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _convert_one(self, inner_state: SDFGState, tasklet: Tasklet) -> bool:
        """Replace ``tasklet`` with a :class:`TileBinop` / :class:`TileUnop` /
        :class:`TileITE` / :class:`TileReduce` lib node if its body matches a
        recognised op shape.

        Dispatch order: unary (1 in) -> reduction (2 in, in-place RMW) -> binop
        (2 in, non-RMW) -> ITE (3 in). The reduction check runs BEFORE binop so
        an RMW accumulator pattern ``_acc = _acc + _val`` is never miscaptured
        as a generic ``TileBinop(+)``.

        :returns: ``True`` on rewrite.
        """
        # Assign first (trivial ``_o = _a``); then unop (single in-connector unary call);
        # then binop-with-symbol (single in-connector but body has a Symbol second operand);
        # then reduction (2 in conns, in-place RMW); then plain binop (2 in conns, non-RMW).
        assign = self._detect_assign(tasklet)
        if assign is not None:
            return self._convert_assign(inner_state, tasklet, assign)
        unop = self._detect_unop(tasklet)
        if unop is not None:
            return self._convert_unop(inner_state, tasklet, unop)
        symbol_binop = self._detect_binop_with_symbol(tasklet)
        if symbol_binop is not None:
            return self._convert_binop_with_symbol(inner_state, tasklet, symbol_binop)
        reduction = self._detect_reduction(tasklet)
        if reduction is not None:
            return self._convert_reduction(inner_state, tasklet, reduction)
        binop = self._detect_binop(tasklet)
        if binop is not None:
            return self._convert_binop(inner_state, tasklet, binop)
        ite = self._detect_ite(tasklet)
        if ite is not None:
            return self._convert_ite(inner_state, tasklet, ite)
        return False

    def _convert_reduction(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        acc_conn, val_conn, op = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = {e.src_conn: e for e in inner_state.out_edges(tasklet)}
        if acc_conn not in in_edges or val_conn not in in_edges or acc_conn not in out_edges:
            return False
        val_edge = in_edges[val_conn]
        out_edge = out_edges[acc_conn]
        mask_an = self._find_mask_an(inner_state)
        reduce_node = TileReduce(name=f"{tasklet.label}_reduce",
                                 widths=tuple(self.widths),
                                 op=op,
                                 has_mask=mask_an is not None)
        inner_state.add_node(reduce_node)
        self._wire_mask(inner_state, reduce_node, mask_an)
        # TileReduce connectors: _src (tile input) -> _dst (scalar accumulator). The acc-input
        # edge dangles: TileReduce reads no separate scalar accumulator -- it accumulates over
        # the full tile in one shot. If the original tasklet's acc_in edge was the initial
        # load of the accumulator from a parent state, the inner_sdfg.states() walk will leave
        # it intact on the source AN; the new _dst edge writes the reduction result on top.
        inner_state.add_edge(val_edge.src, val_edge.src_conn, reduce_node, "_src",
                             dace.Memlet.from_memlet(val_edge.data))
        inner_state.add_edge(reduce_node, "_dst", out_edge.dst, out_edge.dst_conn,
                             dace.Memlet.from_memlet(out_edge.data))
        for edge in list(in_edges.values()) + list(out_edges.values()):
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _convert_ite(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        out_conn, cond_conn, t_conn, e_conn = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if any(c not in in_edges for c in (cond_conn, t_conn, e_conn)) or not out_edges:
            return False
        out_edge = out_edges[0]
        # Output transient shape is pre-determined by InferBodyTransientShapes (forward
        # analysis pre-pass per design 6.2). No reactive widening here.
        mask_an = self._find_mask_an(inner_state)
        ite = TileITE(name=f"{tasklet.label}_ite", widths=tuple(self.widths), has_mask=mask_an is not None)
        inner_state.add_node(ite)
        self._wire_mask(inner_state, ite, mask_an)
        # TileITE connectors: _cond / _t / _e -> _o.
        for src_conn_name, dst_conn_name in (
            (cond_conn, "_cond"),
            (t_conn, "_t"),
            (e_conn, "_e"),
        ):
            edge = in_edges[src_conn_name]
            inner_state.add_edge(edge.src, edge.src_conn, ite, dst_conn_name, dace.Memlet.from_memlet(edge.data))
        inner_state.add_edge(ite, "_o", out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _convert_binop(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        out_conn, a_conn, b_conn, op = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if a_conn not in in_edges or b_conn not in in_edges or not out_edges:
            return False
        out_edge = out_edges[0]
        a_edge = in_edges[a_conn]
        b_edge = in_edges[b_conn]
        # Operand-kind classification from the source's descriptor: Scalar / length-1 Array
        # source = broadcast Scalar operand kind on the lib node (design section 6.5).
        kind_a = self._operand_kind(inner_state, a_edge)
        kind_b = self._operand_kind(inner_state, b_edge)
        # Output transient shape is pre-determined by InferBodyTransientShapes (forward
        # analysis pre-pass per design 6.2). The output kind on the lib node is implied by
        # the descriptor on ``out_edge``'s destination; validate() enforces consistency.
        mask_an = self._find_mask_an(inner_state)
        binop = TileBinop(name=f"{tasklet.label}_binop",
                          widths=tuple(self.widths),
                          op=op,
                          kind_a=kind_a,
                          kind_b=kind_b,
                          has_mask=mask_an is not None)
        inner_state.add_node(binop)
        self._wire_mask(inner_state, binop, mask_an)
        inner_state.add_edge(a_edge.src, a_edge.src_conn, binop, "_a", dace.Memlet.from_memlet(a_edge.data))
        inner_state.add_edge(b_edge.src, b_edge.src_conn, binop, "_b", dace.Memlet.from_memlet(b_edge.data))
        inner_state.add_edge(binop, "_c", out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _convert_binop_with_symbol(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        """Emit a TileBinop whose second operand is an embedded Symbol expression.

        Per design 6.2 the Symbol operand kind has NO connector -- the expression is
        embedded in the lib node body via ``expr_a`` / ``expr_b``. Pass ``kind_b=Symbol``
        + ``expr_b=<expr>`` (or kind_a / expr_a depending on symbol_side).
        """
        out_conn, a_conn, op, symbol_side, symbol_expr = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if a_conn not in in_edges or not out_edges:
            return False
        out_edge = out_edges[0]
        a_edge = in_edges[a_conn]
        kind_tile_side = self._operand_kind(inner_state, a_edge)
        mask_an = self._find_mask_an(inner_state)
        if symbol_side == "b":
            binop = TileBinop(name=f"{tasklet.label}_binop_sym",
                              widths=tuple(self.widths),
                              op=op,
                              kind_a=kind_tile_side,
                              kind_b="Symbol",
                              expr_b=symbol_expr,
                              has_mask=mask_an is not None)
            inner_state.add_node(binop)
            inner_state.add_edge(a_edge.src, a_edge.src_conn, binop, "_a", dace.Memlet.from_memlet(a_edge.data))
        else:
            binop = TileBinop(name=f"{tasklet.label}_binop_sym",
                              widths=tuple(self.widths),
                              op=op,
                              kind_a="Symbol",
                              kind_b=kind_tile_side,
                              expr_a=symbol_expr,
                              has_mask=mask_an is not None)
            inner_state.add_node(binop)
            inner_state.add_edge(a_edge.src, a_edge.src_conn, binop, "_b", dace.Memlet.from_memlet(a_edge.data))
        self._wire_mask(inner_state, binop, mask_an)
        inner_state.add_edge(binop, "_c", out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _convert_unop(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        out_conn, a_conn, op = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if a_conn not in in_edges or not out_edges:
            return False
        out_edge = out_edges[0]
        a_edge = in_edges[a_conn]
        kind_a = self._operand_kind(inner_state, a_edge)
        # Output transient shape is pre-determined by InferBodyTransientShapes.
        mask_an = self._find_mask_an(inner_state)
        unop = TileUnop(name=f"{tasklet.label}_unop",
                        widths=tuple(self.widths),
                        op=op,
                        kind_a=kind_a,
                        has_mask=mask_an is not None)
        inner_state.add_node(unop)
        self._wire_mask(inner_state, unop, mask_an)
        inner_state.add_edge(a_edge.src, a_edge.src_conn, unop, "_a", dace.Memlet.from_memlet(a_edge.data))
        inner_state.add_edge(unop, "_c", out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _convert_inner(self, inner_sdfg: SDFG) -> int:
        """Walk every state of ``inner_sdfg`` and convert recognised binop tasklets.

        :returns: Number of conversions performed.
        """
        converted = 0
        for inner_state in inner_sdfg.states():
            for node in list(inner_state.nodes()):
                if not isinstance(node, Tasklet):
                    continue
                if self._convert_one(inner_state, node):
                    converted += 1
        return converted

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Walk every tile-tagged body NSDFG; convert recognised tasklets to tile lib nodes.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Pipeline results (unused).
        :returns: Number of tasklets converted, or ``None`` if zero.
        """
        total = 0
        for _state, nsdfg_node, _map_entry in self._body_nsdfgs(sdfg):
            total += self._convert_inner(nsdfg_node.sdfg)
        return total or None
