# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Convert in-body tasklets to ``TileBinop`` / ``TileUnop`` / ``TileITE``.

After :class:`InsertTileLoadStore` walks every tile-tagged body NSDFG and
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
from dace.transformation.passes.vectorization.widen_accesses import materialise_per_lane_index_tile
from dace.sdfg import SDFG
from dace.sdfg.nodes import MapEntry, NestedSDFG, Tasklet
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map

#: Subset of binary operators that map directly onto :class:`TileBinop`. Includes
#: comparison (``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``) which produce bool tile
#: outputs used as the cond input of :class:`TileITE` (per design 7.5 cond-mask
#: broadcasting), and ``**`` (Python power; lowers to ``std::pow``). ``PowerOperatorExpansion``
#: upstream rewrites integer-constant exponents (``x**2`` -> ``x*x``); only true runtime
#: exponents reach this dispatch.
_SUPPORTED_BINOPS = {
    "+", "-", "*", "/", "%", "**", "min", "max", "<", "<=", ">", ">=", "==", "!=", "&&", "||", "&", "|", "^"
}

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

    :ivar widths: Per-tile-dim widths; mirrors :class:`InsertTileLoadStore`.
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

    def _is_lane_id_dependent(self, expr: str, iter_vars: Tuple[str, ...]) -> bool:
        """True if ``expr`` references any tile iter_var (lane-id-dependent Symbol).

        Symbol operands that depend on the iter_vars cannot be broadcast at expansion
        time -- they differ per lane and must be materialised as tile-shape transients
        (one element per lane). The non-dependent ("data-independent") symbols are
        loop-invariant from the inner-tile perspective and can be embedded inline as
        ``expr_a`` / ``expr_b`` on the lib node.

        :param expr: The C-like / Python-like expression string (e.g. ``"N + 1"``,
            ``"ii"``, ``"2 * ii + jj"``).
        :param iter_vars: Tile dim iter_var names (e.g. ``("ii", "jj")``).
        """
        try:
            tokens = set(dace.symbolic.SymExpr(expr).free_symbols)
        except Exception:  # noqa: BLE001
            tokens = set()
        for s in tokens:
            if str(s) in iter_vars:
                return True
        return False

    def _resolve_symbol_operand(self, inner_state: SDFGState, expr: str,
                                iter_vars: Tuple[str, ...]) -> Tuple[str, Optional[str], Optional[str]]:
        """Resolve a Symbol-shaped operand into either:

        * an invariant ``Symbol`` -- ``(kind="Symbol", expr=<expr>, an_name=None)``,
        * a lane-id-dependent ``Tile`` -- ``(kind="Tile", expr=None, an_name=<tile_name>)``.

        For the lane-id-dependent case, materialises a per-lane tile via
        :meth:`_materialise_lane_id_tile`. The materialised tile's element type
        is ``int64``; arithmetic on it composes with the Tile operand contract at
        expansion time.

        Key distinction vs the gather-index materialiser
        (``materialise_per_lane_index_tile``): for lane-id Symbols, the iter_var
        ``ii`` represents the OUTER tile start (post-stride), and the per-lane
        value is ``ii + __l``. The gather-index materialiser SUBSTITUTES
        ``ii -> __l`` (correct for gather lookups where ``ii`` is the lane offset),
        which would lose the outer tile's start value.
        """
        if not iter_vars or not self._is_lane_id_dependent(expr, iter_vars):
            return "Symbol", expr, None
        an_name = self._materialise_lane_id_tile(inner_state, expr, iter_vars)
        return "Tile", None, an_name

    def _materialise_lane_id_tile(self, inner_state: SDFGState, expr: str, iter_vars: Tuple[str, ...]) -> str:
        """Mint a per-lane int64 tile containing the evaluation of ``expr`` at
        ``(iter_var_k -> iter_var_k + __l_k)`` for each tile dim ``k``.

        For K=1 with iter_var ``ii`` and expr ``"ii"``: produces
        ``_lane_tile[l] = (ii) + l``.

        For K=2 with iter_vars ``(ii, jj)`` and expr ``"2 * ii + jj"``: produces
        ``_lane_tile[l_0, l_1] = 2 * (ii + l_0) + (jj + l_1)``.
        """
        import dace.dtypes as _dtypes
        from dace.memlet import Memlet as _Memlet
        sdfg = inner_state.sdfg
        widths = tuple(int(w) for w in self.widths)
        K = len(widths)
        # Rewrite each iter_var ``v`` in ``expr`` to ``(v + __l<k>)``.
        body_expr = expr
        for k, v in enumerate(iter_vars):
            # Word-boundary aware replacement to avoid clobbering substring matches.
            import re
            body_expr = re.sub(rf"\b{re.escape(v)}\b", f"({v} + __l{k})", body_expr)
        # Pick a unique transient name for the materialised tile.
        arr_name, _ = sdfg.add_array(
            "_sym_tile",
            shape=widths,
            dtype=dace.int64,
            transient=True,
            storage=_dtypes.StorageType.Register,
            find_new_name=True,
        )
        # Compute the row-major flat offset string.
        parts = []
        for i in range(K):
            inner = 1
            for q in range(i + 1, K):
                inner *= widths[q]
            parts.append(f"__l{i}" if inner == 1 else f"(__l{i} * {inner})")
        flat = " + ".join(parts) if parts else "0"
        code_lines = []
        for d in range(K):
            code_lines.append(f"{'    ' * d}for (std::size_t __l{d} = 0; __l{d} < {widths[d]}; ++__l{d}) {{")
        code_lines.append(f"{'    ' * K}_out[{flat}] = (int64_t)({body_expr});")
        for d in reversed(range(K)):
            code_lines.append(f"{'    ' * d}}}")
        tasklet = inner_state.add_tasklet(
            name=f"lane_id_mat_{arr_name}",
            inputs=set(),
            outputs={"_out"},
            code="\n".join(code_lines),
            language=_dtypes.Language.CPP,
        )
        out_an = inner_state.add_access(arr_name)
        out_subset = ", ".join(f"0:{w}" for w in widths)
        inner_state.add_edge(tasklet, "_out", out_an, None, _Memlet(f"{arr_name}[{out_subset}]"))
        return arr_name

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

        Mirror of the walker shape used by :class:`InsertTileLoadStore` and
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

        Accepts the 1-in-connector ``_out = _a <op> _a`` shape (same connector both
        sides) -- ``PowerOperatorExpansion`` emits this for ``x**2``.
        """
        if len(tasklet.out_connectors) != 1:
            return None
        if len(tasklet.in_connectors) == 1:
            # Same-connector-twice shape: ``_out = _a <op> _a``.
            body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
            body = body.strip().rstrip(";").strip()
            out_conn = next(iter(tasklet.out_connectors))
            a_conn = next(iter(tasklet.in_connectors))
            for op in _SUPPORTED_BINOPS:
                if op in ("min", "max"):
                    continue
                for form in (f"{out_conn} = {a_conn} {op} {a_conn}", f"{out_conn} = ({a_conn} {op} {a_conn})"):
                    if body == form:
                        return out_conn, a_conn, a_conn, op
            return None
        if len(tasklet.in_connectors) != 2:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        in_conns = list(tasklet.in_connectors)
        # Try every op in _SUPPORTED_BINOPS against the two operand orderings. DaCe wraps the
        # RHS of an assignment tasklet in parentheses (``_o = (_a + _b)``); accept both forms.
        # The function-form ops (``min``, ``max``, ``pow``) accept the bare-name function syntax
        # as well; ``pow(a, b)`` is the Python equivalent of ``a ** b``.
        for op in _SUPPORTED_BINOPS:
            for a, b in (in_conns, list(reversed(in_conns))):
                if op in ("min", "max"):
                    forms = (f"{out_conn} = {op}({a}, {b})", f"{out_conn} = ({op}({a}, {b}))")
                elif op == "**":
                    forms = (f"{out_conn} = {a} ** {b}", f"{out_conn} = ({a} ** {b})", f"{out_conn} = pow({a}, {b})",
                             f"{out_conn} = (pow({a}, {b}))")
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

    def _detect_const_assign(self, tasklet: Tasklet) -> Optional[Tuple[str, str]]:
        """If ``tasklet`` is a 0-in-connector pure constant / Symbol assignment
        (``_o = <expr>`` with no operator -- just a literal or a Symbol), return
        ``(out_conn, expr)``. Otherwise ``None``.

        This is the simplest output-side Symbol case: memset / broadcast / set-to-zero
        kernels. The lib node side is a TileUnop with kind_a=Symbol + op="copy" -- but
        since TileUnop has no "copy" op, we emit a constant-fill tasklet via
        ``_materialise_invariant_to_tile`` and feed it as the output.
        """
        if len(tasklet.in_connectors) != 0 or len(tasklet.out_connectors) != 1:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        if not body.startswith(f"{out_conn} = "):
            return None
        rhs = body[len(f"{out_conn} = "):].strip()
        if rhs.startswith("(") and rhs.endswith(")"):
            rhs = rhs[1:-1].strip()
        if not rhs:
            return None
        # Reject anything that looks like an op shape -- those are handled by the other
        # detectors. Heuristic: a constant has no binary operator at top level and no
        # function-call shape with comma-separated args.
        for op_token in (" + ", " - ", " * ", " / ", " % ", " ** ", " < ", " > ", " == ", " != ", " <= ", " >= ", " and ",
                         " or "):
            if op_token in rhs:
                return None
        # Reject function-form binops / ITE.
        for fn in ("min(", "max(", "pow(", "ITE("):
            if rhs.startswith(fn):
                return None
        # Reject leading minus (handled by _detect_unop_with_symbol negation case).
        if rhs.startswith("-") and not rhs.startswith("-"):  # never triggers; placeholder
            return None
        return out_conn, rhs

    def _convert_const_assign(self, inner_state: SDFGState, tasklet: Tasklet, detected,
                              iter_vars: Tuple[str, ...]) -> bool:
        """Replace ``_o = <const_expr>`` with a constant-fill tasklet that writes the
        full-tile output transient. Functionally equivalent to ``TileBroadcastSymbol``
        but emitted as a raw CPP tasklet (the broadcast lib node lands in a follow-up
        slice; this gets memset / broadcast kernels working today)."""
        out_conn, expr = detected
        out_edges = list(inner_state.out_edges(tasklet))
        if not out_edges:
            return False
        out_edge = out_edges[0]
        an_name = self._materialise_invariant_to_tile(inner_state, expr, out_edge)
        from dace.sdfg.nodes import AccessNode
        existing = next((n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == an_name), None)
        src_an = existing if existing is not None else inner_state.add_access(an_name)
        # Wire the materialised tile straight into the output via a direct copy edge.
        # The materialised tile shape matches the output bridge shape (full tile).
        subset = ", ".join(f"0:{w}" for w in self.widths)
        inner_state.add_edge(src_an, None, out_edge.dst, out_edge.dst_conn, dace.Memlet(f"{an_name}[{subset}]"))
        inner_state.remove_edge(out_edge)
        inner_state.remove_node(tasklet)
        return True

    def _detect_unop_with_symbol(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str]]:
        """If ``tasklet`` is a 0-in-connector unary symbol body (``_o = <op>(<expr>)`` or
        ``_o = -<expr>``), return ``(out_conn, op_label, expr)``. Otherwise ``None``.

        The lib node downstream gets ``kind_a=Symbol`` + ``expr_a=<expr>`` with NO ``_a``
        connector (design 6.2: Symbol operands are embedded inline, not materialised).
        """
        if len(tasklet.in_connectors) != 0 or len(tasklet.out_connectors) != 1:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        rhs = body[len(f"{out_conn} = "):]
        if rhs.startswith("(") and rhs.endswith(")"):
            rhs = rhs[1:-1].strip()
        # Negation: `_o = -<expr>` / `_o = (- <expr>)`.
        if rhs.startswith("-"):
            expr = rhs[1:].strip()
            if expr.startswith("(") and expr.endswith(")"):
                expr = expr[1:-1].strip()
            if expr:
                return out_conn, "neg", expr
        # Function-form: `_o = <op>(<expr>)` (DaCe wraps argument in extra parens too).
        for op in _SUPPORTED_UNOPS:
            if op == "neg":
                continue
            for pref in (op, f"math.{op}", f"std::{op}"):
                if rhs.startswith(f"{pref}(") and rhs.endswith(")"):
                    expr = rhs[len(pref) + 1:-1].strip()
                    if expr.startswith("(") and expr.endswith(")"):
                        expr = expr[1:-1].strip()
                    if expr:
                        return out_conn, op, expr
        return None

    def _detect_binop_with_two_symbols(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str, str]]:
        """If ``tasklet`` is a 0-in-connector binary symbol body (``_o = <expr_a> <op> <expr_b>``
        or ``_o = <op>(<expr_a>, <expr_b>)``), return ``(out_conn, op, expr_a, expr_b)``.
        Otherwise ``None``.

        Both operands are inline Symbol expressions; the lib node downstream is constructed
        with ``kind_a=Symbol``, ``kind_b=Symbol``, ``expr_a / expr_b`` set. The output is
        Scalar (per design 6.2: all-Symbol -> output may be Scalar).
        """
        if len(tasklet.in_connectors) != 0 or len(tasklet.out_connectors) != 1:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        rhs = body[len(f"{out_conn} = "):]
        if rhs.startswith("(") and rhs.endswith(")"):
            rhs = rhs[1:-1].strip()
        for op in _SUPPORTED_BINOPS:
            if op in ("min", "max"):
                prefix = f"{op}("
                if rhs.startswith(prefix) and rhs.endswith(")"):
                    inner = rhs[len(prefix):-1]
                    # Split on top-level comma (the simple case -- both expressions are
                    # parenthesised / identifier-shaped without nested commas).
                    depth = 0
                    split_idx = -1
                    for i, ch in enumerate(inner):
                        if ch == "(":
                            depth += 1
                        elif ch == ")":
                            depth -= 1
                        elif ch == "," and depth == 0:
                            split_idx = i
                            break
                    if split_idx >= 0:
                        expr_a = inner[:split_idx].strip()
                        expr_b = inner[split_idx + 1:].strip()
                        if expr_a and expr_b:
                            return out_conn, op, expr_a, expr_b
            else:
                sep = f" {op} "
                if sep not in rhs:
                    continue
                # Top-level split on the op (avoid nested expressions).
                depth = 0
                split_idx = -1
                token_len = len(sep)
                for i in range(len(rhs) - token_len + 1):
                    if rhs[i] == "(":
                        depth += 1
                    elif rhs[i] == ")":
                        depth -= 1
                    elif depth == 0 and rhs[i:i + token_len] == sep:
                        split_idx = i
                        break
                if split_idx >= 0:
                    expr_a = rhs[:split_idx].strip()
                    expr_b = rhs[split_idx + token_len:].strip()
                    if expr_a and expr_b:
                        return out_conn, op, expr_a, expr_b
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

    def _detect_ite(self, tasklet: Tasklet) -> Optional[Tuple[str, ...]]:
        """If ``tasklet`` is a ternary if-then-else body, return
        ``(out_conn, cond_conn_or_expr, t_conn_or_expr, e_conn_or_expr, has_t_sym, has_e_sym)``.

        Otherwise ``None``.

        Recognised forms:

        * Python ternary -- 3 in-conn: ``_o = _t if _cond else _e``
          (possibly parenthesised).
        * SplitTasklets canonical: ``_o = ITE(_cond, _t, _e)``
          (3 in-conn).
        * SplitTasklets with literal/Symbol arms -- 2 in-conn:
          ``_o = ITE(_cond, _t, <expr>)`` (e_arm is a literal/Symbol),
          ``_o = ITE(_cond, <expr>, _e)`` (t_arm is a literal/Symbol).
        """
        n_in = len(tasklet.in_connectors)
        if n_in not in (2, 3) or len(tasklet.out_connectors) != 1:
            return None
        body = (tasklet.code.as_string if hasattr(tasklet.code, "as_string") else str(tasklet.code))
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        in_conns = list(tasklet.in_connectors)
        # Python ternary form -- 3 in-conn only.
        if n_in == 3:
            from itertools import permutations
            for t, cond, e in permutations(in_conns, 3):
                for form in (f"{out_conn} = {t} if {cond} else {e}", f"{out_conn} = ({t} if {cond} else {e})"):
                    if body == form:
                        return out_conn, cond, t, e, False, False
        # ITE(cond, t, e) function form: handles both 3-in-conn and the 2-in-conn-with-symbol
        # cases that SplitTasklets emits.
        rhs = body[len(f"{out_conn} = "):].strip()
        if rhs.startswith("(") and rhs.endswith(")"):
            rhs = rhs[1:-1].strip()
        prefix = "ITE("
        if rhs.startswith(prefix) and rhs.endswith(")"):
            inner = rhs[len(prefix):-1]
            # Split on top-level commas (3 parts).
            parts = self._split_top_level_commas(inner, 3)
            if parts is not None and len(parts) == 3:
                cond_arg, t_arg, e_arg = (p.strip() for p in parts)
                # Each arg is either an in-connector (Tile/Scalar) or a literal/Symbol expression.
                is_cond_conn = cond_arg in in_conns
                is_t_conn = t_arg in in_conns
                is_e_conn = e_arg in in_conns
                # cond is always a connector (it's the comparison result), but t / e may be Symbol.
                if is_cond_conn:
                    if is_t_conn and is_e_conn:
                        return out_conn, cond_arg, t_arg, e_arg, False, False
                    if is_t_conn and not is_e_conn:
                        # e is Symbol/literal.
                        return out_conn, cond_arg, t_arg, e_arg, False, True
                    if not is_t_conn and is_e_conn:
                        # t is Symbol/literal.
                        return out_conn, cond_arg, t_arg, e_arg, True, False
        return None

    def _split_top_level_commas(self, s: str, expected_parts: int) -> Optional[list]:
        """Split ``s`` on top-level commas (skipping commas inside parentheses).
        Returns the parts when their count matches ``expected_parts``; else ``None``."""
        parts = []
        depth = 0
        current = []
        for ch in s:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts if len(parts) == expected_parts else None

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

    def _convert_one(self, inner_state: SDFGState, tasklet: Tasklet, iter_vars: Tuple[str, ...]) -> bool:
        """Replace ``tasklet`` with a Tile lib node if its body matches a recognised
        op shape. Dispatch (in order, by in-connector count): 0-in (Symbol-only) ->
        1-in (assign / unop / binop-with-symbol) -> 2-in (reduction / binop) ->
        3-in (ITE).

        :param iter_vars: Tile dim iter_var names; used to decide whether a Symbol
            operand is loop-invariant (broadcast) or lane-id-dependent (materialise
            to a per-lane tile).

        :returns: ``True`` on rewrite.
        """
        # 0 in-conn tasklets first (purely Symbol-driven). Order: const-assign, then
        # 0-in unary (with a function call shape), then 0-in binop (two operands).
        unop_sym = self._detect_unop_with_symbol(tasklet)
        if unop_sym is not None:
            return self._convert_unop_with_symbol(inner_state, tasklet, unop_sym, iter_vars)
        binop_two_sym = self._detect_binop_with_two_symbols(tasklet)
        if binop_two_sym is not None:
            return self._convert_binop_with_two_symbols(inner_state, tasklet, binop_two_sym, iter_vars)
        const_assign = self._detect_const_assign(tasklet)
        if const_assign is not None:
            return self._convert_const_assign(inner_state, tasklet, const_assign, iter_vars)
        # 1 in-conn tasklets.
        assign = self._detect_assign(tasklet)
        if assign is not None:
            return self._convert_assign(inner_state, tasklet, assign)
        unop = self._detect_unop(tasklet)
        if unop is not None:
            return self._convert_unop(inner_state, tasklet, unop)
        symbol_binop = self._detect_binop_with_symbol(tasklet)
        if symbol_binop is not None:
            return self._convert_binop_with_symbol(inner_state, tasklet, symbol_binop, iter_vars)
        # ITE shape (``if .. else`` or ``ITE(...)`` function form) -- detect FIRST so the
        # 2-in-conn ``ITE(_cond, _t, <symbol>)`` shape isn't miscaptured as a binop or
        # reduction. The ITE detector requires a precise body match so binop fall-through
        # remains safe.
        ite = self._detect_ite(tasklet)
        if ite is not None:
            return self._convert_ite(inner_state, tasklet, ite, iter_vars)
        # 2 in-conn tasklets (reduction before plain binop so accumulators aren't miscaptured).
        reduction = self._detect_reduction(tasklet)
        if reduction is not None:
            return self._convert_reduction(inner_state, tasklet, reduction)
        binop = self._detect_binop(tasklet)
        if binop is not None:
            return self._convert_binop(inner_state, tasklet, binop)
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

    def _convert_ite(self, inner_state: SDFGState, tasklet: Tasklet, detected, iter_vars: Tuple[str, ...]) -> bool:
        """Convert a ternary tasklet to a TileITE lib node.

        ``detected`` shape: ``(out_conn, cond_arg, t_arg, e_arg, t_is_sym, e_is_sym)``.
        When ``t_is_sym`` or ``e_is_sym`` is True the corresponding arm is a Symbol or
        literal expression (not a tasklet in-connector). For those we materialise a
        full-tile transient via :meth:`_materialise_symbol_to_tile` and wire it to the
        TileITE connector -- the lib node sees a uniform Tile operand contract.

        This implements the "transients-are-either-full-tile-or-scalar" invariant of
        design 7.5: a Symbol arm becomes a full-tile transient (broadcast values or
        per-lane materialised), so TileITE doesn't need a Symbol kind.
        """
        out_conn, cond_arg, t_arg, e_arg, t_is_sym, e_is_sym = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if not out_edges or cond_arg not in in_edges:
            return False
        if not t_is_sym and t_arg not in in_edges:
            return False
        if not e_is_sym and e_arg not in in_edges:
            return False
        out_edge = out_edges[0]
        mask_an = self._find_mask_an(inner_state)
        ite = TileITE(name=f"{tasklet.label}_ite", widths=tuple(self.widths), has_mask=mask_an is not None)
        inner_state.add_node(ite)
        self._wire_mask(inner_state, ite, mask_an)
        # cond is always a connector (the result of the comparison upstream). If it's
        # a Scalar transient (from an all-Symbol comparison like ``FLAG > 0``), broadcast
        # to full-tile shape per design 7.5 (cond-mask broadcast). TileITE's pure
        # expansion expects a full-tile bool tile.
        cond_edge = in_edges[cond_arg]
        if self._is_scalar_or_len1_source(inner_state, cond_edge):
            broadcast_name = self._broadcast_scalar_to_tile(inner_state, cond_edge, dtype=dace.bool_)
            from dace.sdfg.nodes import AccessNode
            existing = next((n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == broadcast_name),
                            None)
            cond_an = existing if existing is not None else inner_state.add_access(broadcast_name)
            subset = ", ".join(f"0:{w}" for w in self.widths)
            inner_state.add_edge(cond_an, None, ite, "_cond", dace.Memlet(f"{broadcast_name}[{subset}]"))
        else:
            inner_state.add_edge(cond_edge.src, cond_edge.src_conn, ite, "_cond",
                                 dace.Memlet.from_memlet(cond_edge.data))
        # t arm: connector OR Symbol-materialised tile.
        if t_is_sym:
            t_an = self._materialise_symbol_to_tile(inner_state, t_arg, iter_vars, out_edge)
            subset = ", ".join(f"0:{w}" for w in self.widths)
            inner_state.add_edge(t_an, None, ite, "_t", dace.Memlet(f"{t_an.data}[{subset}]"))
        else:
            t_edge = in_edges[t_arg]
            inner_state.add_edge(t_edge.src, t_edge.src_conn, ite, "_t", dace.Memlet.from_memlet(t_edge.data))
        # e arm: connector OR Symbol-materialised tile.
        if e_is_sym:
            e_an = self._materialise_symbol_to_tile(inner_state, e_arg, iter_vars, out_edge)
            subset = ", ".join(f"0:{w}" for w in self.widths)
            inner_state.add_edge(e_an, None, ite, "_e", dace.Memlet(f"{e_an.data}[{subset}]"))
        else:
            e_edge = in_edges[e_arg]
            inner_state.add_edge(e_edge.src, e_edge.src_conn, ite, "_e", dace.Memlet.from_memlet(e_edge.data))
        inner_state.add_edge(ite, "_o", out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _is_scalar_or_len1_source(self, inner_state: SDFGState, edge) -> bool:
        """Return True when ``edge.src`` reads a Scalar or length-1 Array (the "scalar"
        side of the transients-are-full-tile-or-scalar invariant)."""
        from dace.sdfg.nodes import AccessNode
        import dace.data as dd
        if not isinstance(edge.src, AccessNode):
            return False
        desc = inner_state.sdfg.arrays.get(edge.src.data)
        if desc is None:
            return False
        if isinstance(desc, dd.Scalar):
            return True
        if isinstance(desc, dd.Array):
            try:
                return all(bool(dace.symbolic.simplify(s - 1) == 0) for s in desc.shape)
            except Exception:  # noqa: BLE001
                return False
        return False

    def _broadcast_scalar_to_tile(self, inner_state: SDFGState, src_edge, dtype) -> str:
        """Mint a FULL-TILE transient with element type ``dtype`` and emit a tasklet that
        reads the Scalar / length-1 source via ``src_edge`` and writes the value to every
        lane.

        Reuses the producer's AccessNode as the input (so the SDFG scheduler orders the
        comparison tasklet before the broadcast tasklet).
        """
        import dace.dtypes as _dtypes
        from dace.memlet import Memlet as _Memlet
        sdfg = inner_state.sdfg
        widths = tuple(int(w) for w in self.widths)
        arr_name, _ = sdfg.add_array("_cond_bcast",
                                     shape=widths,
                                     dtype=dtype,
                                     transient=True,
                                     storage=_dtypes.StorageType.Register,
                                     find_new_name=True)
        K = len(widths)
        parts = []
        for i in range(K):
            inner = 1
            for q in range(i + 1, K):
                inner *= widths[q]
            parts.append(f"__l{i}" if inner == 1 else f"(__l{i} * {inner})")
        flat = " + ".join(parts) if parts else "0"
        code_lines = []
        for d in range(K):
            code_lines.append(f"{'    ' * d}for (std::size_t __l{d} = 0; __l{d} < {widths[d]}; ++__l{d}) {{")
        code_lines.append(f"{'    ' * K}_out[{flat}] = ({dtype.ctype})(_in);")
        for d in reversed(range(K)):
            code_lines.append(f"{'    ' * d}}}")
        tasklet = inner_state.add_tasklet(
            name=f"bcast_to_tile_{arr_name}",
            inputs={"_in"},
            outputs={"_out"},
            code="\n".join(code_lines),
            language=_dtypes.Language.CPP,
        )
        # Wire from the source AN (reuse, not a fresh access) and to a fresh broadcast AN.
        inner_state.add_edge(src_edge.src, src_edge.src_conn, tasklet, "_in", dace.Memlet.from_memlet(src_edge.data))
        out_an = inner_state.add_access(arr_name)
        out_subset = ", ".join(f"0:{w}" for w in widths)
        inner_state.add_edge(tasklet, "_out", out_an, None, _Memlet(f"{arr_name}[{out_subset}]"))
        return arr_name

    def _materialise_symbol_to_tile(self, inner_state: SDFGState, expr: str, iter_vars: Tuple[str, ...], out_edge):
        """Materialise a Symbol / literal expression as a FULL-TILE transient.

        Two sub-cases (per design 7.5):

        * **Lane-id-dependent** (expr references an iter_var) -- delegate to
          :meth:`_materialise_lane_id_tile` (int64 tile).
        * **Loop-invariant** (no iter_var refs) -- mint a transient with the OUTPUT
          edge's element dtype and emit a constant-fill tasklet that broadcasts
          ``expr`` across every lane.
        """
        if self._is_lane_id_dependent(expr, iter_vars):
            an_name = self._materialise_lane_id_tile(inner_state, expr, iter_vars)
        else:
            an_name = self._materialise_invariant_to_tile(inner_state, expr, out_edge)
        from dace.sdfg.nodes import AccessNode
        existing = next((n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == an_name), None)
        return existing if existing is not None else inner_state.add_access(an_name)

    def _materialise_invariant_to_tile(self, inner_state: SDFGState, expr: str, out_edge) -> str:
        """Mint a per-lane FULL-TILE transient containing ``expr`` broadcast across
        every lane. The dtype matches the OUTPUT edge's element dtype (so the
        TileITE's _t / _e operand dtype matches _o).
        """
        import dace.dtypes as _dtypes
        from dace.memlet import Memlet as _Memlet
        sdfg = inner_state.sdfg
        widths = tuple(int(w) for w in self.widths)
        # Pick element dtype from the OUTPUT edge's array (the ITE's output dtype).
        out_desc = sdfg.arrays.get(out_edge.data.data)
        dtype = out_desc.dtype if out_desc is not None else dace.float64
        arr_name, _ = sdfg.add_array("_ite_sym_tile",
                                     shape=widths,
                                     dtype=dtype,
                                     transient=True,
                                     storage=_dtypes.StorageType.Register,
                                     find_new_name=True)
        K = len(widths)
        parts = []
        for i in range(K):
            inner = 1
            for q in range(i + 1, K):
                inner *= widths[q]
            parts.append(f"__l{i}" if inner == 1 else f"(__l{i} * {inner})")
        flat = " + ".join(parts) if parts else "0"
        code_lines = []
        for d in range(K):
            code_lines.append(f"{'    ' * d}for (std::size_t __l{d} = 0; __l{d} < {widths[d]}; ++__l{d}) {{")
        code_lines.append(f"{'    ' * K}_out[{flat}] = ({dtype.ctype})({expr});")
        for d in reversed(range(K)):
            code_lines.append(f"{'    ' * d}}}")
        tasklet = inner_state.add_tasklet(
            name=f"sym_broadcast_{arr_name}",
            inputs=set(),
            outputs={"_out"},
            code="\n".join(code_lines),
            language=_dtypes.Language.CPP,
        )
        out_an = inner_state.add_access(arr_name)
        out_subset = ", ".join(f"0:{w}" for w in widths)
        inner_state.add_edge(tasklet, "_out", out_an, None, _Memlet(f"{arr_name}[{out_subset}]"))
        return arr_name
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
        # Per user direction 2026-06-10: mixed-dtype operands are NOT supported. The
        # walker-primary pipeline locks operand dtypes to a single type per lib node
        # (the lib node's tile transient + bridge + downstream copy chain all assume
        # uniform dtype). Refuse and raise NotImplementedError so callers know to
        # rewrite the kernel with explicit casts.
        sdfg = inner_state.sdfg
        a_dtype = sdfg.arrays[a_edge.data.data].dtype if a_edge.data and a_edge.data.data else None
        b_dtype = sdfg.arrays[b_edge.data.data].dtype if b_edge.data and b_edge.data.data else None
        c_dtype = sdfg.arrays[out_edge.data.data].dtype if out_edge.data and out_edge.data.data else None
        operand_dtypes = {d for d in (a_dtype, b_dtype, c_dtype) if d is not None}
        if len(operand_dtypes) > 1:
            raise NotImplementedError(
                f"vec(K-dim): mixed-dtype binop NOT supported. Tasklet {tasklet.label!r} body "
                f"{tasklet.code.as_string!r} mixes dtypes {operand_dtypes}. Per design 6.2 + user "
                f"direction the walker-primary path locks a single dtype per lib node. Rewrite the "
                f"kernel with an explicit cast tasklet upstream OR widen the destination dtype.")
        # Operand-kind classification from the source's descriptor: Scalar / length-1 Array
        # source = broadcast Scalar operand kind on the lib node (design section 6.5).
        kind_a = self._operand_kind(inner_state, a_edge)
        kind_b = self._operand_kind(inner_state, b_edge)
        # Output transient shape is pre-determined by WidenAccesses (forward
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

    def _convert_binop_with_symbol(self, inner_state: SDFGState, tasklet: Tasklet, detected,
                                   iter_vars: Tuple[str, ...]) -> bool:
        """Emit a TileBinop whose second operand is a Symbol expression.

        The Symbol expression may be:

        * **Loop-invariant** (no iter_var refs) -> ``kind=Symbol`` + ``expr_*``
          (no connector); broadcast at expansion time.
        * **Lane-id-dependent** (references an iter_var) -> materialise a per-lane
          tile via :func:`materialise_per_lane_index_tile` and wire as
          ``kind=Tile`` operand.
        """
        out_conn, a_conn, op, symbol_side, symbol_expr = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if a_conn not in in_edges or not out_edges:
            return False
        out_edge = out_edges[0]
        a_edge = in_edges[a_conn]
        kind_tile_side = self._operand_kind(inner_state, a_edge)
        sym_kind, sym_expr, sym_an_name = self._resolve_symbol_operand(inner_state, symbol_expr, iter_vars)
        mask_an = self._find_mask_an(inner_state)
        if symbol_side == "b":
            binop = TileBinop(name=f"{tasklet.label}_binop_sym",
                              widths=tuple(self.widths),
                              op=op,
                              kind_a=kind_tile_side,
                              kind_b=sym_kind,
                              expr_b=sym_expr,
                              has_mask=mask_an is not None)
            inner_state.add_node(binop)
            inner_state.add_edge(a_edge.src, a_edge.src_conn, binop, "_a", dace.Memlet.from_memlet(a_edge.data))
            if sym_kind == "Tile":
                self._wire_materialised_tile(inner_state, binop, "_b", sym_an_name)
        else:
            binop = TileBinop(name=f"{tasklet.label}_binop_sym",
                              widths=tuple(self.widths),
                              op=op,
                              kind_a=sym_kind,
                              kind_b=kind_tile_side,
                              expr_a=sym_expr,
                              has_mask=mask_an is not None)
            inner_state.add_node(binop)
            inner_state.add_edge(a_edge.src, a_edge.src_conn, binop, "_b", dace.Memlet.from_memlet(a_edge.data))
            if sym_kind == "Tile":
                self._wire_materialised_tile(inner_state, binop, "_a", sym_an_name)
        self._wire_mask(inner_state, binop, mask_an)
        inner_state.add_edge(binop, "_c", out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _wire_materialised_tile(self, inner_state: SDFGState, lib_node, dst_conn: str, tile_name: str) -> None:
        """Wire ``<materialised_tile_an> -> lib_node.<dst_conn>`` with the full tile subset."""
        from dace.sdfg.nodes import AccessNode
        # Find or create the AN. The materialiser already adds it; reuse to keep scheduling.
        existing = next((n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == tile_name), None)
        an = existing if existing is not None else inner_state.add_access(tile_name)
        subset = ", ".join(f"0:{w}" for w in self.widths)
        inner_state.add_edge(an, None, lib_node, dst_conn, dace.Memlet(f"{tile_name}[{subset}]"))

    def _convert_unop_with_symbol(self, inner_state: SDFGState, tasklet: Tasklet, detected,
                                  iter_vars: Tuple[str, ...]) -> bool:
        """0-in-conn unary: ``_o = <op>(<expr>)`` or ``_o = -<expr>``.

        Resolves the Symbol like ``_convert_binop_with_symbol``: invariant -> Symbol with
        ``expr_a``; lane-id-dependent -> materialised tile wired to ``_a``.
        """
        out_conn, op, symbol_expr = detected
        out_edges = list(inner_state.out_edges(tasklet))
        if not out_edges:
            return False
        out_edge = out_edges[0]
        sym_kind, sym_expr, sym_an_name = self._resolve_symbol_operand(inner_state, symbol_expr, iter_vars)
        mask_an = self._find_mask_an(inner_state)
        unop = TileUnop(name=f"{tasklet.label}_unop_sym",
                        widths=tuple(self.widths),
                        op=op,
                        kind_a=sym_kind,
                        expr_a=sym_expr,
                        has_mask=mask_an is not None)
        inner_state.add_node(unop)
        if sym_kind == "Tile":
            self._wire_materialised_tile(inner_state, unop, "_a", sym_an_name)
        self._wire_mask(inner_state, unop, mask_an)
        inner_state.add_edge(unop, "_c", out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))
        for edge in out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _convert_binop_with_two_symbols(self, inner_state: SDFGState, tasklet: Tasklet, detected,
                                        iter_vars: Tuple[str, ...]) -> bool:
        """0-in-conn binary: ``_o = <expr_a> <op> <expr_b>``.

        Each operand resolved independently to invariant Symbol or materialised Tile.
        """
        out_conn, op, expr_a_str, expr_b_str = detected
        out_edges = list(inner_state.out_edges(tasklet))
        if not out_edges:
            return False
        out_edge = out_edges[0]
        kind_a, sym_expr_a, an_a = self._resolve_symbol_operand(inner_state, expr_a_str, iter_vars)
        kind_b, sym_expr_b, an_b = self._resolve_symbol_operand(inner_state, expr_b_str, iter_vars)
        mask_an = self._find_mask_an(inner_state)
        binop = TileBinop(name=f"{tasklet.label}_binop_two_sym",
                          widths=tuple(self.widths),
                          op=op,
                          kind_a=kind_a,
                          kind_b=kind_b,
                          expr_a=sym_expr_a,
                          expr_b=sym_expr_b,
                          has_mask=mask_an is not None)
        inner_state.add_node(binop)
        if kind_a == "Tile":
            self._wire_materialised_tile(inner_state, binop, "_a", an_a)
        if kind_b == "Tile":
            self._wire_materialised_tile(inner_state, binop, "_b", an_b)
        self._wire_mask(inner_state, binop, mask_an)
        inner_state.add_edge(binop, "_c", out_edge.dst, out_edge.dst_conn, dace.Memlet.from_memlet(out_edge.data))
        for edge in out_edges:
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
        # Output transient shape is pre-determined by WidenAccesses.
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

    def _convert_inner(self, inner_sdfg: SDFG, iter_vars: Tuple[str, ...]) -> int:
        """Walk every state of ``inner_sdfg`` and convert recognised binop tasklets.

        :returns: Number of conversions performed.
        """
        converted = 0
        for inner_state in inner_sdfg.states():
            for node in list(inner_state.nodes()):
                if not isinstance(node, Tasklet):
                    continue
                if self._convert_one(inner_state, node, iter_vars):
                    converted += 1
        return converted

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Walk every tile-tagged body NSDFG; convert recognised tasklets to tile lib nodes.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Pipeline results (unused).
        :returns: Number of tasklets converted, or ``None`` if zero.
        """
        total = 0
        K = len(self.widths)
        for _state, nsdfg_node, map_entry in self._body_nsdfgs(sdfg):
            iter_vars = tuple(map_entry.map.params[-K:])
            total += self._convert_inner(nsdfg_node.sdfg, iter_vars)
        return total or None
