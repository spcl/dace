# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Convert in-body tasklets to ``TileBinop`` / ``TileUnop`` / ``TileITE`` / ``TileReduce``.

After :class:`InsertTileLoadStore` stages non-transient reads through tile transients,
the body still holds raw per-lane scalar tasklets. Walk the same tile-tagged body NSDFGs,
replace each with the matching tile lib node → post-expansion pure-loop body operates on
tile-shape register transients (design 5.1 + 6.7). Handles binary / unary / ITE /
masked-write / reduction shapes over Tile / Scalar / Symbol operands.
"""
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np

import dace
from dace import properties
from dace.libraries.tileops import TileBinop, TileITE, TileLoad, TileMaskGen, TileReduce, TileStore, TileUnop
from dace.sdfg import SDFG
from dace.sdfg.nodes import CodeBlock, MapEntry, NestedSDFG, Tasklet
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.map_predicates import is_vectorizable_map
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant, logical_binops_are_bool,
                                                                            mask_connectors_are_bool,
                                                                            no_duplicate_connector_edges,
                                                                            no_memlet_dim_mismatch)

#: Binary ops → :class:`TileBinop`. Comparisons (``< <= > >= == !=``) produce bool tile
#: outputs → :class:`TileITE` cond input (design 7.5). Powers arrive as the function-form
#: ``pow`` / ``ipow`` (``PowerOperatorExpansion`` rewrites every ``**`` to ``pow`` or an
#: unrolled product; ``RelaxIntegerPowers`` relaxes an integer-exponent ``pow`` → ``ipow``).
#: ``**`` is retained for robustness against a residual bare operator.
_SUPPORTED_BINOPS = {
    "+", "-", "*", "/", "%", "py_mod", "**", "pow", "ipow", "min", "max", "atan2", "hypot", "fmod", "<", "<=", ">",
    ">=", "==", "!=", "&&", "||", "&", "|", "^"
}

#: Binops in function-call form ``op(a, b)``, not infix. ``py_mod`` = Python/NumPy modulo
#: → ``dace::math::py_mod`` (``RewriteModuloToPyMod`` rewrites every ``%`` to it for
#: divisor-sign semantics). ``pow`` / ``ipow`` are the canonical power spellings; ``**``
#: keeps its own infix case below.
_FUNCTION_FORM_BINOPS = ("min", "max", "py_mod", "atan2", "hypot", "fmod", "pow", "ipow")

#: Comparison ops: result dtype ``bool`` regardless of operand dtype (``double > double
#: → bool``). :meth:`_convert_binop`'s mixed-dtype guard excludes output dtype from
#: operand-uniformity for these.
_COMPARISON_BINOPS = {"<", "<=", ">", ">=", "==", "!="}

#: Reduction ops → :class:`TileReduce`: associative subset of :data:`_SUPPORTED_BINOPS`
#: (excludes non-associative ``-`` / ``/``).
_SUPPORTED_REDUCE_OPS = {"+", "*", "min", "max"}

#: ``TileUnop.op`` labels this pass lowers to, matched against the tasklet body
#: (after the DaCe paren wrap is stripped). The leading ``-`` form aliases ``neg``.
_SUPPORTED_UNOPS = {
    "neg",
    "not",
    "abs",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "floor",
    "ceil",
    "tanh",
}

#: Registered dtype-cast call names (``float64`` / ``int32`` / ...), from the dtype
#: registry (never hardcoded). A call ``[dace.]<dtype>(<arg>)`` → ``TileUnop`` whose
#: ``op`` IS the dtype name (an explicit per-lane cast). Set membership (not a regex)
#: excludes non-cast calls like ``int_floor``.
_CAST_OP_NAMES = frozenset(s.split("::")[-1] for s in dace.dtypes.TYPECLASS_TO_STRING.values())


def _cast_call_inner(rhs: str) -> Optional[Tuple[str, str]]:
    """If ``rhs`` is a dtype-cast call ``[dace.]<dtype>(<arg>)``, return
    ``(dtype_name, arg)`` (``dtype_name`` a registered cast op); else ``None``.

    :param rhs: Tasklet-body RHS (paren-stripped).
    """
    rhs = rhs.strip()
    open_idx = rhs.find("(")
    if open_idx < 0 or not rhs.endswith(")"):
        return None
    name = rhs[:open_idx].strip()
    if name.startswith("dace."):
        name = name[len("dace."):]
    if name not in _CAST_OP_NAMES:
        return None
    inner = rhs[open_idx + 1:-1].strip()
    return (name, inner) if inner else None


def _normalize_python_tasklet_body(body: str) -> Optional[str]:
    """Rewrite Python boolean syntax (``or`` / ``and``) to the C forms (``||`` / ``&&``)
    the binop detectors match. Returns ``None`` for bodies containing ``@`` (matmul is
    not per-lane element-wise, so refuse rather than mis-expand).

    Lift-emitted tasklets (e.g. ``SameWriteSetIfElseToITECFG``'s ``combine_cond``) write
    Python ``_o = (_c_0 or _c_1)``; without this the detectors miss them
    (``_SUPPORTED_BINOPS`` only holds ``||`` / ``&&``) and codegen emits a scalar-bool
    assign into a widened ``bool*`` tile — a hard compile error. Unary ``not`` is left
    for ``_detect_unop`` (→ ``TileUnop(op='!')``); mapping it to ``x ^ 1`` would be
    XOR-with-1, not logical NOT, and break on a scalar-bool ``x``.

    :param body: Stripped tasklet body (no trailing ``;``).
    """
    if '@' in body:
        return None
    out = re.sub(r'\bor\b', '||', body)
    out = re.sub(r'\band\b', '&&', out)
    return out


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

        Iter_var-dependent Symbols differ per lane → cannot broadcast; must materialise
        as tile-shape transients. Independent ones are loop-invariant and embed inline as
        ``expr_a`` / ``expr_b`` on the lib node.

        :param expr: Expression string (e.g. ``"N + 1"``, ``"2 * ii + jj"``).
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

        * invariant ``Symbol`` → ``(kind="Symbol", expr=<expr>, an_name=None)``,
        * lane-id-dependent ``Tile`` → ``(kind="Tile", expr=None, an_name=<tile>)``,
          materialising an int64 per-lane tile via :meth:`_materialise_lane_id_tile`.

        Distinct from the gather-index path
        (:meth:`InsertTileLoadStore._stage_index_via_tileops`): here iter_var ``ii`` is
        the OUTER tile start, per-lane value ``ii + __l``; gather reads a data-dependent
        ``idx[...]`` into an index tile (lane offset IS ``__l``). Do not conflate.
        """
        if not iter_vars or not self._is_lane_id_dependent(expr, iter_vars):
            return "Symbol", expr, None
        an_name = self._materialise_lane_id_tile(inner_state, expr, iter_vars)
        return "Tile", None, an_name

    def _materialise_lane_id_tile(self, inner_state: SDFGState, expr: str, iter_vars: Tuple[str, ...]) -> str:
        """Mint a per-lane int64 tile = ``expr`` evaluated at
        ``(iter_var_k → iter_var_k + __l_k)`` for each tile dim ``k``.

        K=1, expr ``"ii"`` → ``_lane_tile[l] = (ii) + l``.
        K=2, expr ``"2 * ii + jj"`` → ``_lane_tile[l0, l1] = 2*(ii+l0) + (jj+l1)``.
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
            # constexpr width + DACE_UNROLL → lane loop lowers to SIMD.
            code_lines.append(f"{'    ' * d}constexpr std::size_t __W{d} = {widths[d]};")
            code_lines.append(f"{'    ' * d}DACE_UNROLL")
            code_lines.append(f"{'    ' * d}for (std::size_t __l{d} = 0; __l{d} < __W{d}; ++__l{d}) {{")
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
        # Cross-state fallback: a masked-tail body NSDFG can split compute and store
        # across states (TileMaskGen in one, TileReduce / TileBinop in another). The mask
        # transient is SDFG-wide and states run producer-first, so read it here through a
        # fresh AccessNode. Skipping this lowers the masked op UNMASKED — benign for
        # elementwise (the masked store discards inactive lanes) but WRONG for TileReduce
        # (folds past-the-tail lanes into the result).
        sdfg = inner_state.sdfg
        mask_names = {
            e.dst.data
            for s in sdfg.states()
            for n in s.nodes() if isinstance(n, TileMaskGen) for e in s.out_edges(n)
            if e.src_conn == "_o" and isinstance(e.dst, AccessNode)
        }
        if len(mask_names) == 1:
            mask_name = next(iter(mask_names))
            existing = next((n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == mask_name), None)
            return existing if existing is not None else inner_state.add_access(mask_name)
        return None

    def _wire_mask(self, inner_state: SDFGState, lib_node, mask_an) -> None:
        """Wire ``mask_an → lib_node._mask`` with a full-tile subset memlet.

        Mask-combine contract (user 2026-06-12): if a ``_mask`` edge is ALREADY wired,
        AND-combine the new condition with the existing one (before the consumer) via
        :class:`TileBinop` ``op='&'`` over two bool tiles — no dedicated ``TileMaskAnd``
        node. Today the pipeline wires iter-mask once and never re-wires, so this AND
        path is not exercised yet.

        Connector-name contract (user 2026-06-12): every K-dim tile lib node's gating
        predicate is wired via ``_mask`` (TileLoad / Store / Binop / Unop / Reduce /
        ITE — the last previously used ``_cond``).
        """
        if mask_an is None:
            return
        subset = ", ".join(f"0:{w}" for w in self.widths)
        existing = [e for e in inner_state.in_edges(lib_node) if e.dst_conn == "_mask"]
        if existing:
            # AND-combine: a ``_mask`` edge is already wired (e.g. a masked-write ``cond``
            # on a store already carrying the tile iteration mask). Emit ``TileBinop(op='&')``
            # over the two bool tiles and rewire — the new condition gates in addition.
            existing_edge = existing[0]
            combined = self._and_mask_tiles(inner_state, existing_edge.src, mask_an)
            inner_state.remove_edge(existing_edge)
            inner_state.add_edge(combined, None, lib_node, "_mask", dace.Memlet(f"{combined.data}[{subset}]"))
            return
        inner_state.add_edge(mask_an, None, lib_node, "_mask", dace.Memlet(f"{mask_an.data}[{subset}]"))

    def _and_mask_tiles(self, inner_state: SDFGState, an_a, an_b):
        """Mint a bool tile ``= an_a && an_b`` via :class:`TileBinop`, return its output
        AccessNode. Both operands are ``widths``-shaped bool mask tiles; ``&&`` lowers
        per-lane to logical AND and satisfies ``logical_binops_are_bool`` (bool in/out)."""
        sdfg = inner_state.sdfg
        widths = tuple(self.widths)
        name, _ = sdfg.add_array("_mask_and",
                                 shape=widths,
                                 dtype=dace.bool_,
                                 transient=True,
                                 storage=dace.dtypes.StorageType.Register,
                                 find_new_name=True)
        binop = TileBinop(name="mask_and", widths=widths, op="&&", kind_a="Tile", kind_b="Tile")
        inner_state.add_node(binop)
        subset = ", ".join(f"0:{w}" for w in widths)
        inner_state.add_edge(an_a, None, binop, "_a", dace.Memlet(f"{an_a.data}[{subset}]"))
        inner_state.add_edge(an_b, None, binop, "_b", dace.Memlet(f"{an_b.data}[{subset}]"))
        combined = inner_state.add_access(name)
        inner_state.add_edge(binop, "_c", combined, None, dace.Memlet(f"{name}[{subset}]"))
        return combined

    def _body_nsdfgs(self, sdfg: SDFG):
        """Yield ``(state, nsdfg_node, map_entry)`` for every tile-tagged body NSDFG.

        Mirror of the walker shape used by :class:`InsertTileLoadStore`.
        Skips ``__scalar_tail`` (postamble step-1 loop) and ``__tile_k1_tail``
        (pinned-K=1 postamble) since neither runs the K-D tile-op chain.
        """
        from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                           TILE_K1_TAIL_MARKER)
        K = len(self.widths)
        for node, parent in sdfg.all_nodes_recursive():
            if not isinstance(node, MapEntry):
                continue
            if not isinstance(parent, SDFGState):
                continue
            try:
                if not is_vectorizable_map(parent, node):
                    continue
            except (StopIteration, ValueError):
                continue
            if len(node.map.params) < K:
                continue
            if node.map.label.endswith(SCALAR_TAIL_MARKER) or node.map.label.endswith(TILE_K1_TAIL_MARKER):
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
            body = tasklet.code.as_string
            body = body.strip().rstrip(";").strip()
            out_conn = next(iter(tasklet.out_connectors))
            a_conn = next(iter(tasklet.in_connectors))
            body = _normalize_python_tasklet_body(body)
            if body is None:
                return None
            for op in _SUPPORTED_BINOPS:
                if op in _FUNCTION_FORM_BINOPS:
                    continue
                for form in (f"{out_conn} = {a_conn} {op} {a_conn}", f"{out_conn} = ({a_conn} {op} {a_conn})"):
                    if body == form:
                        return out_conn, a_conn, a_conn, op
            return None
        if len(tasklet.in_connectors) != 2:
            return None
        body = tasklet.code.as_string
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        in_conns = list(tasklet.in_connectors)
        body = _normalize_python_tasklet_body(body)
        if body is None:
            return None
        # Try every op against both operand orderings. DaCe wraps the RHS in parens
        # (``_o = (_a + _b)``); accept both. Function-form ops use ``op(a, b)`` (this
        # covers ``pow`` / ``ipow``); ``**`` keeps the bare-operator infix case for a
        # residual power PowerOperatorExpansion did not rewrite.
        for op in _SUPPORTED_BINOPS:
            for a, b in (in_conns, list(reversed(in_conns))):
                if op in _FUNCTION_FORM_BINOPS:
                    forms = (f"{out_conn} = {op}({a}, {b})", f"{out_conn} = ({op}({a}, {b}))")
                elif op == "**":
                    forms = (f"{out_conn} = {a} ** {b}", f"{out_conn} = ({a} ** {b})")
                else:
                    forms = (f"{out_conn} = {a} {op} {b}", f"{out_conn} = ({a} {op} {b})")
                if body in forms:
                    return out_conn, a, b, op
        return None

    def _detect_binop_with_symbol(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str, str, str]]:
        """Detect a binop with ONE Tile/Scalar operand and ONE Symbol operand.

        Matches ``_out = _a <op> <expr>`` or ``_out = <expr> <op> _a`` where ``<expr>``
        is an outer-scope symbol or numeric literal (not a connector). Returns
        ``(out_conn, tile_or_scalar_conn, op, symbol_side, symbol_expr)``, ``symbol_side``
        "a" (symbol LHS) or "b" (symbol RHS). Complements ``_detect_binop`` (2 in-conns):
        the Symbol case has ONE in-connector, the other operand an inline string.
        """
        if len(tasklet.in_connectors) != 1 or len(tasklet.out_connectors) != 1:
            return None
        body = tasklet.code.as_string
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        a_conn = next(iter(tasklet.in_connectors))
        body = _normalize_python_tasklet_body(body)
        if body is None:
            return None
        # Strip outer parens (DaCe wraps the RHS).
        rhs = body[len(f"{out_conn} = "):]
        if rhs.startswith("(") and rhs.endswith(")"):
            rhs = rhs[1:-1]
        # For each op, split the rhs into ``<a_conn> <op> <expr>`` or ``<expr> <op> <a_conn>``.
        for op in _SUPPORTED_BINOPS:
            if op in _FUNCTION_FORM_BINOPS:
                # min(_a, expr) / min(expr, _a) -- both orderings.
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

    def _detect_affine_unit_with_symbol(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str, str, str]]:
        """Detect a 1-tile-operand body AFFINE in that operand with unit coefficient.

        Complements :meth:`_detect_binop_with_symbol`, whose textual split cannot isolate
        the operand when it is BURIED among symbolic terms — e.g. reverse gather index
        ``_o = (LEN_1D - _in0) - 1`` (TSVC s4114 ``c[LEN_1D - ip[i] - 1]``). Such a body is
        ``coeff * _in0 + offset`` (``offset`` free of ``_in0``); ``coeff == +1`` →
        ``_in0 + offset`` (side b), ``coeff == -1`` → ``offset - _in0`` (side a), each a
        single-symbol :class:`TileBinop`. Returns the :meth:`_detect_binop_with_symbol`
        tuple shape. Non-unit coefficients (need a multiply too) → ``None``.
        """
        if len(tasklet.in_connectors) != 1 or len(tasklet.out_connectors) != 1:
            return None
        out_conn = next(iter(tasklet.out_connectors))
        a_conn = next(iter(tasklet.in_connectors))
        body = _normalize_python_tasklet_body(tasklet.code.as_string.strip().rstrip(";").strip())
        if body is None or not body.startswith(f"{out_conn} = "):
            return None
        rhs = body[len(f"{out_conn} = "):].strip()
        from dace import symbolic
        from dace.transformation.passes.vectorization.utils.tile_access import (_affine_coeff_for, _affine_offset_for)
        try:
            expr = symbolic.pystr_to_symbolic(rhs)
        except Exception:  # noqa: BLE001 -- unparseable RHS: not an affine body
            return None
        # Poly-based affine helpers return None for non-affine / degree>1 / relational.
        # Only |coeff| == 1 maps to a single-symbol binop; others need a multiply too.
        coeff = _affine_coeff_for(expr, a_conn)
        offset = _affine_offset_for(expr, a_conn)
        if coeff is None or offset is None:
            return None
        if coeff == 1:
            return out_conn, a_conn, "+", "b", str(offset)
        if coeff == -1:
            return out_conn, a_conn, "-", "a", str(offset)
        return None  # |coeff| != 1 needs a multiply too -- deferred

    def _detect_const_assign(self, tasklet: Tasklet) -> Optional[Tuple[str, str]]:
        """If ``tasklet`` is a 0-in-connector pure constant / Symbol assignment
        (``_o = <expr>``, no operator — a literal or Symbol), return ``(out_conn, expr)``;
        else ``None``. The simplest output-side Symbol case (memset / broadcast /
        set-to-zero); :meth:`_convert_const_assign` lowers it to a Symbol-source
        ``TileLoad`` broadcast.
        """
        if len(tasklet.in_connectors) != 0 or len(tasklet.out_connectors) != 1:
            return None
        body = tasklet.code.as_string
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        if not body.startswith(f"{out_conn} = "):
            return None
        rhs = body[len(f"{out_conn} = "):].strip()
        if rhs.startswith("(") and rhs.endswith(")"):
            rhs = rhs[1:-1].strip()
        if not rhs:
            return None
        # Reject op shapes (handled by other detectors): no top-level binary operator,
        # no function-call with comma-separated args.
        for op_token in (" + ", " - ", " * ", " / ", " % ", " ** ", " < ", " > ", " == ", " != ", " <= ", " >= ",
                         " and ", " or "):
            if op_token in rhs:
                return None
        # Reject function-form binops / ITE.
        for fn in ("min(", "max(", "pow(", "ipow(", "ITE("):
            if rhs.startswith(fn):
                return None
        # Reject leading minus (handled by _detect_unop_with_symbol negation case).
        if rhs.startswith("-") and not rhs.startswith("-"):  # never triggers; placeholder
            return None
        return out_conn, rhs

    def _convert_const_assign(self, inner_state: SDFGState, tasklet: Tasklet, detected, iter_vars: Tuple[str,
                                                                                                         ...]) -> bool:
        """Replace ``_o = <const_expr>`` (loop-invariant literal / symbol store) with a
        ``TileLoad(src_kind='Symbol')`` broadcast writing the value to every lane — no CPP
        fill, no intermediate transient, no AN→AN copy (user 2026-06-15: const/symbol→tile
        broadcast is a tile op). Symbol-source ``TileLoad`` declares no ``_src``; the
        expansion embeds ``src_expr`` inline. A non-tile (true scalar) output stays a
        single-statement python scalar tasklet — scalar→scalar needs no tile op."""
        out_conn, expr = detected
        out_edges = list(inner_state.out_edges(tasklet))
        if not out_edges:
            return False
        out_edge = out_edges[0]
        # Widen the output transient to tile shape when widenable; a genuinely
        # scalar (non-widenable) output stays the python scalar assign.
        self._ensure_output_widened(inner_state, out_edge)
        # Tile broadcast ONLY when the output writes a tile-shape AccessNode transient. A
        # const feeding directly into another tasklet (``__t = 0`` → an ITE-select operand)
        # has a Tasklet (no ``.data``) as ``out_edge.dst``; leave it as the scalar tasklet
        # the downstream tile op reads as a Symbol/Scalar broadcast.
        out_dst = out_edge.dst
        out_desc = (inner_state.sdfg.arrays.get(out_dst.data)
                    if out_edge.data is not None and isinstance(out_dst, dace.nodes.AccessNode) else None)
        is_tile_out = (out_desc is not None and isinstance(out_desc, dace.data.Array)
                       and tuple(out_desc.shape) == tuple(self.widths))
        if not is_tile_out:
            return False  # scalar const store -- keep the single-statement python tasklet
        tl = TileLoad(name=f"{tasklet.label}_const_bcast",
                      widths=tuple(self.widths),
                      src_kind="Symbol",
                      src_expr=str(expr))
        inner_state.add_node(tl)
        subset = ", ".join(f"0:{w}" for w in self.widths)
        inner_state.add_edge(tl, "_dst", out_edge.dst, out_edge.dst_conn, dace.Memlet(f"{out_edge.dst.data}[{subset}]"))
        inner_state.remove_edge(out_edge)
        inner_state.remove_node(tasklet)
        return True

    def _detect_unop_with_symbol(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str]]:
        """If ``tasklet`` is a 0-in-connector unary symbol body (``_o = <op>(<expr>)`` or
        ``_o = -<expr>``), return ``(out_conn, op_label, expr)``; else ``None``.

        Downstream lib node gets ``kind_a=Symbol`` + ``expr_a=<expr>``, no ``_a`` connector
        (design 6.2: Symbol operands embedded inline).
        """
        if len(tasklet.in_connectors) != 0 or len(tasklet.out_connectors) != 1:
            return None
        body = tasklet.code.as_string
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        rhs = body[len(f"{out_conn} = "):]
        if rhs.startswith("(") and rhs.endswith(")"):
            rhs = rhs[1:-1].strip()
        # Explicit dtype cast ``_o = dace.float64(<expr>)`` → TileUnop kind_a=Symbol,
        # op = the dtype name.
        hit = _cast_call_inner(rhs)
        if hit is not None:
            return out_conn, hit[0], hit[1]
        # Negation: ``_o = -<expr>`` / ``_o = (- <expr>)``.
        if rhs.startswith("-"):
            expr = rhs[1:].strip()
            if expr.startswith("(") and expr.endswith(")"):
                expr = expr[1:-1].strip()
            if expr:
                return out_conn, "neg", expr
        # Function-form: ``_o = <op>(<expr>)`` (DaCe may wrap the argument in extra parens).
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
        or ``_o = <op>(<expr_a>, <expr_b>)``), return ``(out_conn, op, expr_a, expr_b)``;
        else ``None``.

        Both operands inline Symbol exprs; downstream lib node built with ``kind_a=Symbol``,
        ``kind_b=Symbol``, ``expr_a / expr_b`` set. Output may be Scalar (design 6.2:
        all-Symbol → Scalar output).
        """
        if len(tasklet.in_connectors) != 0 or len(tasklet.out_connectors) != 1:
            return None
        body = tasklet.code.as_string
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        rhs = body[len(f"{out_conn} = "):]
        if rhs.startswith("(") and rhs.endswith(")"):
            rhs = rhs[1:-1].strip()
        for op in _SUPPORTED_BINOPS:
            if op in _FUNCTION_FORM_BINOPS:
                prefix = f"{op}("
                if rhs.startswith(prefix) and rhs.endswith(")"):
                    inner = rhs[len(prefix):-1]
                    # Split on the top-level comma (both exprs paren/identifier-shaped).
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
                # Top-level split on the op (skip nested exprs).
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
        body = tasklet.code.as_string
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        a_conn = next(iter(tasklet.in_connectors))
        # Explicit dtype cast ``_o = dace.float64(_a)`` → TileUnop whose op IS the dtype
        # name. DaCe may wrap the RHS in an extra paren layer; try as-is and stripped.
        rhs = body[len(f"{out_conn} = "):].strip()
        for cand in (rhs, rhs[1:-1].strip() if rhs.startswith("(") and rhs.endswith(")") else None):
            if not cand:
                continue
            hit = _cast_call_inner(cand)
            if hit is not None and hit[1] == a_conn:
                return out_conn, a_conn, hit[0]
        # Negation: ``_o = -_a`` / ``(-_a)`` / ``(- _a)`` (DaCe may space the unary minus).
        for form in (f"{out_conn} = -{a_conn}", f"{out_conn} = (-{a_conn})", f"{out_conn} = (- {a_conn})"):
            if body == form:
                return out_conn, a_conn, "neg"
        # Logical NOT: Python ``not _a`` (keyword, not call). ``SameWriteSetIfElseToITECFG``
        # emits ``_o = (not _c_0)`` → per-lane TileUnop(op='!'). Body is Python, so only
        # ``not`` appears (never C ``!``).
        for form in (f"{out_conn} = not {a_conn}", f"{out_conn} = (not {a_conn})"):
            if body == form:
                return out_conn, a_conn, "not"
        # Function-call ops: ``abs(_a)`` / ``math.exp(_a)`` / etc. Accept bare name and
        # ``math.`` / ``std::`` prefixes (in case RemoveMathCall didn't run / wasn't needed).
        for op in _SUPPORTED_UNOPS:
            if op in ("neg", "not"):
                continue
            prefixes = (op, f"math.{op}", f"std::{op}")
            for pref in prefixes:
                for form in (f"{out_conn} = {pref}({a_conn})", f"{out_conn} = ({pref}({a_conn}))"):
                    if body == form:
                        return out_conn, a_conn, op
        return None

    def _operand_kind(self, inner_state: SDFGState, edge) -> str:
        """Classify the lib-node operand kind from the source descriptor (design 6.5):
        Scalar / length-1 Array source → ``"Scalar"`` (hardware splat across lanes);
        tile-shape source (LINEAR / AFFINE / REPLICATE / MODULAR / GATHER) → ``"Tile"``.

        :param inner_state: Inner SDFGState holding the edge.
        :param edge: The wired in-edge to the tasklet / lib node connector.
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
        """If ``tasklet`` is an in-place RMW reduction ``_acc = _acc <op> _val`` (or the
        commutative ``_acc = _val <op> _acc``), return ``(acc_conn, val_conn, op)``; else
        ``None``. Criteria: 2 in-conns + 1 out-conn; out-conn name matches an in-conn
        (in-place RMW); body exactly ``<out> = <out> <op> <other>`` (or symmetric) with op
        in :data:`_SUPPORTED_REDUCE_OPS`.

        Post-walker shape: tile input on ``_val`` (from a bridge), scalar accumulator on
        ``_acc``; ``TileReduce`` folds the full tile to a single boundary scalar write
        (user 2026-06-09).
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
        body = tasklet.code.as_string
        body = body.strip().rstrip(";").strip()
        for op in _SUPPORTED_REDUCE_OPS:
            if op in _FUNCTION_FORM_BINOPS:
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

    def _detect_augassign_reduction(self, inner_state: SDFGState, tasklet: Tasklet) -> Optional[Tuple[str, str, str]]:
        """Recognize the standard ``WCRToAugAssign`` reduction ``__out = __in1 op __in2``.

        Unlike :meth:`_detect_reduction`'s same-connector RMW, ``WCRToAugAssign`` uses
        DISTINCT connector names (one input reads the accumulator back, the other is the
        addend, ``__out`` writes the accumulator). Recognized by SHAPE signature via edge
        descriptors (invisible to the tasklet-only view): addend input is a TILE, output
        accumulator a SCALAR / length-1 sink. Design 6.2: a TileBinop with a tile input
        MUST produce a tile output, so tile-in + scalar-out is necessarily a fold. The
        scalar read-back may be staged through a copy, so data-identity would miss it —
        the shape signature does not.

        Returns ``(acc_conn, val_conn, op)`` for :meth:`_convert_reduction`, or ``None``.
        A plain element-wise binop (tile in → tile out) never matches.
        """
        if len(tasklet.in_connectors) != 2 or len(tasklet.out_connectors) != 1:
            return None
        out_conn = next(iter(tasklet.out_connectors))
        if out_conn in tasklet.in_connectors:
            return None  # same-connector RMW -> _detect_reduction owns it
        a, b = list(tasklet.in_connectors)
        body = tasklet.code.as_string.strip().rstrip(";").strip()
        matched_op = None
        for op in _SUPPORTED_REDUCE_OPS:
            if op in _FUNCTION_FORM_BINOPS:
                forms = (f"{out_conn} = {op}({a}, {b})", f"{out_conn} = ({op}({a}, {b}))",
                         f"{out_conn} = {op}({b}, {a})", f"{out_conn} = ({op}({b}, {a}))")
            else:
                forms = (f"{out_conn} = {a} {op} {b}", f"{out_conn} = ({a} {op} {b})", f"{out_conn} = {b} {op} {a}",
                         f"{out_conn} = ({b} {op} {a})")
            if body in forms:
                matched_op = op
                break
        if matched_op is None:
            return None
        out_edges = inner_state.out_edges(tasklet)
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        if len(out_edges) != 1 or a not in in_edges or b not in in_edges:
            return None

        def _elems(edge) -> Optional[int]:
            """Element count of an edge's memlet subset (the widened tile / scalar)."""
            desc = inner_state.sdfg.arrays.get(edge.data.data)
            if desc is None:
                return None
            sub = edge.data.subset
            n = sub.num_elements() if sub is not None else desc.total_size
            try:
                return int(n)
            except (TypeError, ValueError):
                return None

        out_n = _elems(out_edges[0])
        if out_n != 1:  # the accumulator sink must be a scalar / length-1
            return None
        a_n, b_n = _elems(in_edges[a]), _elems(in_edges[b])
        # Exactly one input is the tile addend (>1 lanes); the other is the scalar
        # accumulator read-back. A tile input with a scalar output is a fold, not a binop.
        if a_n is not None and a_n > 1 and b_n == 1:
            return b, a, matched_op
        if b_n is not None and b_n > 1 and a_n == 1:
            return a, b, matched_op
        return None

    def _detect_ite(self, tasklet: Tasklet) -> Optional[Tuple[str, ...]]:
        """If ``tasklet`` is a ternary if-then-else, return
        ``(out_conn, cond, t, e, has_t_sym, has_e_sym)``; else ``None``.

        Recognised forms:

        * Python ternary (3 in-conn): ``_o = _t if _cond else _e`` (maybe parenthesised).
        * ITE call (3 in-conn): ``_o = ITE(_cond, _t, _e)``.
        * ITE call with literal/Symbol arm (2 in-conn): ``_o = ITE(_cond, _t, <expr>)`` or
          ``_o = ITE(_cond, <expr>, _e)``.
        """
        n_in = len(tasklet.in_connectors)
        # n_in == 1: ``ITE(cond, <sym>, <sym>)`` — only cond is a connector, both arms
        # Symbols/literals (find-first phi ``ITE(pred, _loop_it_0, LEN_1D)``). n_in in
        # (2, 3): one or zero Symbol arms.
        if n_in not in (1, 2, 3) or len(tasklet.out_connectors) != 1:
            return None
        body = tasklet.code.as_string
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
        # ITE(cond, t, e) function form: 3-in-conn and the 2-in-conn-with-symbol cases.
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
                        return out_conn, cond_arg, t_arg, e_arg, False, True
                    if not is_t_conn and is_e_conn:
                        return out_conn, cond_arg, t_arg, e_arg, True, False
                    # BOTH arms Symbol/literal — find-first phi ``ITE(cond, _loop_it_0,
                    # LEN_1D)`` (lane-id index vs sentinel). ``_convert_ite`` lowers each
                    # arm via ``_plan_arm`` (lane-id → per-lane tile, invariant → inline).
                    return out_conn, cond_arg, t_arg, e_arg, True, True
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
        body = tasklet.code.as_string
        body = body.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        a_conn = next(iter(tasklet.in_connectors))
        if body in (f"{out_conn} = {a_conn}", f"{out_conn} = ({a_conn})"):
            return out_conn, a_conn
        return None

    def _convert_assign(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        """Replace a trivial ``_o = _a`` tasklet with a direct AN→AN edge (DaCe copies
        array-to-array natively; both edges already have matching tile-shape descriptors).

        The new memlet references the SOURCE bridge (read side) — a memlet typed against
        the destination confuses the validator's data-vs-endpoint consistency check.
        """
        out_conn, a_conn = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if a_conn not in in_edges or not out_edges:
            return False
        a_edge = in_edges[a_conn]
        out_edge = out_edges[0]
        # Scalar src → tile dst is a BROADCAST (e.g. ``c[jk, jc] = a[0]``): lower to
        # ``TileLoad(src_kind="Scalar")`` (per-lane splat), not a rank-mismatched AN→AN
        # copy (user 2026-06-14).
        if self._maybe_emit_scalar_broadcast(inner_state, tasklet, a_edge, out_edge):
            return True
        # Use the input-side memlet so the new edge's memlet.data matches its source.
        inner_state.add_edge(a_edge.src, a_edge.src_conn, out_edge.dst, out_edge.dst_conn,
                             dace.Memlet.from_memlet(a_edge.data))
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _maybe_emit_scalar_broadcast(self, inner_state: SDFGState, tasklet: Tasklet, a_edge, out_edge) -> bool:
        """Lower a trivial assign with a scalar SOURCE and ``widths``-shaped tile DEST to a
        ``TileLoad(src_kind="Scalar")`` broadcast (single value splat across lanes).
        Returns ``True`` when emitted (user 2026-06-14). The pure expansion handles
        ``src_kind="Scalar"`` (reads ``_src[0]``, writes every lane); an intrinsic
        hardware-splat is a TODO.
        """
        sdfg = inner_state.sdfg
        src_desc = sdfg.arrays.get(a_edge.data.data) if a_edge.data is not None else None
        dst_desc = sdfg.arrays.get(out_edge.data.data) if out_edge.data is not None else None
        if src_desc is None or dst_desc is None:
            return False
        widths = tuple(self.widths)
        src_is_scalar = (isinstance(src_desc, dace.data.Scalar)
                         or (isinstance(src_desc, dace.data.Array) and tuple(src_desc.shape) == (1, )))
        dst_is_tile = isinstance(dst_desc, dace.data.Array) and tuple(dst_desc.shape) == widths
        if not (src_is_scalar and dst_is_tile):
            return False
        tl = TileLoad(name=f"{tasklet.label}_bcast", widths=widths, src_kind="Scalar")
        inner_state.add_node(tl)
        # ``_src`` <- the scalar source (keep its scalar memlet); ``_dst`` -> the full tile.
        inner_state.add_edge(a_edge.src, a_edge.src_conn, tl, "_src", dace.Memlet.from_memlet(a_edge.data))
        tile_subset = ", ".join(f"0:{w}" for w in widths)
        inner_state.add_edge(tl, "_dst", out_edge.dst, out_edge.dst_conn,
                             dace.Memlet(data=out_edge.data.data, subset=tile_subset))
        inner_state.remove_edge(a_edge)
        inner_state.remove_edge(out_edge)
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
        # Only PYTHON tasklets carry the scalar op shapes this pass lowers. A CPP tasklet
        # is an already-lowered tile loop or a hand-written intrinsic — both pass through
        # untouched (user direction: keep intrinsics). Re-parsing a CPP for-loop mis-reads
        # its ``<`` bound as a comparison binop and emits garbage C++. Skip non-Python.
        if tasklet.language != dace.dtypes.Language.Python:
            return False
        # Masked conditional write ``_o = IT(cond, val)`` — detect FIRST (``IT(`` prefix
        # unambiguous vs every op shape and vs ``ITE(``). Lowers to a masked ``TileStore``,
        # then re-dispatches the stripped ``_o = val`` through the normal path.
        cond_write = self._detect_conditional_write(tasklet)
        if cond_write is not None:
            return self._convert_conditional_write(inner_state, tasklet, cond_write, iter_vars)
        # 0-in-conn (purely Symbol-driven) tasklets first.
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
        # Affine-in-one-operand, operand buried among symbolic terms (e.g. reverse
        # gather ``(LEN_1D - _in0) - 1``); reuses the symbol-binop converter.
        affine_sym = self._detect_affine_unit_with_symbol(tasklet)
        if affine_sym is not None:
            return self._convert_binop_with_symbol(inner_state, tasklet, affine_sym, iter_vars)
        # ITE (``if..else`` or ``ITE(...)``) — detect before binop/reduction so the
        # 2-in-conn ``ITE(_cond, _t, <symbol>)`` isn't miscaptured. Precise body match
        # keeps binop fall-through safe.
        ite = self._detect_ite(tasklet)
        if ite is not None:
            return self._convert_ite(inner_state, tasklet, ite, iter_vars)
        # 2 in-conn tasklets (reduction before plain binop so accumulators aren't miscaptured).
        reduction = self._detect_reduction(tasklet)
        if reduction is not None:
            return self._convert_reduction(inner_state, tasklet, reduction)
        # ``WCRToAugAssign`` reduction ``__out = __in1 op __in2`` uses DISTINCT connector
        # names (accumulator read-back + write), which :meth:`_detect_reduction` (same-conn
        # RMW) misses. Recognize by shape signature (tile-in + scalar-out) so it folds to a
        # TileReduce, BEFORE the binop check.
        aug_reduction = self._detect_augassign_reduction(inner_state, tasklet)
        if aug_reduction is not None:
            return self._convert_reduction(inner_state, tasklet, aug_reduction)
        binop = self._detect_binop(tasklet)
        if binop is not None:
            return self._convert_binop(inner_state, tasklet, binop)
        return False

    def _convert_reduction(self, inner_state: SDFGState, tasklet: Tasklet, detected) -> bool:
        acc_conn, val_conn, op = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edge_list = inner_state.out_edges(tasklet)
        # A reduction tasklet has exactly one output (the accumulator write). For the
        # same-connector RMW form ``out_conn == acc_conn``; for the ``WCRToAugAssign``
        # form (``_detect_augassign_reduction``) the output connector differs from the
        # accumulator read-back connector -- so resolve the write by the SINGLE out edge
        # rather than by ``out_edges[acc_conn]``.
        if acc_conn not in in_edges or val_conn not in in_edges or len(out_edge_list) != 1:
            return False
        val_edge = in_edges[val_conn]
        out_edge = out_edge_list[0]
        acc_in_edge = in_edges[acc_conn]
        mask_an = self._find_mask_an(inner_state)
        reduce_node = TileReduce(name=f"{tasklet.label}_reduce",
                                 widths=tuple(self.widths),
                                 op=op,
                                 has_mask=mask_an is not None)
        inner_state.add_node(reduce_node)
        self._wire_mask(inner_state, reduce_node, mask_an)
        # TileReduce connectors: _src (tile input) → _dst (scalar accumulator). The acc-input
        # edge dangles — TileReduce folds the whole tile in one shot, reading no separate
        # scalar accumulator; the new _dst edge writes the result on top.
        inner_state.add_edge(val_edge.src, val_edge.src_conn, reduce_node, "_src",
                             dace.Memlet.from_memlet(val_edge.data))
        inner_state.add_edge(reduce_node, "_dst", out_edge.dst, out_edge.dst_conn,
                             dace.Memlet.from_memlet(out_edge.data))
        acc_src = acc_in_edge.src
        for edge in list(in_edges.values()) + list(out_edge_list):
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        # The dangled accumulator read-back may leave its source AccessNode isolated — drop
        # it (a shared accumulator AN still carrying other edges is left intact).
        from dace.sdfg.nodes import AccessNode
        if isinstance(acc_src, AccessNode) and inner_state.degree(acc_src) == 0:
            inner_state.remove_node(acc_src)
        return True

    def _convert_ite(self, inner_state: SDFGState, tasklet: Tasklet, detected, iter_vars: Tuple[str, ...]) -> bool:
        """Convert a ternary tasklet to a TileITE lib node.

        ``detected`` = ``(out_conn, cond_arg, t_arg, e_arg, t_is_sym, e_is_sym)``; a
        ``*_is_sym`` arm is a Symbol / literal expr, not an in-connector. Arm lowering
        (user 2026-06-15): invariant Symbol → embedded inline (``kind_*='Symbol'`` +
        ``expr_*``, no connector, broadcast at expansion); lane-id-dependent Symbol →
        per-lane tile; connector reading a Scalar / length-1 source → broadcast full tile.
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
        out_dtype = inner_state.sdfg.arrays[out_edge.data.data].dtype
        subset = ", ".join(f"0:{w}" for w in self.widths)

        def _plan_arm(arg, is_sym):
            """Decide an arm's lowering. Returns ``(kind, expr, wire)`` where ``wire``
            is ``None`` for an inline Symbol arm or ``(src, src_conn, memlet)`` for a
            connector arm. ``kind`` is one of ``'Symbol'`` / ``'Tile'``."""
            from dace.sdfg.nodes import AccessNode
            if is_sym and not self._is_lane_id_dependent(arg, iter_vars):
                return "Symbol", arg, None  # inline -- no connector, no CPP fill
            if is_sym:
                an = self._materialise_symbol_to_tile(inner_state, arg, iter_vars, out_edge)
                return "Tile", None, (an, None, dace.Memlet(f"{an.data}[{subset}]"))
            edge = in_edges[arg]
            if self._is_scalar_or_len1_source(inner_state, edge):
                bname = self._broadcast_scalar_to_tile(inner_state, edge, dtype=out_dtype)
                ban = next(n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == bname)
                return "Tile", None, (ban, None, dace.Memlet(f"{bname}[{subset}]"))
            return "Tile", None, (edge.src, edge.src_conn, dace.Memlet.from_memlet(edge.data))

        kind_t, expr_t, wire_t = _plan_arm(t_arg, t_is_sym)
        kind_e, expr_e, wire_e = _plan_arm(e_arg, e_is_sym)
        # TileITE select-arm predicate wired via the unified ``_mask`` connector (user
        # 2026-06-12). Downstream global TileStore gates the iter-mask; no separate one.
        ite = TileITE(name=f"{tasklet.label}_ite",
                      widths=tuple(self.widths),
                      kind_t=kind_t,
                      kind_e=kind_e,
                      expr_t=expr_t,
                      expr_e=expr_e)
        inner_state.add_node(ite)
        # cond is always a connector (the upstream comparison result). A Scalar cond (from
        # an all-Symbol comparison like ``FLAG > 0``) is broadcast to a full bool tile
        # (design 7.5) — TileITE's pure expansion expects one.
        cond_edge = in_edges[cond_arg]
        if self._is_scalar_or_len1_source(inner_state, cond_edge):
            broadcast_name = self._broadcast_scalar_to_tile(inner_state, cond_edge, dtype=dace.bool_)
            from dace.sdfg.nodes import AccessNode
            existing = next((n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == broadcast_name),
                            None)
            cond_an = existing if existing is not None else inner_state.add_access(broadcast_name)
            inner_state.add_edge(cond_an, None, ite, "_mask", dace.Memlet(f"{broadcast_name}[{subset}]"))
        else:
            inner_state.add_edge(cond_edge.src, cond_edge.src_conn, ite, "_mask",
                                 dace.Memlet.from_memlet(cond_edge.data))
        # Wire the materialised arms (an inline Symbol arm carries no connector).
        if wire_t is not None:
            inner_state.add_edge(wire_t[0], wire_t[1], ite, "_t", wire_t[2])
        if wire_e is not None:
            inner_state.add_edge(wire_e[0], wire_e[1], ite, "_e", wire_e[2])
        if self._ensure_output_widened(inner_state, out_edge, ite):
            subset_str = ", ".join(f"0:{w}" for w in self.widths)
            _out_memlet = dace.Memlet(f"{out_edge.dst.data}[{subset_str}]")
        else:
            _out_memlet = dace.Memlet.from_memlet(out_edge.data)
        inner_state.add_edge(ite, "_o", out_edge.dst, out_edge.dst_conn, _out_memlet)
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _detect_conditional_write(self, tasklet: Tasklet) -> Optional[Tuple[str, str, str, bool]]:
        """If ``tasklet`` is a masked write ``_o = IT(cond, val)``, return
        ``(out_conn, cond_conn, val_arg, val_is_sym)``; else ``None``.

        ``IT(cond, val)`` is the write-only conditional write (``NormalizeMaskedWriteTasklets``
        rewrites the frontend's ``if cond: _o = val`` boolean-mask assignment): write ``val``
        where ``cond``, else leave the destination — NO old-value read. Lowers to a masked
        ``TileStore`` (``cond`` gates it). ``cond`` is always an in-connector (bool tile /
        scalar); ``val`` is an in-connector or an inline Symbol / literal (masked-const-write
        ``_o = IT(cond, 2.0)``). ``IT(`` is unambiguous: ``'ITE('`` does not start with it.
        """
        if len(tasklet.out_connectors) != 1:
            return None
        body = tasklet.code.as_string.strip().rstrip(";").strip()
        out_conn = next(iter(tasklet.out_connectors))
        if not body.startswith(f"{out_conn} = "):
            return None
        rhs = body[len(f"{out_conn} = "):].strip()
        if rhs.startswith("(") and rhs.endswith(")"):
            rhs = rhs[1:-1].strip()
        prefix = "IT("
        if not (rhs.startswith(prefix) and rhs.endswith(")")):
            return None
        parts = self._split_top_level_commas(rhs[len(prefix):-1], 2)
        if parts is None or len(parts) != 2:
            return None
        cond_arg, val_arg = (p.strip() for p in parts)
        in_conns = list(tasklet.in_connectors)
        if cond_arg not in in_conns:
            return None
        return out_conn, cond_arg, val_arg, val_arg not in in_conns

    def _convert_conditional_write(self, inner_state: SDFGState, tasklet: Tasklet, detected,
                                   iter_vars: Tuple[str, ...]) -> bool:
        """Lower ``_o = IT(cond, val)`` to a masked ``TileStore`` + a plain value copy.

        Steps: (1) find the downstream ``TileStore`` writing this tasklet's output tile;
        (2) resolve ``cond`` to a bool tile (broadcasting a scalar cond); (3) gate the
        store on ``cond`` -- first-wire when the store is unmasked (the divisible main
        map), else AND-combine with the tile iteration mask already on it (a masked
        remainder slab); (4) strip the ``IT`` wrapper, leaving ``_o = val``, and drop the
        now-unused ``cond`` input; (5) re-dispatch that plain assign / const-broadcast
        through the normal converter path. The value is computed for every lane (the
        masked store discards inactive lanes), so no compute op needs the mask.
        """
        out_conn, cond_conn, val_arg, _val_is_sym = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if cond_conn not in in_edges or not out_edges:
            return False
        out_edge = out_edges[0]
        store = self._find_downstream_store(inner_state, out_edge)
        if store is None:
            raise NotImplementedError(f"{tasklet.label}: masked write ``_o = IT(cond, val)`` whose output "
                                      f"{out_edge.dst!r} does not feed a single downstream ``TileStore._src``. The "
                                      f"masked-store lowering needs exactly one store to gate on ``cond``; this shape "
                                      f"(no store / fan-out to several stores) is not yet handled.")
        cond_edge = in_edges[cond_conn]
        cond_an = self._resolve_cond_tile(inner_state, cond_edge)
        self._apply_cond_mask_to_store(inner_state, store, cond_an)
        # Strip ``IT``: leave a plain ``_o = val`` copy / const, drop the cond input, and
        # re-dispatch so ``_convert_assign`` / ``_convert_const_assign`` lowers the value.
        inner_state.remove_edge(cond_edge)
        if cond_conn in tasklet.in_connectors:
            tasklet.remove_in_connector(cond_conn)
        tasklet.code = CodeBlock(f"{out_conn} = {val_arg}", language=dace.dtypes.Language.Python)
        return self._convert_one(inner_state, tasklet, iter_vars)

    def _find_downstream_store(self, inner_state: SDFGState, out_edge):
        """Return the single downstream ``TileStore`` fed (via its ``_src``) by this
        tasklet's output tile, or ``None`` when there is not exactly one."""
        from dace.sdfg.nodes import AccessNode
        dst = out_edge.dst
        if isinstance(dst, TileStore) and out_edge.dst_conn == "_src":
            return dst
        if not isinstance(dst, AccessNode):
            return None
        stores = [e.dst for e in inner_state.out_edges(dst) if isinstance(e.dst, TileStore) and e.dst_conn == "_src"]
        return stores[0] if len(stores) == 1 else None

    def _resolve_cond_tile(self, inner_state: SDFGState, cond_edge):
        """Resolve the masked-write condition to a ``widths``-shaped bool tile AccessNode.

        A per-element mask (``m_tile`` / an upstream comparison's bool tile) is used as-is;
        a scalar / length-1 cond is broadcast to a full bool tile (mirrors ``_convert_ite``'s
        cond handling) so the store's ``_mask`` is always the locked bool[widths] shape.
        """
        from dace.sdfg.nodes import AccessNode
        if self._is_scalar_or_len1_source(inner_state, cond_edge):
            bname = self._broadcast_scalar_to_tile(inner_state, cond_edge, dtype=dace.bool_)
            existing = next((n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == bname), None)
            return existing if existing is not None else inner_state.add_access(bname)
        return cond_edge.src

    def _apply_cond_mask_to_store(self, inner_state: SDFGState, store, cond_an) -> None:
        """Gate ``store`` on ``cond_an``. Flips ``has_mask`` on + adds the ``_mask`` connector
        when the store was unmasked (divisible main map), then wires ``cond_an`` -- first-wire,
        or AND-combined with an existing iteration mask (masked remainder) via ``_wire_mask``."""
        if not store.has_mask:
            store.has_mask = True
            store.add_in_connector("_mask")
        self._wire_mask(inner_state, store, cond_an)

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
        # Reference the source by its connector ABI — NO C-style cast (user 2026-06-15).
        # ABI follows the memlet's element COUNT, not the descriptor kind: a single-element
        # read (``Scalar`` or length-1 ``Array[0]``) is a by-value ``T _in``, referenced
        # bare; a multi-element source is a pointer ``T* _in``, element 0 via ``_in[0]``.
        # The destination tile's element type drives the implicit conversion.
        src_ref = "_in" if src_edge.data.subset.num_elements() == 1 else "_in[0]"
        code_lines = []
        for d in range(K):
            # constexpr width + DACE_UNROLL → lane loop lowers to SIMD.
            code_lines.append(f"{'    ' * d}constexpr std::size_t __W{d} = {widths[d]};")
            code_lines.append(f"{'    ' * d}DACE_UNROLL")
            code_lines.append(f"{'    ' * d}for (std::size_t __l{d} = 0; __l{d} < __W{d}; ++__l{d}) {{")
        code_lines.append(f"{'    ' * K}_out[{flat}] = {src_ref};")
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
        """Materialise a Symbol / literal expr as a FULL-TILE transient (design 7.5):
        lane-id-dependent (references an iter_var) → :meth:`_materialise_lane_id_tile`
        (int64); loop-invariant → OUTPUT-edge-dtype transient + constant-fill broadcast.
        """
        if self._is_lane_id_dependent(expr, iter_vars):
            an_name = self._materialise_lane_id_tile(inner_state, expr, iter_vars)
        else:
            an_name = self._materialise_invariant_to_tile(inner_state, expr, out_edge)
        from dace.sdfg.nodes import AccessNode
        existing = next((n for n in inner_state.nodes() if isinstance(n, AccessNode) and n.data == an_name), None)
        return existing if existing is not None else inner_state.add_access(an_name)

    def _materialise_invariant_to_tile(self, inner_state: SDFGState, expr: str, out_edge) -> str:
        """Mint a FULL-TILE transient = ``expr`` broadcast across every lane. Dtype matches
        the OUTPUT edge (so the TileITE's _t / _e operand dtype matches _o).
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
            # constexpr width + DACE_UNROLL → lane loop lowers to SIMD.
            code_lines.append(f"{'    ' * d}constexpr std::size_t __W{d} = {widths[d]};")
            code_lines.append(f"{'    ' * d}DACE_UNROLL")
            code_lines.append(f"{'    ' * d}for (std::size_t __l{d} = 0; __l{d} < __W{d}; ++__l{d}) {{")
        # No C-style cast (user 2026-06-15): the destination tile's element type drives
        # the implicit conversion of the broadcast literal / symbolic expression.
        code_lines.append(f"{'    ' * K}_out[{flat}] = ({expr});")
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
        # A two-tile power ``a ** b`` has a per-lane (runtime) exponent -- not a provable
        # integer -- so it lowers to ``std::pow``. (``pow(a, b)`` call form likewise.)
        if op in ("**", "pow"):
            op = "pow"
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if a_conn not in in_edges or b_conn not in in_edges or not out_edges:
            return False
        out_edge = out_edges[0]
        a_edge = in_edges[a_conn]
        b_edge = in_edges[b_conn]
        # Mixed-dtype operands NOT supported (user 2026-06-10): the walker-primary pipeline
        # locks one dtype per lib node (tile transient + bridge + downstream copy all assume
        # it). Refuse → NotImplementedError so callers add explicit casts.
        #
        # Comparison ops (``< <= > >= == !=``) are the exception: result dtype is ``bool``
        # regardless of operand dtype, so enforce uniformity on the OPERANDS only and let
        # the output be ``bool``.
        sdfg = inner_state.sdfg
        a_dtype = sdfg.arrays[a_edge.data.data].dtype if a_edge.data and a_edge.data.data else None
        b_dtype = sdfg.arrays[b_edge.data.data].dtype if b_edge.data and b_edge.data.data else None
        c_dtype = sdfg.arrays[out_edge.data.data].dtype if out_edge.data and out_edge.data.data else None
        if op in _COMPARISON_BINOPS:
            checked_dtypes = {d for d in (a_dtype, b_dtype) if d is not None}
        else:
            checked_dtypes = {d for d in (a_dtype, b_dtype, c_dtype) if d is not None}
        if len(checked_dtypes) > 1:
            raise NotImplementedError(
                f"vec(K-dim): mixed-dtype binop NOT supported. Tasklet {tasklet.label!r} body "
                f"{tasklet.code.as_string!r} mixes dtypes {checked_dtypes}. Per design 6.2 + user "
                f"direction the walker-primary path locks a single dtype per lib node. Rewrite the "
                f"kernel with an explicit cast tasklet upstream OR widen the destination dtype.")
        # Operand kind from the source descriptor: Scalar / length-1 Array → Scalar
        # broadcast operand (design 6.5).
        kind_a = self._operand_kind(inner_state, a_edge)
        kind_b = self._operand_kind(inner_state, b_edge)
        # Output transient shape is pre-set by WidenAccesses (design 6.2); the lib-node
        # output kind is implied by ``out_edge``'s destination descriptor (validate()
        # enforces consistency).
        # Mask-when-partial (user 2026-06-12): when an iter_mask is in scope (remainder /
        # cond-mask region), inactive lanes hold garbage that can trap (div-by-0,
        # log-of-neg) or propagate NaN — mask the op so they skip the compute. The
        # divisible main map (no mask AN) stays unmasked (fast path).
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
        _was_widened = self._ensure_output_widened(inner_state, out_edge, binop)

        if _was_widened:

            _subset_str = ", ".join(f"0:{w}" for w in self.widths)

            _out_memlet = dace.Memlet(f"{out_edge.dst.data}[{_subset_str}]")

        else:

            _out_memlet = dace.Memlet.from_memlet(out_edge.data)

        inner_state.add_edge(binop, "_c", out_edge.dst, out_edge.dst_conn, _out_memlet)
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _classify_power_op(self, inner_state: SDFGState, op: str, symbol_side: str, exponent_expr: str,
                           base_edge) -> str:
        """Resolve a power operator to its concrete tile op from the base dtype and exponent.

        ``**`` (and the ``pow`` call form) are kept verbatim in the tasklet body; the tile
        emitter decides ``pow`` vs ``ipow`` HERE. ``ipow`` (exact repeated-multiply
        ``dace::math::ipow``) is chosen iff BOTH:

        * the base is an integer tile (side ``a``) -- NumPy raises a float base to a power with
          libm ``pow``, so ``ipow`` there is not bit-exact (it diverges in the low bits, badly
          for a large exponent); only an integer base uses repeated multiply, which ``ipow`` is;
        * the exponent (side ``b``) is a provable non-negative integer, via
          :func:`~dace.transformation.passes.relax_integer_powers.exponent_relaxes_to_ipow` --
          the SAME proof the pow->ipow relaxation applies to size powers, so both agree.

        Otherwise (float base, runtime/tile exponent, or an unprovable exponent) -> ``pow``
        (``std::pow``). A non-power ``op`` is returned unchanged.

        :param inner_state: the state holding the tasklet (its SDFG roots the assumption scope).
        :param op: the detected operator (``"**"`` / ``"pow"`` for a power, else unchanged).
        :param symbol_side: which operand is the inline Symbol -- ``"b"`` = exponent (the
            relaxable case), ``"a"`` = base (so the tile operand is the exponent -> ``pow``).
        :param exponent_expr: the exponent's source text (a Symbol / literal, never a connector).
        :param base_edge: the in-edge feeding the tile (base) operand, for its dtype.
        :returns: ``"ipow"`` / ``"pow"`` for a power, else ``op``.
        """
        if op not in ("**", "pow"):
            return op
        # The tile operand must be the BASE (symbol on side b = exponent); otherwise the
        # exponent is the runtime tile -> std::pow.
        if symbol_side != "b":
            return "pow"
        base_dtype = (inner_state.sdfg.arrays[base_edge.data.data].dtype
                      if base_edge.data is not None and base_edge.data.data is not None else None)
        if base_dtype is None or not np.issubdtype(base_dtype.type, np.integer):
            return "pow"  # float base -> libm pow (NumPy semantics); ipow would not be bit-exact
        from dace import symbolic
        from dace.transformation.passes.relax_integer_powers import exponent_relaxes_to_ipow
        root = inner_state.sdfg
        while root.parent_sdfg is not None:
            root = root.parent_sdfg
        try:
            exponent = symbolic.pystr_to_symbolic(exponent_expr)
        except Exception:  # noqa: BLE001 -- unparseable exponent: fall back to std::pow
            return "pow"
        return "ipow" if exponent_relaxes_to_ipow(exponent, root) else "pow"

    def _convert_binop_with_symbol(self, inner_state: SDFGState, tasklet: Tasklet, detected,
                                   iter_vars: Tuple[str, ...]) -> bool:
        """Emit a TileBinop whose second operand is a Symbol expr: loop-invariant →
        ``kind=Symbol`` + ``expr_*`` (no connector, broadcast at expansion);
        lane-id-dependent → per-lane tile (:meth:`_materialise_lane_id_tile`), ``kind=Tile``.
        """
        out_conn, a_conn, op, symbol_side, symbol_expr = detected
        in_edges = {e.dst_conn: e for e in inner_state.in_edges(tasklet)}
        out_edges = list(inner_state.out_edges(tasklet))
        if a_conn not in in_edges or not out_edges:
            return False
        out_edge = out_edges[0]
        a_edge = in_edges[a_conn]
        # A power keeps its ``**`` (or ``pow``) form until here; classify it per operand from
        # the base dtype + exponent -- integer base & integer exponent -> ``ipow`` (exact
        # repeated multiply), else ``pow`` (``std::pow``).
        op = self._classify_power_op(inner_state, op, symbol_side, symbol_expr, a_edge)
        kind_tile_side = self._operand_kind(inner_state, a_edge)
        sym_kind, sym_expr, sym_an_name = self._resolve_symbol_operand(inner_state, symbol_expr, iter_vars)
        # Mask-when-partial.
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
        _was_widened = self._ensure_output_widened(inner_state, out_edge, binop)

        if _was_widened:

            _subset_str = ", ".join(f"0:{w}" for w in self.widths)

            _out_memlet = dace.Memlet(f"{out_edge.dst.data}[{_subset_str}]")

        else:

            _out_memlet = dace.Memlet.from_memlet(out_edge.data)

        inner_state.add_edge(binop, "_c", out_edge.dst, out_edge.dst_conn, _out_memlet)
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

    def _ensure_output_widened(self, inner_state: SDFGState, out_edge, lib_node=None) -> bool:
        """Widen the destination transient + memlets to ``(W_0, ..., W_{K-1})``.

        Only fires when the destination is a WIDENABLE TRANSIENT (Scalar or
        length-1 Array) AND at least one input to ``lib_node`` is a
        tile-shape ``(W,)`` array. Per user direction: all-Scalar /
        Scalar-Symbol / Symbol-Symbol op -> Scalar output stays Scalar.
        """
        from dace import data as _dd
        from dace import subsets as _subsets
        from dace.sdfg.nodes import AccessNode
        if not isinstance(out_edge.dst, AccessNode):
            return False
        sdfg = inner_state.sdfg
        desc = sdfg.arrays.get(out_edge.dst.data)
        if desc is None or not desc.transient:
            return False
        widths = tuple(self.widths)
        if lib_node is not None:
            any_tile_in = False
            for e in inner_state.in_edges(lib_node):
                if not isinstance(e.src, AccessNode):
                    continue
                src_desc = sdfg.arrays.get(e.src.data)
                if src_desc is None or isinstance(src_desc, _dd.Scalar):
                    continue
                if isinstance(src_desc, _dd.Array) and tuple(src_desc.shape) == widths:
                    any_tile_in = True
                    break
            if not any_tile_in:
                return False
        is_widenable = False
        if isinstance(desc, _dd.Scalar):
            is_widenable = True
        elif isinstance(desc, _dd.Array):
            shape = tuple(desc.shape)
            if shape:
                if tuple(shape) == widths:
                    return True
                try:
                    is_widenable = all(bool(dace.symbolic.simplify(s - 1) == 0) for s in shape)
                except Exception:  # noqa: BLE001
                    is_widenable = False
        if not is_widenable:
            return False
        new_desc = _dd.Array(dtype=desc.dtype,
                             shape=widths,
                             transient=True,
                             storage=getattr(desc, "storage", dace.dtypes.StorageType.Register))
        sdfg.arrays[out_edge.dst.data] = new_desc
        target_subset = ", ".join(f"0:{w}" for w in widths)
        target_range = _subsets.Range.from_string(target_subset)
        for state in sdfg.states():
            for edge in state.edges():
                if edge.data is None or edge.data.data != out_edge.dst.data:
                    continue
                new_sub = _subsets.Range(list(target_range.ranges))
                edge.data.subset = new_sub
                edge.data.volume = new_sub.num_elements()
                # Widen ``other_subset`` symmetrically (user 2026-06-12) so the AN→AN
                # bridge ``a[0:W] → b[0:W]`` is well-formed.
                if edge.data.other_subset is not None:
                    edge.data.other_subset = _subsets.Range(list(target_range.ranges))
        return True

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
        # Mask-when-partial.
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
        _was_widened = self._ensure_output_widened(inner_state, out_edge, unop)

        if _was_widened:

            _subset_str = ", ".join(f"0:{w}" for w in self.widths)

            _out_memlet = dace.Memlet(f"{out_edge.dst.data}[{_subset_str}]")

        else:

            _out_memlet = dace.Memlet.from_memlet(out_edge.data)

        inner_state.add_edge(unop, "_c", out_edge.dst, out_edge.dst_conn, _out_memlet)
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
        # Mask-when-partial.
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
        _was_widened = self._ensure_output_widened(inner_state, out_edge, binop)

        if _was_widened:

            _subset_str = ", ".join(f"0:{w}" for w in self.widths)

            _out_memlet = dace.Memlet(f"{out_edge.dst.data}[{_subset_str}]")

        else:

            _out_memlet = dace.Memlet.from_memlet(out_edge.data)

        inner_state.add_edge(binop, "_c", out_edge.dst, out_edge.dst_conn, _out_memlet)
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
        # Mask-when-partial.
        mask_an = self._find_mask_an(inner_state)
        unop = TileUnop(name=f"{tasklet.label}_unop",
                        widths=tuple(self.widths),
                        op=op,
                        kind_a=kind_a,
                        has_mask=mask_an is not None)
        inner_state.add_node(unop)
        self._wire_mask(inner_state, unop, mask_an)
        inner_state.add_edge(a_edge.src, a_edge.src_conn, unop, "_a", dace.Memlet.from_memlet(a_edge.data))
        _was_widened = self._ensure_output_widened(inner_state, out_edge, unop)

        if _was_widened:

            _subset_str = ", ".join(f"0:{w}" for w in self.widths)

            _out_memlet = dace.Memlet(f"{out_edge.dst.data}[{_subset_str}]")

        else:

            _out_memlet = dace.Memlet.from_memlet(out_edge.data)

        inner_state.add_edge(unop, "_c", out_edge.dst, out_edge.dst_conn, _out_memlet)
        for edge in list(in_edges.values()) + out_edges:
            inner_state.remove_edge(edge)
        inner_state.remove_node(tasklet)
        return True

    def _convert_inner(self, inner_sdfg: SDFG, iter_vars: Tuple[str, ...]) -> int:
        """Walk every state of ``inner_sdfg`` and convert recognised tasklets.

        :returns: Number of conversions performed.
        """
        converted = 0
        # Two phases so ITE arm sources are already full tiles when the ITE is planned. An
        # ITE arm produced by a tile-producing op (e.g. ``_then_b = 0.0`` const-assign that
        # ``_convert_const_assign`` widens to ``(W,)``) must wire to the TileITE directly.
        # Planning the ITE FIRST would see the un-widened ``(1,)`` scalar and route it
        # through ``_broadcast_scalar_to_tile``; once the producer widens, that broadcast
        # reads a tile pointer as a scalar (malformed code).
        #
        # Spans ALL states: merge branch lowering puts producer (``compute_then``) and
        # consumer ITE (``apply_ITE``) in SEPARATE states, not necessarily visited
        # producer-first. Phase 1 lowers every non-ITE tasklet, phase 2 the ITEs — by
        # which point every arm source has its final tile shape.
        for want_ite in (False, True):
            for inner_state in inner_sdfg.states():
                for node in [n for n in inner_state.nodes() if isinstance(n, Tasklet)]:
                    if (self._detect_ite(node) is not None) != want_ite:
                        continue
                    if node not in inner_state.nodes():
                        continue  # already removed/replaced by an earlier conversion
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
        # Post-conditions.
        assert_invariant(no_memlet_dim_mismatch(sdfg), "ConvertTaskletsToTileOps",
                         "memlet subset and other_subset have matching dimensionality")
        assert_invariant(no_duplicate_connector_edges(sdfg), "ConvertTaskletsToTileOps",
                         "no duplicate connector edges on lib nodes")
        assert_invariant(mask_connectors_are_bool(sdfg), "ConvertTaskletsToTileOps",
                         "every tile-op _mask connector is fed by a bool array")
        assert_invariant(logical_binops_are_bool(sdfg), "ConvertTaskletsToTileOps",
                         "every logical (&& / ||) TileBinop has bool inputs and bool output")
        return total or None
