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
from dace.libraries.tileops import TileBinop, TileGather, TileLoad, TileReduce, TileScatter, TileStore, TileUnop
from dace.libraries.tileops._pure_codegen import nested_loops, tile_offset
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER, TILE_MAIN_MARKER)
from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
from dace.transformation.passes.vectorization.utils.name_schemes import TileConnectors
from dace.transformation.passes.vectorization.utils.tile_dims import (
    TileAccessKind,
    TileDimSpec,
    classify_tile_access,
)


def _lane_index_expr(begin_str: str, iter_vars: Tuple[str, ...]) -> Optional[str]:
    """Per-lane gather/scatter index for one source array dim.

    Substitutes each tile var ``v_p`` with ``v_p + __l_p`` in the begin
    expression, so the index at lane ``(l_0, …, l_{K-1})`` is
    ``begin(v_p -> v_p + __l_p)``. This handles every deterministic index
    uniformly: affine (``cc[j,i]`` -> ``i + __l0``; diagonal ``a[i,i]``),
    non-unit coefficient (``a[2*i]``), and STRUCTURED (``int_floor(i, 2)`` ->
    ``int_floor(i + __l0, 2)``). The lane loop vars match the nested-loop
    emitter's ``__l0 .. __l{K-1}``. A data-dependent (indirect) index never
    reaches here — it carries a loaded symbol, not a tile var, so the
    classifier does not route it through the affine gather map.

    :param begin_str: The dim's per-iteration begin expression.
    :param iter_vars: Tile iter-var names, innermost-last.
    :returns: The C++ index expression, or ``None`` if ``begin_str`` cannot
        be parsed.
    """
    try:
        b = symbolic.pystr_to_symbolic(begin_str)
    except Exception:  # noqa: BLE001 - unparseable begin -> caller skips
        return None
    subs = {}
    for p, v in enumerate(iter_vars):
        vsym = symbolic.pystr_to_symbolic(v)
        if vsym in b.free_symbols:
            subs[vsym] = vsym + symbolic.pystr_to_symbolic(f"__l{p}")
    if subs:
        b = b.subs(subs)
    return symbolic.symstr(b)


_OPERAND = r"(?:[A-Za-z_]\w*|\d+\.?\d*(?:[eE][+-]?\d+)?)"
_BINOP_RE = re.compile(rf"^\s*(?P<out>\w+)\s*=\s*"
                       rf"(?:\(\s*)?(?P<a>{_OPERAND})\s*"
                       rf"(?P<op>\+|-|\*|/|%|<=|>=|==|!=|<|>|and|or|\^|&|\|)\s*"
                       rf"(?P<b>{_OPERAND})\s*\)?\s*;?\s*$")

# Function-call binops ``out = max(a, b)`` / ``out = min(a, b)`` — two-operand
# ops with no infix spelling. They map to the same TileBinop the infix ops use
# (``max`` -> 'M', ``min`` -> 'm' in the backend headers), so they are binops,
# not unops: capture both operands and the function name.
_FUNC_BINOP_RE = re.compile(rf"^\s*(?P<out>\w+)\s*=\s*(?P<op>max|min)\s*\(\s*"
                            rf"(?P<a>{_OPERAND})\s*,\s*(?P<b>{_OPERAND})\s*\)\s*;?\s*$")

# A numeric literal RHS, allowing one layer of parens and a space after a
# unary minus (the frontend emits ``A[i, j] = -999`` as ``(- 999)``). Matched
# before the unop classifier so a negative constant stays a constant store
# (a TileUnop(neg, literal) would be a wasteful round-trip and mis-handled).
_CONST_STORE_RE = re.compile(r"^\s*(?P<out>\w+)\s*=\s*\(?\s*(?P<val>-?\s*\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\)?\s*;?\s*$")


def _constant_store_value(tasklet: dace.nodes.Tasklet) -> Optional[str]:
    """Return the literal a constant-store tasklet writes, or ``None``.

    A no-binop body ``out = <numeric literal>`` (e.g. ``aa[j, i] = 0.0`` or
    ``A[i, j] = -999`` -> ``(- 999)``) is a broadcast store: every lane gets the
    same constant. Recognised so it can lower to a constant-fill tile +
    ``TileStore`` rather than be mis-parsed as a binop / unary-minus.

    :param tasklet: Tasklet to inspect.
    :returns: The literal string with internal spaces removed (e.g. ``"0.0"``,
        ``"-999"``), or ``None``.
    """
    body = tasklet.code.as_string.strip().rstrip(";").strip()
    m = _CONST_STORE_RE.match(body)
    return m.group("val").replace(" ", "") if m else None


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
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "%": "%",
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
    "==": "==",
    "!=": "!=",
    "and": "&&",
    "or": "||",
    "&": "&",
    "|": "|",
    "^": "^",
}
_ASSIGN_RE = re.compile(r"^\s*(?P<out>\w+)\s*=\s*(?P<inp>\w+)\s*;?\s*$")

#: Map a WCR lambda body (the ``cpp.unparse_cr_split`` output) to one of the
#: reduction operators :class:`TileReduce` understands. Lambdas come in
#: well-known shapes from the DaCe frontend: ``lambda x, y: (x + y)`` /
#: ``lambda x, y: max(x, y)`` etc. We pattern-match the operator token in
#: the body so identity / per-arch lowering is consistent across reductions.
_WCR_OP_PATTERNS = (
    ("max(", "max"),
    ("min(", "min"),
    (" + ", "+"),
    (" * ", "*"),
    (" and ", "&&"),
    (" or ", "||"),
)


def _wcr_op(wcr_code: Optional[str]) -> Optional[str]:
    """Extract the reduction operator from a WCR lambda body string.

    :param wcr_code: The WCR lambda's source (e.g. ``lambda x, y: (x + y)``).
    :returns: One of ``+ * min max && ||``, or ``None`` if not recognised.
    """
    if not wcr_code:
        return None
    for token, op in _WCR_OP_PATTERNS:
        if token in wcr_code:
            return op
    return None


def _is_assign_tasklet(tasklet: dace.nodes.Tasklet) -> bool:
    """Return True iff the tasklet body is a trivial connector-to-connector
    ``out = inp`` copy.

    The frontend emits these at the end of multi-tasklet chains (post-
    ``SplitTasklets``) to route an intermediate transient to the outer
    access node; they are pass-throughs for the tile-op rewrite. The RHS must
    be an actual input connector: ``out = 0`` (a constant store, handled by
    :func:`_constant_store_value`) and ``out = N`` (a symbol read) both match
    the bare ``\\w+`` RHS but are NOT copies and have no input edge.

    :param tasklet: Tasklet to inspect.
    :returns: True iff the body is ``out = inp`` with ``out`` an output
        connector and ``inp`` an input connector.
    """
    body = tasklet.code.as_string.strip().rstrip(";").strip()
    m = _ASSIGN_RE.match(body)
    return (m is not None and m.group("out") in tasklet.out_connectors and m.group("inp") in tasklet.in_connectors)


def _classify_binop_tasklet_body(tasklet: dace.nodes.Tasklet) -> Optional[Tuple[str, str, str, str]]:
    """Parse a split binop tasklet ``out = lhs OP rhs``, or ``None``.

    Non-raising counterpart of :meth:`EmitTileOps._classify_binop_tasklet`,
    so callers can *filter* binop tasklets without try/except.

    :param tasklet: Compute tasklet to inspect.
    :returns: ``(out_conn, lhs_conn, op, rhs_conn)`` with ``op`` mapped to
        its TileBinop spelling, or ``None`` if the body is not a binop or
        the output is not a declared out-connector.
    """
    body = tasklet.code.as_string.strip().rstrip(";").strip()
    m = _BINOP_RE.match(body)
    if m is not None:
        out = m.group("out")
        if out not in tasklet.out_connectors:
            return None
        return out, m.group("a"), _PY_TO_TILEBINOP_OP[m.group("op")], m.group("b")
    # Function-form two-operand op (``max`` / ``min``): same TileBinop spelling.
    fm = _FUNC_BINOP_RE.match(body)
    if fm is not None:
        out = fm.group("out")
        if out not in tasklet.out_connectors:
            return None
        return out, fm.group("a"), fm.group("op"), fm.group("b")
    return None


# Unary function name (as emitted by the frontend / math-lowering) -> the
# TileUnop op spelling. The dace-mangled ``dace_<fn>_d`` / ``dace_<fn>_f``
# math shims map to the same op as the bare name.
_FUNC_UNOP = {
    "abs": "abs",
    "fabs": "abs",
    "exp": "exp",
    "dace_exp_d": "exp",
    "dace_exp_f": "exp",
    "log": "log",
    "dace_log_d": "log",
    "dace_log_f": "log",
    "sqrt": "sqrt",
    "dace_sqrt_d": "sqrt",
    "dace_sqrt_f": "sqrt",
    "sin": "sin",
    "cos": "cos",
    "floor": "floor",
    "ceil": "ceil",
    "tanh": "tanh",
}
_FUNC_UNOP_RE = re.compile(rf"^\s*(?P<out>\w+)\s*=\s*(?P<fn>\w+)\s*\(\s*(?P<a>{_OPERAND})\s*\)\s*;?\s*$")
_NEG_RE = re.compile(rf"^\s*(?P<out>\w+)\s*=\s*\(?\s*-\s*(?P<a>{_OPERAND})\s*\)?\s*;?\s*$")
#: Logical NOT — Python ``not a`` or C ``! a`` (post-SplitTasklets / branch
#: normalisation often emits both spellings depending on the source pass).
_NOT_RE = re.compile(rf"^\s*(?P<out>\w+)\s*=\s*\(?\s*(?:not\s+|!\s*)(?P<a>{_OPERAND})\s*\)?\s*;?\s*$")


def _classify_unop_tasklet_body(tasklet: dace.nodes.Tasklet) -> Optional[Tuple[str, str, str]]:
    """Parse a single-operand unary tasklet ``out = <op> a``, or ``None``.

    Recognises the function forms ``out = abs(a)`` / ``exp(a)`` / ``log(a)`` /
    ``sqrt(a)`` / ``sin``/``cos``/``floor``/``ceil``/``tanh`` (and the
    dace-mangled ``dace_<fn>_d`` math shims) plus unary minus ``out = -a``.

    :param tasklet: Compute tasklet to inspect.
    :returns: ``(out_conn, op, operand_token)`` with ``op`` a :class:`TileUnop`
        spelling, or ``None`` if the body is not a recognised unop or the
        output is not a declared out-connector.
    """
    body = tasklet.code.as_string.strip().rstrip(";").strip()
    fm = _FUNC_UNOP_RE.match(body)
    if fm is not None and fm.group("fn") in _FUNC_UNOP:
        out = fm.group("out")
        if out not in tasklet.out_connectors:
            return None
        return out, _FUNC_UNOP[fm.group("fn")], fm.group("a")
    nm = _NEG_RE.match(body)
    if nm is not None:
        out = nm.group("out")
        if out not in tasklet.out_connectors:
            return None
        return out, "neg", nm.group("a")
    nt = _NOT_RE.match(body)
    if nt is not None:
        out = nt.group("out")
        if out not in tasklet.out_connectors:
            return None
        return out, "not", nt.group("a")
    return None


def _normalize_operand_kind(kind: str) -> str:
    """Collapse a resolved operand kind to a :class:`TileBinop` / :class:`TileUnop`
    ``kind_*`` property value.

    A gathered or already-materialised intermediate tile is just a ``Tile``
    operand to the lib node; ``Scalar`` / ``Symbol`` pass through unchanged.

    :param kind: Resolver kind (``Tile`` / ``Tile-existing`` / ``Gather`` /
        ``Scalar`` / ``Symbol``).
    :returns: ``Tile`` for tile-shaped operands, else ``kind`` unchanged.
    """
    return "Tile" if kind in ("Tile-existing", "Gather") else kind


def _tile_region_subset(orig_subset: subsets.Range, iter_vars: Tuple[str, ...], widths: Tuple[int,
                                                                                              ...]) -> subsets.Range:
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


def _mask_name_for_map(state: dace.SDFGState, map_entry: MapEntry) -> Optional[str]:
    """Return the iteration-mask transient produced inside ``map_entry``'s
    scope, or ``None`` for an unmasked tile map.

    Each map in a state has its own per-map mask (``_tile_iter_mask``,
    ``_tile_iter_mask_1``, ...), so the name is read from THIS map's scope. A
    map with no :class:`TileMaskGen` in scope is the provably-divisible interior
    of a ``masked_tail`` split — every tile is fully in bounds, so ``None``
    routes emission / descent to the ``has_mask=False`` fast path.

    :param state: Parent state.
    :param map_entry: Inner map entry.
    :returns: The ``TileMaskGen`` output array name, or ``None`` if the scope
        has no mask producer.
    """
    from dace.libraries.tileops import TileMaskGen
    scope = state.all_nodes_between(map_entry, state.exit_node(map_entry)) or set()
    for node in scope:
        if isinstance(node, TileMaskGen):
            oe = [e for e in state.out_edges(node) if e.src_conn == TileConnectors.O]
            if oe:
                return oe[0].data.data
    return None


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

    def __init__(self, widths: Tuple[int, ...] = (8, )):
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

    def _find_body_tasklets(self, state: dace.SDFGState, map_entry: MapEntry) -> List[dace.nodes.Tasklet]:
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
        parsed = _classify_binop_tasklet_body(tasklet)
        if parsed is None:
            body = tasklet.code.as_string.strip().rstrip(";").strip()
            raise NotImplementedError(f"EmitTileOps: tasklet {tasklet.label!r} body {body!r} is not 'out = lhs OP rhs' "
                                      f"with the output in out_connectors; run SplitTasklets first.")
        return parsed

    def _operand_kind(self, state: dace.SDFGState, tasklet: dace.nodes.Tasklet, conn_name: str,
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
            raise NotImplementedError(f"EmitTileOps: tasklet {tasklet.label!r} input {conn_name!r} has "
                                      f"{len(in_e)} in-edges; expected exactly 1")
        edge = in_e[0]
        if edge.data is None or edge.data.data is None:
            raise NotImplementedError(f"EmitTileOps: tasklet {tasklet.label!r} input {conn_name!r} has no memlet")
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
        if cls.kind in (TileAccessKind.GATHER, TileAccessKind.STRUCTURED):
            # Non-perfect-box (diagonal ``a[i,i]``) or structured-replication
            # (``a[i//2]`` -> ``int_floor``) read: lower to a TileGather over a
            # per-dim index map ("gather map"), built by substituting the lane
            # offsets into each dim's begin (``_emit_gather_load`` ->
            # ``_lane_index_expr``). Both index forms are deterministic.
            return "Gather", (source_edge, edge, edge.data.subset)
        raise NotImplementedError(
            f"EmitTileOps: tasklet {tasklet.label!r} input {conn_name!r} access "
            f"{source_edge.data!r} is {cls.kind.value}; T5 MVP only handles Tile / Scalar / Symbol")

    def _add_tile_transient(self, sdfg: dace.SDFG, base: str, dtype: dace.dtypes.typeclass, widths: Tuple[int,
                                                                                                          ...]) -> str:
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

    def _emit_tile_load(self, state: dace.SDFGState, tasklet: dace.nodes.Tasklet, conn: str, source_edge, in_edge,
                        per_iter_subset: subsets.Range, dim_strides: Tuple[int, ...], src_dims: Tuple[int, ...],
                        spec: TileDimSpec, mask_name: str) -> str:
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
        # ``source_edge`` is the outermost edge in the memlet path; when an
        # inner tasklet shares a MapEntry connector pair with a
        # dependency-only in-edge (``Memlet(None)``), the outer connector edge
        # can lose its ``data`` and ``source_edge.data.data`` becomes ``None``.
        # Fall back to ``in_edge.data.data`` (the inner memlet) which still
        # carries the source data name.
        src_name = source_edge.data.data if source_edge.data.data is not None else in_edge.data.data
        sdfg = state.sdfg
        src_arr = sdfg.arrays[src_name]
        tile_name = self._add_tile_transient(
            sdfg,
            f"_tile_{conn.lstrip('_')}",
            src_arr.dtype,
            spec.widths,
        )
        load = TileLoad(
            name=f"{tasklet.label}_load_{conn.lstrip('_')}",
            widths=spec.widths,
            dim_strides=tuple(dim_strides),
            src_dims=tuple(src_dims),
            has_mask=mask_name is not None,
        )
        state.add_node(load)
        promoted_subset = _tile_region_subset(per_iter_subset, spec.iter_vars, spec.widths)
        state.add_edge(in_edge.src, in_edge.src_conn, load, TileConnectors.SRC,
                       dace.Memlet(data=src_name, subset=promoted_subset))
        subset = ", ".join(f"0:{w}" for w in spec.widths)
        mask_access = self._mask_access_or_none(state, mask_name)
        if mask_access is not None:
            state.add_edge(mask_access, None, load, TileConnectors.MASK, dace.Memlet(f"{mask_name}[{subset}]"))
        tile_access = state.add_access(tile_name)
        state.add_edge(load, TileConnectors.DST, tile_access, None, dace.Memlet(f"{tile_name}[{subset}]"))
        return tile_name, tile_access

    def _emit_gather_index_tiles(self, state: dace.SDFGState, map_entry: MapEntry, per_iter_subset: subsets.Range,
                                 src_ndim: int, spec: TileDimSpec) -> List[dace.nodes.AccessNode]:
        """Build one affine per-dim index tile (the "gather map") per source
        array dim. Each ``_gidx_k`` tile is filled by a small CPP tasklet
        ``_gidx_k[lane] = begin_k + sum_p coeff_p * __l_p`` (begin_k read from
        ``per_iter_subset``); the tasklet hangs off the map entry so the tile
        iter-vars are in scope.

        :param state: Parent state.
        :param map_entry: Inner (strided) map entry — the dependency source.
        :param per_iter_subset: Per-iteration subset of the gathered access.
        :param src_ndim: Number of source-array dims (one index tile each).
        :param spec: Surrounding tile spec.
        :returns: One produced index-tile AccessNode per source dim.
        :raises NotImplementedError: If any dim's index is non-affine
            (indirect) — the affine gather map cannot be built.
        """
        sdfg = state.sdfg
        out_subset = ", ".join(f"0:{w}" for w in spec.widths)
        idx_accesses: List[dace.nodes.AccessNode] = []
        for k in range(src_ndim):
            begin_str = str(per_iter_subset.ranges[k][0])
            expr = _lane_index_expr(begin_str, spec.iter_vars)
            if expr is None:
                raise NotImplementedError(f"EmitTileOps: gather index dim {k} ({begin_str!r}) is not affine "
                                          f"in the tile vars (indirect gather not yet emitted)")
            idx_name = self._add_tile_transient(sdfg, "_gidx", dace.int64, spec.widths)
            # Use a generic ``_out`` connector (not the array name) — a tasklet
            # connector must not collide with an SDFG data/symbol name.
            body = f"_out[{tile_offset(spec.widths)}] = {expr};"
            fill = state.add_tasklet(
                name=f"gidx_{idx_name}",
                inputs=set(),
                outputs={"_out"},
                code=nested_loops(spec.widths, body),
                language=dace.dtypes.Language.CPP,
            )
            # Keep the fill tasklet inside the map scope so the tile iter-vars
            # are defined symbols (mirrors the TileMaskGen wiring).
            state.add_nedge(map_entry, fill, dace.Memlet())
            idx_acc = state.add_access(idx_name)
            state.add_edge(fill, "_out", idx_acc, None, dace.Memlet(f"{idx_name}[{out_subset}]"))
            idx_accesses.append(idx_acc)
        return idx_accesses

    def _emit_gather_load(self, state: dace.SDFGState, tasklet: dace.nodes.Tasklet, conn: str, source_edge, in_edge,
                          per_iter_subset: subsets.Range, spec: TileDimSpec,
                          mask_name: str) -> Tuple[str, dace.nodes.AccessNode]:
        """Insert a :class:`TileGather` materializing one tile transient from a
        non-box (diagonal / affine-indirect) read.

        :returns: ``(tile_name, tile_access)`` of the gathered tile.
        """
        # Mirrors the fallback in :meth:`_emit_tile_load`: if the outer
        # connector edge lost its data (dep-edge MapEntry-connector sharing),
        # the inner connector edge still carries the source name.
        src_name = source_edge.data.data if source_edge.data.data is not None else in_edge.data.data
        sdfg = state.sdfg
        src_arr = sdfg.arrays[src_name]
        src_ndim = len(src_arr.shape)
        map_entry = self._map_entry_of(state, tasklet)
        idx_accesses = self._emit_gather_index_tiles(state, map_entry, per_iter_subset, src_ndim, spec)
        tile_name = self._add_tile_transient(sdfg, f"_tile_{conn.lstrip('_')}", src_arr.dtype, spec.widths)
        gather = TileGather(name=f"{tasklet.label}_gather_{conn.lstrip('_')}",
                            widths=spec.widths,
                            source_ndim=src_ndim,
                            has_mask=mask_name is not None)
        state.add_node(gather)
        full = ", ".join(f"0:{s}" for s in src_arr.shape)
        state.add_edge(in_edge.src, in_edge.src_conn, gather, TileConnectors.SRC,
                       dace.Memlet(data=src_name, subset=subsets.Range.from_string(full)))
        for k, idx_acc in enumerate(idx_accesses):
            state.add_edge(idx_acc, None, gather, TileConnectors.idx(k),
                           dace.Memlet(f"{idx_acc.data}[{', '.join(f'0:{w}' for w in spec.widths)}]"))
        out_subset = ", ".join(f"0:{w}" for w in spec.widths)
        mask_access = self._mask_access_or_none(state, mask_name)
        if mask_access is not None:
            state.add_edge(mask_access, None, gather, TileConnectors.MASK, dace.Memlet(f"{mask_name}[{out_subset}]"))
        tile_access = state.add_access(tile_name)
        state.add_edge(gather, TileConnectors.DST, tile_access, None, dace.Memlet(f"{tile_name}[{out_subset}]"))
        return tile_name, tile_access

    def _emit_scatter_store(self, state: dace.SDFGState, map_entry: MapEntry, out_access: dace.nodes.AccessNode,
                            out_edge, spec: TileDimSpec, mask_access: dace.nodes.AccessNode) -> None:
        """Scatter a result tile back to a non-box (diagonal / affine) output
        via a :class:`TileScatter` over the same affine index map.

        :param state: Parent state.
        :param map_entry: Inner map entry (index-tile dependency source).
        :param out_access: AccessNode holding the result tile.
        :param out_edge: The original store edge (to the MapExit).
        :param spec: Surrounding tile spec.
        :param mask_access: The iteration-mask access node.
        """
        sdfg = state.sdfg
        dst_name = out_edge.data.data
        dst_arr = sdfg.arrays[dst_name]
        dst_ndim = len(dst_arr.shape)
        out_subset = ", ".join(f"0:{w}" for w in spec.widths)
        idx_accesses = self._emit_gather_index_tiles(state, map_entry, out_edge.data.subset, dst_ndim, spec)
        scatter = TileScatter(name=f"{out_access.data}_scatter",
                              widths=spec.widths,
                              dest_ndim=dst_ndim,
                              has_mask=mask_access is not None)
        state.add_node(scatter)
        state.add_edge(out_access, None, scatter, TileConnectors.SRC, dace.Memlet(f"{out_access.data}[{out_subset}]"))
        for k, idx_acc in enumerate(idx_accesses):
            state.add_edge(idx_acc, None, scatter, TileConnectors.idx(k), dace.Memlet(f"{idx_acc.data}[{out_subset}]"))
        if mask_access is not None:
            state.add_edge(mask_access, None, scatter, TileConnectors.MASK,
                           dace.Memlet(f"{mask_access.data}[{out_subset}]"))
        full = ", ".join(f"0:{s}" for s in dst_arr.shape)
        state.add_edge(scatter, TileConnectors.DST, out_edge.dst, out_edge.dst_conn,
                       dace.Memlet(data=dst_name, subset=subsets.Range.from_string(full)))

    def _emit_const_tile(self, state: dace.SDFGState, map_entry: MapEntry, tasklet: dace.nodes.Tasklet, val: str,
                         spec: TileDimSpec) -> Tuple[str, dace.nodes.AccessNode]:
        """Materialize a tile filled with a constant (for a ``out = <const>``
        broadcast-store body), returning ``(out_data, tile_access)``.

        :param state: Parent state.
        :param map_entry: Inner map entry (dependency source for scope).
        :param tasklet: The constant-store tasklet (its out-edge names the
            destination data).
        :param val: The constant literal string.
        :param spec: Surrounding tile spec.
        :returns: ``(out_data_name, const_tile_access)``.
        """
        sdfg = state.sdfg
        out_data = self._binop_output_data(state, tasklet)
        dtype = sdfg.arrays[out_data].dtype
        tile_name = self._add_tile_transient(sdfg, "_tile_c", dtype, spec.widths)
        out_subset = ", ".join(f"0:{w}" for w in spec.widths)
        body = f"_out[{tile_offset(spec.widths)}] = {val};"
        fill = state.add_tasklet(
            name=f"const_{tile_name}",
            inputs=set(),
            outputs={"_out"},
            code=nested_loops(spec.widths, body),
            language=dace.dtypes.Language.CPP,
        )
        state.add_nedge(map_entry, fill, dace.Memlet())
        acc = state.add_access(tile_name)
        state.add_edge(fill, "_out", acc, None, dace.Memlet(f"{tile_name}[{out_subset}]"))
        return out_data, acc

    def _map_entry_of(self, state: dace.SDFGState, node: dace.nodes.Node) -> MapEntry:
        """Return the MapEntry whose scope contains ``node``.

        :param state: Parent state.
        :param node: A node inside an inner map scope.
        :returns: The enclosing :class:`MapEntry`.
        """
        scope = state.scope_dict()
        cur = scope[node]
        while cur is not None and not isinstance(cur, MapEntry):
            cur = scope[cur]
        return cur

    def _find_mask_access(self, state: dace.SDFGState, mask_name: str) -> Optional[dace.nodes.AccessNode]:
        """Return the AccessNode for ``mask_name`` placed by
        :class:`GenerateTileIterationMask`, if present.

        :param state: Parent state.
        :param mask_name: Name of the iteration-mask transient.
        :returns: The producer-fed access node, or ``None``.
        """
        producers = [n for n in state.data_nodes() if n.data == mask_name and state.in_edges(n)]
        return producers[0] if producers else None

    def _mask_access_or_none(self, state: dace.SDFGState, mask_name: Optional[str]) -> Optional[dace.nodes.AccessNode]:
        """Resolve the mask access node for ``mask_name``, or ``None`` when
        the map is unmasked (``mask_name is None``).

        :param state: Parent state.
        :param mask_name: Iteration-mask transient name, or ``None``.
        :returns: The mask access node, or ``None`` for the unmasked fast path.
        """
        if mask_name is None:
            return None
        return self._find_mask_access(state, mask_name) or state.add_access(mask_name)

    def _resolve_map_mask(self, state: dace.SDFGState, map_entry: MapEntry) -> Optional[str]:
        """Resolve the map's mask name, enforcing the no-mask safety contract.

        A map with no :class:`TileMaskGen` in scope is only legitimately
        unmasked when it is the ``masked_tail`` split's provably-divisible
        interior (the ``__tile_main`` marker) — every tile is fully in bounds,
        so ``has_mask=False`` is OOB-safe. An *unmarked* map with no mask means
        :class:`GenerateTileIterationMask` was not run (or the map is not a
        divisible interior); emitting ``has_mask=False`` there could be
        OOB-unsafe, so refuse loudly.

        :param state: Parent state.
        :param map_entry: Inner map entry.
        :returns: The mask array name, or ``None`` for the marked interior.
        :raises NotImplementedError: If the map is unmasked but not a
            ``__tile_main`` interior.
        """
        mask_name = _mask_name_for_map(state, map_entry)
        if mask_name is None and not map_entry.map.label.endswith(TILE_MAIN_MARKER):
            raise NotImplementedError(f"EmitTileOps: map {map_entry.label!r} has no TileMaskGen in scope and is not a "
                                      f"masked_tail interior (__tile_main); run GenerateTileIterationMask first.")
        return mask_name

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
                # An AccessNode whose only outgoing edge carries a WCR is the
                # post-:class:`NormalizeWCRSource` private-scalar reduction
                # target — DO NOT walk past it. The caller routes such an
                # edge into ``reductions`` and emits a ``TileReduce`` that
                # writes ``dst``; walking past would re-target the WCR onto
                # the outer scalar (acc) and drop the reduction.
                if nxt[0].data is not None and nxt[0].data.wcr is not None:
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

    def _wire_scalar_operand(self, state: dace.SDFGState, binop, conn: str, source_edge, in_edge) -> None:
        """Wire a Scalar (length-1 / ``dace.data.Scalar``) operand into
        ``binop`` via its ``_a`` / ``_b`` connector, reusing the original
        in-edge's source (the MapEntry pass-through) so no connector is
        orphaned.

        :param state: Parent state.
        :param binop: The ``TileBinop`` lib node.
        :param conn: ``TileConnectors.A`` or ``TileConnectors.B`` — the scalar connector.
        :param source_edge: Outermost memlet-path edge (its source is
            the original AccessNode).
        :param in_edge: The original binop tasklet's in-edge for this
            operand (carries the MapEntry pass-through source + conn).
        """
        state.add_edge(in_edge.src, in_edge.src_conn, binop, conn, copy.deepcopy(in_edge.data))

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

    def _topo_order_binops(self, state: dace.SDFGState, binops: List[dace.nodes.Tasklet]) -> List[dace.nodes.Tasklet]:
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
                    e.data.data
                    for e in state.in_edges(b) if isinstance(e.src, dace.nodes.AccessNode)
                    and e.data.data in produced_by_binop and e.data.data != out_data[b]
                }
                if deps <= produced:
                    ordered.append(b)
                    if out_data[b] is not None:
                        produced.add(out_data[b])
                    remaining.remove(b)
                    progressed = True
            if not progressed:
                raise NotImplementedError("EmitTileOps: cyclic / unresolvable binop dependency in map body")
        return ordered

    def _resolve_operand(self, state: dace.SDFGState, tasklet: dace.nodes.Tasklet, token: str,
                         tile_map: Dict[str, Tuple[str,
                                                   dace.nodes.AccessNode]], spec: TileDimSpec) -> Tuple[str, object]:
        """Classify one operand token of a body tasklet.

        :param state: Parent state.
        :param tasklet: The compute tasklet owning the operand.
        :param token: The in-connector name (or numeric literal) of the operand.
        :param tile_map: Intermediate-data-name -> ``(tile_name, tile_access)``.
        :param spec: Per-dim tile specification.
        :returns: ``(kind, info)`` where kind is ``Symbol`` (numeric literal,
            info is the literal), ``Tile-existing`` (info is the intermediate
            tile's access node), or whatever :meth:`_operand_kind` returns
            (``Tile`` / ``Gather`` / ``Scalar`` with their info tuples).
        :raises NotImplementedError: If the operand does not have exactly one
            in-edge.
        """
        if _is_numeric_literal(token):
            return "Symbol", token
        in_edges = [e for e in state.in_edges(tasklet) if e.dst_conn == token]
        if len(in_edges) == 0 and len(spec.widths) == 1 and token in spec.iter_vars:
            # Tile iter-var read as a free symbol (no connector): embed inline
            # via per-lane expansion ``v -> v + __l_p`` so each lane in the
            # tile sees its own iteration value. Mirrors the ``Symbol`` path
            # in :class:`PromoteNSDFGBodyToTiles`. Restricted to K=1 because
            # the scatter/affine-index shape for K>=2 is not yet wired.
            expr = _lane_index_expr(token, spec.iter_vars) or token
            return "Symbol", expr
        if len(in_edges) != 1:
            raise NotImplementedError(f"EmitTileOps: tasklet {tasklet.label!r} operand {token!r} has "
                                      f"{len(in_edges)} in-edges")
        edge = in_edges[0]
        src_data = edge.data.data
        if isinstance(edge.src, dace.nodes.AccessNode) and src_data in tile_map:
            return "Tile-existing", tile_map[src_data][1]
        return self._operand_kind(state, tasklet, token, spec)

    def _wire_operand(self, state: dace.SDFGState, node: dace.nodes.Node, conn: str, kind: str, info: object,
                      tasklet: dace.nodes.Tasklet, spec: TileDimSpec, mask_name: Optional[str], subset: str) -> None:
        """Wire a resolved operand into tile-node connector ``conn``.

        A ``Symbol`` operand is embedded inline by the lib node, so nothing is
        wired for it.

        :param state: Parent state.
        :param node: The :class:`TileBinop` / :class:`TileUnop` consuming the operand.
        :param conn: Destination connector on ``node`` (e.g. ``TileConnectors.A``).
        :param kind: Resolver kind from :meth:`_resolve_operand`.
        :param info: Resolver payload from :meth:`_resolve_operand`.
        :param tasklet: The originating body tasklet (for load emission context).
        :param spec: Per-dim tile specification.
        :param mask_name: Iteration-mask transient name, or ``None``.
        :param subset: ``"0:W0, 0:W1, ..."`` full-tile subset string.
        """
        if kind == "Tile-existing":
            state.add_edge(info, None, node, conn, dace.Memlet(f"{info.data}[{subset}]"))
        elif kind == "Tile":
            tname, tacc = self._emit_tile_load(state, tasklet, conn[-1], info[0], info[1], info[2], info[3], info[4],
                                               spec, mask_name)
            state.add_edge(tacc, None, node, conn, dace.Memlet(f"{tname}[{subset}]"))
        elif kind == "Gather":
            tname, tacc = self._emit_gather_load(state, tasklet, conn[-1], info[0], info[1], info[2], spec, mask_name)
            state.add_edge(tacc, None, node, conn, dace.Memlet(f"{tname}[{subset}]"))
        elif kind == "Scalar":
            self._wire_scalar_operand(state, node, conn, info[0], info[1])
        # Symbol: nothing to wire (embedded inline).

    def _finish_tile_op(self, state: dace.SDFGState, node: dace.nodes.Node, tasklet: dace.nodes.Tasklet,
                        mask_name: Optional[str], spec: TileDimSpec, subset: str) -> Tuple[str, dace.nodes.AccessNode]:
        """Wire the iteration mask (if any) and a fresh output tile for a tile op.

        :param state: Parent state.
        :param node: The :class:`TileBinop` / :class:`TileUnop` to finish.
        :param tasklet: The originating body tasklet (for output dtype + data name).
        :param mask_name: Iteration-mask transient name, or ``None``.
        :param spec: Per-dim tile specification.
        :param subset: ``"0:W0, 0:W1, ..."`` full-tile subset string.
        :returns: ``(out_data_name, out_tile_access)``.
        """
        out_dtype = state.sdfg.arrays[self._binop_output_data(state, tasklet)].dtype
        out_tile_name = self._add_tile_transient(state.sdfg, "_tile_t", out_dtype, spec.widths)
        mask_access = self._mask_access_or_none(state, mask_name)
        if mask_access is not None:
            state.add_edge(mask_access, None, node, TileConnectors.MASK, dace.Memlet(f"{mask_name}[{subset}]"))
        out_access = state.add_access(out_tile_name)
        state.add_edge(node, TileConnectors.C, out_access, None, dace.Memlet(f"{out_tile_name}[{subset}]"))
        return self._binop_output_data(state, tasklet), out_access

    def _wcr_scalar_target(self, state: dace.SDFGState,
                           candidate) -> Optional[Tuple[dace.nodes.AccessNode, str]]:
        """Detect the post-:class:`NormalizeWCRSource` reduction shape.

        After ``NormalizeWCRSource`` interposes a private scalar between a
        CodeNode and its WCR sink, an inner tasklet's downstream chain looks
        like ``tasklet -> _wcr_src (Scalar) -[wcr]-> sink``. The candidate
        passed in is the tasklet's final-edge destination (after walking past
        any plain assign tasklets).

        :param state: Parent state.
        :param candidate: The downstream node of the tasklet's final edge.
        :returns: ``(scalar_access, op)`` when ``candidate`` is a Scalar
            AccessNode with exactly one outgoing WCR edge whose operator is
            recognised by :func:`_wcr_op`; ``None`` otherwise.
        """
        if not isinstance(candidate, dace.nodes.AccessNode):
            return None
        desc = state.sdfg.arrays.get(candidate.data)
        if desc is None:
            return None
        if not (isinstance(desc, dace.data.Scalar)
                or (isinstance(desc, dace.data.Array) and tuple(desc.shape) == (1, ))):
            return None
        wcr_edges = [e for e in state.out_edges(candidate) if e.data is not None and e.data.wcr is not None]
        if len(wcr_edges) != 1:
            return None
        op = _wcr_op(wcr_edges[0].data.wcr)
        if op is None or op not in ("+", "*", "min", "max"):
            return None
        return candidate, op

    def _emit_tile_reduce(self, state: dace.SDFGState, tile_access: dace.nodes.AccessNode,
                          scalar_access: dace.nodes.AccessNode, op: str,
                          spec: TileDimSpec, mask_name: Optional[str]) -> None:
        """Emit a :class:`TileReduce` writing the tile to the scalar.

        The original tasklet that fed ``scalar_access`` is removed by the
        downstream cleanup; this helper re-wires the source side so the WCR
        edge from ``scalar_access`` to its sink keeps its semantics — the
        tile is reduced to a single value, written into ``scalar_access``,
        and the outer WCR aggregates across worker chunks.

        :param state: Parent state.
        :param tile_access: The tile transient produced by the binop / copy
            (``widths``-shaped).
        :param scalar_access: The destination Scalar AccessNode created by
            :class:`NormalizeWCRSource`.
        :param op: Reduction operator (``+ * min max``).
        :param spec: Per-dim tile specification.
        :param mask_name: Iteration-mask transient name, or ``None``.
        """
        subset = ", ".join(f"0:{w}" for w in spec.widths)
        mask_access = self._mask_access_or_none(state, mask_name)
        reduce_node = TileReduce(
            name=f"{tile_access.data}_reduce_{op}".replace("+", "add").replace("*", "mul"),
            widths=spec.widths,
            op=op,
            axis=None,
            has_mask=mask_access is not None,
        )
        state.add_node(reduce_node)
        state.add_edge(tile_access, None, reduce_node, "_src", dace.Memlet(f"{tile_access.data}[{subset}]"))
        if mask_access is not None:
            state.add_edge(mask_access, None, reduce_node, "_mask", dace.Memlet(f"{mask_name}[{subset}]"))
        state.add_edge(reduce_node, "_dst", scalar_access, None, dace.Memlet(f"{scalar_access.data}[0]"))

    def _emit_one_copy(self, state: dace.SDFGState, tasklet: dace.nodes.Tasklet, spec: TileDimSpec,
                       tile_map: Dict[str, Tuple[str, dace.nodes.AccessNode]],
                       mask_name: Optional[str]) -> Optional[dace.nodes.AccessNode]:
        """Emit a load for a pure-copy ``_out = _in`` body tasklet.

        Used when the map scope contains only assign tasklets (no compute):
        ``dst[i] = src[i//2]``-style kernels where the load itself (TileLoad
        / TileGather, selected by the in-edge's classification) IS the tile
        op. The returned tile transient access node feeds the stores phase
        which then emits the matching TileStore back to the output array.

        :param state: Parent state.
        :param tasklet: The assign tasklet.
        :param spec: Per-dim tile specification.
        :param tile_map: Intermediate-data-name -> ``(tile_name, tile_access)``.
        :param mask_name: Iteration-mask transient name, or ``None``.
        :returns: The loaded tile's access node, or ``None`` when the body
            does not match the supported pure-copy shape.
        """
        body = tasklet.code.as_string.strip().rstrip(";").strip()
        m = _ASSIGN_RE.match(body)
        if m is None:
            return None
        in_conn = m.group("inp")
        kind, info = self._resolve_operand(state, tasklet, in_conn, tile_map, spec)
        if kind == "Tile-existing":
            return info
        if kind == "Tile":
            _, tacc = self._emit_tile_load(state, tasklet, in_conn.lstrip("_"), info[0], info[1], info[2], info[3],
                                           info[4], spec, mask_name)
            return tacc
        if kind == "Gather":
            _, tacc = self._emit_gather_load(state, tasklet, in_conn.lstrip("_"), info[0], info[1], info[2], spec,
                                             mask_name)
            return tacc
        # Scalar / Symbol broadcast as a pure-copy body is unusual; the
        # standard binop emit path supports it via a TileBinop wrapper. Fall
        # through to the original "no binop" error so the gap is visible.
        return None

    def _emit_one_binop(self, state: dace.SDFGState, tasklet: dace.nodes.Tasklet, spec: TileDimSpec,
                        tile_map: Dict[str, Tuple[str, dace.nodes.AccessNode]],
                        mask_name: str) -> Tuple[str, dace.nodes.AccessNode]:
        """Emit a single :class:`TileBinop` for one body tasklet.

        Resolves each operand (:meth:`_resolve_operand`) to a Tile (existing
        intermediate from ``tile_map`` or a fresh load), a Scalar, or a Symbol
        (numeric literal embedded inline), wires them in, and writes the result
        to a fresh tile transient.

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
        kind_a, info_a = self._resolve_operand(state, tasklet, a_tok, tile_map, spec)
        kind_b, info_b = self._resolve_operand(state, tasklet, b_tok, tile_map, spec)
        norm_a = _normalize_operand_kind(kind_a)
        norm_b = _normalize_operand_kind(kind_b)
        if norm_a != "Tile" and norm_b != "Tile":
            raise NotImplementedError(f"EmitTileOps: tasklet {tasklet.label!r} has no Tile operand "
                                      f"({norm_a}/{norm_b})")
        kwargs = dict(name=f"{tasklet.label}_binop",
                      widths=spec.widths,
                      op=op,
                      has_mask=mask_name is not None,
                      kind_a=norm_a,
                      kind_b=norm_b)
        if norm_a == "Symbol":
            kwargs["expr_a"] = info_a
        if norm_b == "Symbol":
            kwargs["expr_b"] = info_b
        binop = TileBinop(**kwargs)
        state.add_node(binop)
        self._wire_operand(state, binop, TileConnectors.A, kind_a, info_a, tasklet, spec, mask_name, subset)
        self._wire_operand(state, binop, TileConnectors.B, kind_b, info_b, tasklet, spec, mask_name, subset)
        return self._finish_tile_op(state, binop, tasklet, mask_name, spec, subset)

    def _emit_one_unop(self, state: dace.SDFGState, tasklet: dace.nodes.Tasklet, spec: TileDimSpec,
                       tile_map: Dict[str, Tuple[str, dace.nodes.AccessNode]],
                       mask_name: str) -> Tuple[str, dace.nodes.AccessNode]:
        """Emit a single :class:`TileUnop` for one unary body tasklet.

        Mirrors :meth:`_emit_one_binop` with a single operand (resolved and
        wired through the same :meth:`_resolve_operand` / :meth:`_wire_operand`
        / :meth:`_finish_tile_op` helpers).

        :param state: Parent state.
        :param tasklet: The unary tasklet.
        :param spec: Per-dim tile specification.
        :param tile_map: Intermediate-data-name -> ``(tile_name, tile_access)``.
        :param mask_name: Iteration-mask transient name.
        :returns: ``(out_data_name, out_tile_access)``.
        :raises NotImplementedError: For operand shapes not handled.
        """
        out_conn, op, a_tok = _classify_unop_tasklet_body(tasklet)
        subset = ", ".join(f"0:{w}" for w in spec.widths)
        kind_a, info_a = self._resolve_operand(state, tasklet, a_tok, tile_map, spec)
        norm_a = _normalize_operand_kind(kind_a)
        kwargs = dict(name=f"{tasklet.label}_unop",
                      widths=spec.widths,
                      op=op,
                      has_mask=mask_name is not None,
                      kind_a=norm_a)
        if norm_a == "Symbol":
            kwargs["expr_a"] = info_a
        unop = TileUnop(**kwargs)
        state.add_node(unop)
        self._wire_operand(state, unop, TileConnectors.A, kind_a, info_a, tasklet, spec, mask_name, subset)
        return self._finish_tile_op(state, unop, tasklet, mask_name, spec, subset)

    def _rewrite_one_map(self, state: dace.SDFGState, map_entry: MapEntry, spec: TileDimSpec) -> None:
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
        assign_tasklets = [t for t in tasklets if _is_assign_tasklet(t)]
        const_stores = [t for t in tasklets if not _is_assign_tasklet(t) and _constant_store_value(t) is not None]
        binops = [t for t in tasklets if not _is_assign_tasklet(t) and _constant_store_value(t) is None]
        # Pure-copy bodies (``dst[i] = src[i//2]`` style) reach here with only
        # assign tasklets: there is no binop / unop / const-store to anchor on,
        # but each assign IS the tile op — a load (possibly with a gather /
        # strided index) directly feeding the output. Emit each assign as a
        # standalone load whose tile transient becomes the store source; the
        # stores phase below picks it up exactly like a binop's output tile.
        copy_only_emit = (not binops and not const_stores)
        if copy_only_emit and not assign_tasklets:
            raise NotImplementedError(
                f"EmitTileOps: map {map_entry.label!r} body has no binop, constant-store, "
                f"or assign tasklet")

        mask_name = self._resolve_map_mask(state, map_entry)
        subset = ", ".join(f"0:{w}" for w in spec.widths)
        # Compute tasklets (binops + unops) share one dataflow order; each is
        # dispatched to the right tile node by its body shape.
        ordered = self._topo_order_binops(state, binops)

        tile_map: Dict[str, Tuple[str, dace.nodes.AccessNode]] = {}
        stores: List[Tuple[dace.nodes.AccessNode, object]] = []
        # ``reductions`` carries the post-:class:`NormalizeWCRSource` pattern
        # ``tile -> _wcr_src (Scalar) -[wcr]-> sink``: each entry is the source
        # tile AccessNode, the (un-wired) tasklet-to-scalar edge, and the WCR
        # target tuple ``(scalar_access, wcr_op_str)``. Lowered to a TileReduce
        # in the post-walk emission below.
        reductions: List[Tuple[dace.nodes.AccessNode, object, Tuple[dace.nodes.AccessNode, str]]] = []
        all_intermediates: set = set()
        for t in ordered:
            out_e = list(state.out_edges(t))
            if _classify_unop_tasklet_body(t) is not None:
                out_data, out_access = self._emit_one_unop(state, t, spec, tile_map, mask_name)
            else:
                out_data, out_access = self._emit_one_binop(state, t, spec, tile_map, mask_name)
            tile_map[out_data] = (out_access.data, out_access)
            final_edge, inters = self._walk_through_assigns(state, out_e[0], assign_tasklets)
            all_intermediates |= set(inters)
            if isinstance(final_edge.dst, dace.nodes.MapExit):
                stores.append((out_access, final_edge))
            else:
                wcr_target = self._wcr_scalar_target(state, final_edge.dst)
                if wcr_target is not None:
                    reductions.append((out_access, final_edge, wcr_target))
        if copy_only_emit:
            # Walk forward from each assign; if it lands on the MapExit, emit a
            # load for its in-edge and queue the load's tile as the store source.
            # An assign whose final dst is a Scalar AccessNode with a downstream
            # WCR (the post-:class:`NormalizeWCRSource` shape ``tasklet ->
            # _wcr_src (Scalar) -[wcr]-> sink``) lowers to a TileReduce instead.
            for t in assign_tasklets:
                out_e = list(state.out_edges(t))
                if not out_e:
                    continue
                final_edge, inters = self._walk_through_assigns(state, out_e[0], assign_tasklets)
                wcr_target = self._wcr_scalar_target(state, final_edge.dst)
                if not (isinstance(final_edge.dst, dace.nodes.MapExit) or wcr_target is not None):
                    continue
                tile_access = self._emit_one_copy(state, t, spec, tile_map, mask_name)
                if tile_access is None:
                    continue
                all_intermediates |= set(inters)
                if wcr_target is not None:
                    reductions.append((tile_access, final_edge, wcr_target))
                else:
                    stores.append((tile_access, final_edge))
        for t in const_stores:
            out_e = list(state.out_edges(t))
            out_data, out_access = self._emit_const_tile(state, map_entry, t, _constant_store_value(t), spec)
            tile_map[out_data] = (out_access.data, out_access)
            final_edge, inters = self._walk_through_assigns(state, out_e[0], assign_tasklets)
            all_intermediates |= set(inters)
            if isinstance(final_edge.dst, dace.nodes.MapExit):
                stores.append((out_access, final_edge))

        for out_access, out_edge in stores:
            out_dst_name = out_edge.data.data
            out_arr = state.sdfg.arrays[out_dst_name]
            out_cls = classify_tile_access(out_edge.data.subset, tuple(out_arr.strides), spec.iter_vars)
            mask_access = self._mask_access_or_none(state, mask_name)
            if out_cls.kind in (TileAccessKind.GATHER, TileAccessKind.STRUCTURED):
                # Non-box / structured write (``a[i,i]=…`` / ``a[i//2]=…``):
                # scatter the tile back through the same per-dim index map.
                self._emit_scatter_store(state, map_entry, out_access, out_edge, spec, mask_access)
                continue
            if out_cls.kind not in (TileAccessKind.CONTIGUOUS, TileAccessKind.STRIDED):
                raise NotImplementedError(f"EmitTileOps: output {out_dst_name!r} access is {out_cls.kind.value}; "
                                          f"only perfect-box (strided) or gather stores are supported")
            promoted = _tile_region_subset(out_edge.data.subset, spec.iter_vars, spec.widths)
            store = TileStore(name=f"{out_access.data}_store",
                              widths=spec.widths,
                              dim_strides=out_cls.dim_strides,
                              dst_dims=out_cls.match_dims,
                              has_mask=mask_access is not None)
            state.add_node(store)
            state.add_edge(out_access, None, store, TileConnectors.SRC, dace.Memlet(f"{out_access.data}[{subset}]"))
            if mask_access is not None:
                state.add_edge(mask_access, None, store, TileConnectors.MASK, dace.Memlet(f"{mask_name}[{subset}]"))
            state.add_edge(store, TileConnectors.DST, out_edge.dst, out_edge.dst_conn,
                           dace.Memlet(data=out_dst_name, subset=promoted))

        for tile_acc, _orig_edge, (scalar_acc, op) in reductions:
            self._emit_tile_reduce(state, tile_acc, scalar_acc, op, spec, mask_name)

        # Remove the original body tasklets + the intermediate transients
        # that only connected them — scoped, so no isolated node remains.
        for node in list(binops) + list(const_stores) + list(assign_tasklets) + list(all_intermediates):
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
        # Maps whose body NSDFG was already tiled in place by
        # PromoteNSDFGBodyToTiles: skip them (their flat scope has no binop
        # tasklet, so _rewrite_one_map would wrongly raise "no binop").
        handled: set = set()
        if pipeline_results and "PromoteNSDFGBodyToTiles" in pipeline_results:
            handled = pipeline_results["PromoteNSDFGBodyToTiles"] or set()
        K = len(self.widths)
        rewritten = 0
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, MapEntry) or not isinstance(g, dace.SDFGState):
                continue
            if not is_innermost_map(g, n):
                continue
            if n.map.label.endswith(SCALAR_TAIL_MARKER):  # scalar_postamble tail: stays scalar
                continue
            if n in handled:
                continue
            if specs is not None and n not in specs:
                continue
            if len(n.map.params) < K:
                continue
            # Verify this map either has a mask producer in scope or is a
            # masked_tail divisible interior (raises NotImplementedError
            # otherwise — a forgotten GenerateTileIterationMask).
            self._resolve_map_mask(g, n)
            spec = specs[n] if specs is not None and n in specs else self._spec_for(n)
            self._rewrite_one_map(g, n, spec)
            rewritten += 1
        return rewritten or None
