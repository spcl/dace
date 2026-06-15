# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared CPP-codegen for the K=1 ISA tile-op expansions.

The five ISA backends (scalar / avx512 / avx2 / arm_neon / arm_sve) expose the
SAME ``dace::tileops::tile_<op>`` function signatures (see
``dace/runtime/include/dace/tile_ops/<backend>.h``), so a single call-builder per
op produces the CPP tasklet body for every backend; the per-backend expansion
class differs ONLY in which environment (hence which header) it declares. These
expansions are K=1 only — the selector routes K>=2 tiles to ``pure``.

Each ``make_<op>_tasklet`` returns the finished :class:`~dace.sdfg.nodes.Tasklet`
(its ``label`` carries the backend ``suffix`` for readability); the calling
expansion class attaches the backend environment.
"""
import dace
from dace.sdfg import nodes
from dace.symbolic import symstr

# TileBinop.op -> the single-char op code the backend headers template on
# (``dace::tileops::tile_binop<T, VLEN, Op, ...>``; legend in scalar.h).
_OP_TO_CHAR = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "min": "m",
    "max": "M",
    "<": "<",
    "<=": "l",
    ">": ">",
    ">=": "g",
    "==": "=",
    "!=": "!",
    "&&": "&",
    "||": "|",
}

_TILE = "Tile"
_SYMBOL = "Symbol"
_SCALAR = "Scalar"

# TileUnop.op -> the single-char op code the backend headers' ``tile_unop``
# templates on (legend in scalar.h: n neg, a abs, e exp, l log, s sqrt, ...).
_UNOP_TO_CHAR = {
    "neg": "n",
    "not": "!",
    "abs": "a",
    "exp": "e",
    "log": "l",
    "sqrt": "s",
    "sin": "S",
    "cos": "C",
    "floor": "f",
    "ceil": "c",
    "tanh": "t",
}


def _require_k1(node) -> int:
    """Return the K=1 tile width, or raise if the node is not K=1.

    :param node: A tile lib node carrying ``widths``.
    :returns: The single tile width ``widths[0]``.
    :raises NotImplementedError: If ``len(widths) != 1`` (the selector should
        route K>=2 tiles to the ``pure`` expansion, never here).
    """
    widths = list(node.widths)
    if len(widths) != 1:
        raise NotImplementedError(f"{node.label}: ISA tile-op backend is K=1 only; K>=2 lowers to 'pure'")
    return widths[0]


def _out_ctype(node, parent_state, parent_sdfg, out_conn: str) -> str:
    """C++ element type of the array on ``node``'s ``out_conn`` output edge."""
    e = next(e for e in parent_state.out_edges(node) if e.src_conn == out_conn)
    return parent_sdfg.arrays[e.data.data].dtype.ctype


def _in_ctype(node, parent_state, parent_sdfg, in_conn: str) -> str:
    """C++ element type of the array on ``node``'s ``in_conn`` input edge."""
    e = next(e for e in parent_state.in_edges(node) if e.dst_conn == in_conn)
    return parent_sdfg.arrays[e.data.data].dtype.ctype


def _resolve_operand_ctype(node, parent_state, parent_sdfg, conns, out_dtype: str) -> str:
    """Resolve the C++ type the VALUE operands share (mirror of
    ``ExpandTileBinopPure._operand_dtype``).

    Prefer a data operand's (``_TILE`` / ``_SCALAR``) descriptor dtype; else a
    symbol operand's declared type; else fall back to ``out_dtype``. Used to
    decide whether the single-``T`` ISA runtime can carry this op (operands and
    output share a type) or whether it must defer to the ``pure`` expansion
    (operands and output differ -- a comparison's ``double`` operands vs ``bool``
    output). The ISA runtime is type-strict (``out``, ``a``, ``b`` are all
    ``T*``), so a mismatch would force a value-truncating ``(out_dtype)`` cast on
    the operands; per user direction 2026-06-15 such a C-style cast is always
    incorrect code, so we route to ``pure`` instead.
    """
    in_e = {e.dst_conn: e for e in parent_state.in_edges(node) if e.dst_conn is not None}
    for kind, conn in conns:
        if kind in (_TILE, _SCALAR) and conn in in_e:
            return parent_sdfg.arrays[in_e[conn].data.data].dtype.ctype
    for kind, conn, expr in [(k, c, e) for (k, c), e in zip(conns, _operand_exprs(node, conns))]:
        if kind == _SYMBOL and expr:
            try:
                for s in dace.symbolic.symlist(dace.symbolic.pystr_to_symbolic(expr)):
                    if str(s) in parent_sdfg.symbols:
                        return parent_sdfg.symbols[str(s)].ctype
            except Exception:  # noqa: BLE001
                pass
    return out_dtype


def _operand_exprs(node, conns):
    """Per-operand inline ``expr_*`` strings, aligned with ``conns`` order."""
    mapping = {"_a": node.expr_a, "_b": getattr(node, "expr_b", None)}
    return [mapping.get(conn) for _kind, conn in conns]


def _scalar_ref(conn: str, desc, subset) -> str:
    """C++ reference for a Scalar/broadcast operand connector.

    DaCe passes a tasklet input connector by value (``T conn``) for a true
    :class:`dace.data.Scalar` and for a single-element access into a larger array
    (``a[j]`` -> ``double conn = a[j]``); only a genuine length-1 *array*
    connector is passed as a pointer (``T* conn``) and must be dereferenced
    ``conn[0]``. The earlier code dereferenced every non-Scalar source, which
    faulted on the ``a[j]`` loop-invariant broadcast (``conn[0]`` on a scalar).

    :param conn: Connector name.
    :param desc: Source array/scalar descriptor.
    :param subset: The connector edge's subset (the access into ``desc``).
    :returns: ``conn`` (by value) or ``conn[0]`` (length-1 array pointer).
    """
    is_len1_array = (isinstance(desc, dace.data.Array)
                     and all(bool(dace.symbolic.simplify(s == 1)) for s in desc.shape))
    return f"{conn}[0]" if is_len1_array else conn


def make_binop_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP tasklet calling ``dace::tileops::tile_binop`` for ``node``.

    Maps ``op`` -> :data:`_OP_TO_CHAR`; each operand's ``kind`` -> the broadcast
    boolean (Tile reads the per-lane connector with ``Broadcast=false``; Scalar /
    Symbol are materialised into a length-1 ``T`` buffer read with
    ``Broadcast=true``, casting to the output type); ``has_mask`` -> ``Masked``.
    """
    node.validate(parent_sdfg, parent_state)
    vlen = _require_k1(node)
    in_e = {e.dst_conn: e for e in parent_state.in_edges(node) if e.dst_conn is not None}
    out_dtype = _out_ctype(node, parent_state, parent_sdfg, "_c")
    # The single-``T`` ISA runtime cannot carry an op whose operands and output
    # differ in type (a comparison: ``double`` operands, ``bool`` output). The
    # old code forced ``T = out_dtype`` and cast each operand to it -- truncating
    # ``(bool)1e-12 -> 1`` and corrupting the predicate. Defer to the ``pure``
    # expansion (which keeps operands at their own dtype and stores the result
    # with the natural implicit conversion) instead of emitting an incorrect
    # C-style cast (user direction 2026-06-15).
    operand_ctype = _resolve_operand_ctype(node, parent_state, parent_sdfg, [(node.kind_a, "_a"), (node.kind_b, "_b")],
                                           out_dtype)
    if operand_ctype != out_dtype:
        from dace.libraries.tileops.nodes.tile_binop import ExpandTileBinopPure
        return ExpandTileBinopPure.expansion(node, parent_state, parent_sdfg)
    pre = []

    def operand(kind, conn, expr):
        if kind == _TILE:
            src = parent_sdfg.arrays[in_e[conn].data.data].dtype.ctype
            if src == out_dtype:
                return "false", conn
            # Widening promotion (validate() already rejected narrowing): the
            # header takes a uniform ``T*``, so copy the input tile into an
            # out-dtype buffer with a per-lane cast before the op.
            buf = f"_cast{conn}"
            pre.append(f"{out_dtype} {buf}[{vlen}];")
            pre.append(f"for (int _ci = 0; _ci < {vlen}; ++_ci) {buf}[_ci] = ({out_dtype}){conn}[_ci];")
            return "false", buf
        if kind == _SYMBOL:
            val = f"({out_dtype})({expr})"
        else:
            desc = parent_sdfg.arrays[in_e[conn].data.data]
            val = f"({out_dtype})({_scalar_ref(conn, desc, in_e[conn].data.subset)})"
        buf = f"_bc{conn}"
        pre.append(f"{out_dtype} {buf}[1] = {{ {val} }};")
        return "true", buf

    a_bcast, a_ptr = operand(node.kind_a, "_a", node.expr_a)
    b_bcast, b_ptr = operand(node.kind_b, "_b", node.expr_b)
    op_char = _OP_TO_CHAR[node.op]
    masked = "true" if node.has_mask else "false"
    mask_arg = "_mask" if node.has_mask else "nullptr"
    call = (f"dace::tileops::tile_binop<{out_dtype}, {vlen}, '{op_char}', "
            f"{a_bcast}, {b_bcast}, {masked}>(_c, {a_ptr}, {b_ptr}, {mask_arg});")
    inputs = set()
    if node.kind_a in (_TILE, _SCALAR):
        inputs.add("_a")
    if node.kind_b in (_TILE, _SCALAR):
        inputs.add("_b")
    if node.has_mask:
        inputs.add("_mask")
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in inputs},
        outputs={"_c": None},
        code="\n".join(pre + [call]),
        language=dace.dtypes.Language.CPP,
    )


def make_unop_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP tasklet calling ``dace::tileops::tile_unop`` for ``node`` (K=1).

    Maps ``op`` -> :data:`_UNOP_TO_CHAR`; the operand's ``kind`` -> the
    broadcast boolean (Tile reads the per-lane connector with
    ``Broadcast=false``; Scalar / Symbol are materialised into a length-1 ``T``
    buffer read with ``Broadcast=true``, casting to the output type);
    ``has_mask`` -> ``Masked``.
    """
    node.validate(parent_sdfg, parent_state)
    vlen = _require_k1(node)
    in_e = {e.dst_conn: e for e in parent_state.in_edges(node) if e.dst_conn is not None}
    out_dtype = _out_ctype(node, parent_state, parent_sdfg, "_c")
    # Defer mixed operand/output-type unops (e.g. logical ``not`` of a numeric
    # tile -> bool) to the ``pure`` expansion rather than casting the operand to
    # the output type (user direction 2026-06-15: never emit a C-style cast).
    operand_ctype = _resolve_operand_ctype(node, parent_state, parent_sdfg, [(node.kind_a, "_a")], out_dtype)
    if operand_ctype != out_dtype:
        from dace.libraries.tileops.nodes.tile_unop import ExpandTileUnopPure
        return ExpandTileUnopPure.expansion(node, parent_state, parent_sdfg)
    pre = []
    if node.kind_a == _TILE:
        src = parent_sdfg.arrays[in_e["_a"].data.data].dtype.ctype
        if src == out_dtype:
            a_bcast, a_ptr = "false", "_a"
        else:
            # Widening promotion (validate() rejected narrowing): copy the input
            # tile into an out-dtype buffer with a per-lane cast before the op.
            a_ptr = "_cast_a"
            pre.append(f"{out_dtype} {a_ptr}[{vlen}];")
            pre.append(f"for (int _ci = 0; _ci < {vlen}; ++_ci) {a_ptr}[_ci] = ({out_dtype})_a[_ci];")
            a_bcast = "false"
    else:
        if node.kind_a == _SYMBOL:
            val = f"({out_dtype})({node.expr_a})"
        else:
            desc = parent_sdfg.arrays[in_e["_a"].data.data]
            val = f"({out_dtype})({_scalar_ref('_a', desc, in_e['_a'].data.subset)})"
        a_ptr = "_bc_a"
        pre.append(f"{out_dtype} {a_ptr}[1] = {{ {val} }};")
        a_bcast = "true"
    op_char = _UNOP_TO_CHAR[node.op]
    masked = "true" if node.has_mask else "false"
    mask_arg = "_mask" if node.has_mask else "nullptr"
    call = (f"dace::tileops::tile_unop<{out_dtype}, {vlen}, '{op_char}', "
            f"{a_bcast}, {masked}>(_c, {a_ptr}, {mask_arg});")
    inputs = set()
    if node.kind_a in (_TILE, _SCALAR):
        inputs.add("_a")
    if node.has_mask:
        inputs.add("_mask")
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in inputs},
        outputs={"_c": None},
        code="\n".join(pre + [call]),
        language=dace.dtypes.Language.CPP,
    )


def make_ite_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP tasklet calling ``dace::tileops::tile_ite`` (per-lane select).

    Unified-mask connector contract (user direction 2026-06-12): ``_mask``
    is the select-arm predicate; downstream global TileStore handles
    iter-mask gating. ``MaskT`` is the predicate tile's element type.
    """
    node.validate(parent_sdfg, parent_state)
    vlen = _require_k1(node)
    out_dtype = _out_ctype(node, parent_state, parent_sdfg, "_o")
    cond_dtype = _in_ctype(node, parent_state, parent_sdfg, "_mask")
    call = (f"dace::tileops::tile_ite<{out_dtype}, {cond_dtype}, {vlen}, false, false, false>"
            f"(_o, _mask, _t, _e, nullptr);")
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in ("_mask", "_t", "_e")},
        outputs={"_o": None},
        code=call,
        language=dace.dtypes.Language.CPP,
    )


def _k1_array_stride(node, parent_sdfg, edge, dims_prop) -> str:
    """Return the linear element stride of the K=1 tile dim into the array on
    ``edge`` (``dim_strides`` coefficient * the array's own stride along the
    mapped dim), as a C++ expression.
    """
    arr = parent_sdfg.arrays[edge.data.data]
    ndim = len(arr.strides)
    dims = list(dims_prop) if dims_prop else [ndim - 1]
    coeff = (list(node.dim_strides)[0] if node.dim_strides else 1)
    return f"({coeff}) * ({symstr(arr.strides[dims[0]])})"


def make_load_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP tasklet calling ``dace::tileops::tile_load`` (contiguous / strided).

    The K=1 tile-dim linear stride into the source array is passed as the
    ``stride`` argument; the header SIMD-loads when it is 1 and falls back to a
    scalar gathered read otherwise.

    ``src_kind != 'Tile'`` (a broadcast ``Symbol`` literal or a ``Scalar``
    length-1 read) has no per-lane ``_src`` tile pointer, so the K=1 runtime
    ``tile_load`` (which streams ``_src`` into ``_dst``) cannot express it; the
    pure expansion broadcasts the literal / scalar to every lane instead.
    Delegate to it -- symmetric to the ``src_kind != 'Tile'`` fallback in
    :func:`make_store_tasklet`.

    When ``gather_dims`` is set the runtime ``tile_load`` does not apply -- it
    has no per-lane index input -- so the lowering delegates to
    :class:`ExpandTileLoadPure`, which emits the per-lane indirect read using
    the ``_idx_<d>`` connectors (design section 9.3).
    """
    if node.src_kind != "Tile" or node.gather_dims:
        from dace.libraries.tileops.nodes.tile_load import ExpandTileLoadPure
        return ExpandTileLoadPure.expansion(node, parent_state, parent_sdfg)
    # REPLICATE codegen (user direction 2026-06-10): the K=1 intrinsic
    # ``tile_load<T, VLEN, Masked>(_dst, _src, mask, stride)`` does
    # ``dst[i] = src[i * stride]`` -- no per-lane replicate divisor. When
    # ``replicate_factor_per_dim[d] > 1`` (e.g. ``c[i // 2]`` -> factor 2
    # means lanes 0,1 share c[i//2], lanes 2,3 share c[i//2+1], ...), the
    # intrinsic produces wrong values. Fall back to the pure expansion,
    # which emits ``src[(__l/replicate) * stride]`` per lane.
    if node.replicate_factor_per_dim:
        needs_pure = False
        for r in node.replicate_factor_per_dim:
            try:
                if int(r) > 1:
                    needs_pure = True
                    break
            except (TypeError, ValueError):
                # Symbolic factor (e.g. ``DV``): always fall back to pure --
                # we can't prove it's 1 at compile time, so route safely.
                needs_pure = True
                break
        if needs_pure:
            from dace.libraries.tileops.nodes.tile_load import ExpandTileLoadPure
            return ExpandTileLoadPure.expansion(node, parent_state, parent_sdfg)
    vlen = _require_k1(node)
    src_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == "_src")
    dst_dtype = _out_ctype(node, parent_state, parent_sdfg, "_dst")
    stride = _k1_array_stride(node, parent_sdfg, src_edge, node.src_dims)
    masked = "true" if node.has_mask else "false"
    mask_arg = "_mask" if node.has_mask else "nullptr"
    call = f"dace::tileops::tile_load<{dst_dtype}, {vlen}, {masked}>(_dst, _src, {mask_arg}, {stride});"
    inputs = {"_src"} | ({"_mask"} if node.has_mask else set())
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in inputs},
        outputs={"_dst": None},
        code=call,
        language=dace.dtypes.Language.CPP,
    )


def make_store_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP tasklet calling ``dace::tileops::tile_store`` (RMW skip-inactive).

    The ISA runtime ``tile_store`` takes a tile pointer (``_src``) and
    streams it to the destination; ``src_kind != 'Tile'`` (a broadcast
    Symbol literal or a Scalar length-1 read) has no per-lane source
    pointer, so the pure expansion's per-lane store is the right
    lowering for those shapes. Delegate to it instead.

    When ``gather_dims`` is set the runtime ``tile_store`` does not apply
    either -- it has no per-lane index input -- so the lowering delegates
    to :class:`ExpandTileStorePure`, which emits the per-lane indirect
    store using the ``_idx_<d>`` connectors (design section 9.3 scatter).
    Symmetric to the ``make_load_tasklet`` gather fallback added in
    commit 4ad424945.
    """
    if node.src_kind != "Tile" or node.gather_dims:
        from dace.libraries.tileops.nodes.tile_store import ExpandTileStorePure
        return ExpandTileStorePure.expansion(node, parent_state, parent_sdfg)
    vlen = _require_k1(node)
    dst_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == "_dst")
    dst_dtype = parent_sdfg.arrays[dst_edge.data.data].dtype.ctype
    stride = _k1_array_stride(node, parent_sdfg, dst_edge, node.dst_dims)
    masked = "true" if node.has_mask else "false"
    mask_arg = "_mask" if node.has_mask else "nullptr"
    call = f"dace::tileops::tile_store<{dst_dtype}, {vlen}, {masked}>(_dst, _src, {mask_arg}, {stride});"
    inputs = {"_src"} | ({"_mask"} if node.has_mask else set())
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in inputs},
        outputs={"_dst": None},
        code=call,
        language=dace.dtypes.Language.CPP,
    )
