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

# ISA backends shared by every tile-op node: (implementation key, class-name
# suffix, environment attribute on ``..environments``). Each backend exposes the
# SAME ``dace::tileops::tile_<op>`` signature and differs ONLY in the header its
# environment pulls in -- so a single factory builds all five expansion classes.
_ISA_BACKENDS = (
    ("scalar", "Scalar", "TileOpsScalar"),
    ("avx512", "AVX512", "TileOpsAVX512"),
    ("avx2", "AVX2", "TileOpsAVX2"),
    ("neon", "Neon", "TileOpsNeon"),
    ("sve", "SVE", "TileOpsSVE"),
    ("cuda", "CUDA", "TileOpsCUDA"),
)


def make_isa_expansions(node_label: str, maker, module_globals: dict) -> dict:
    """Build the five per-ISA ``ExpandTransformation`` classes for a tile-op node.

    Every tile-op node (``TileBinop`` / ``TileUnop`` / ``TileITE`` / ``TileLoad``
    / ``TileStore`` / ``TileMaskGen``) exposes the same five K=1 ISA backends,
    each of which just calls the op's ``maker(node, state, sdfg, key)`` builder;
    the only per-class difference is the environment (hence the backend header)
    it declares. This factory replaces the five hand-written, near-identical
    expansion classes per node (~30 classes across the package).

    Each class is given the same ``__name__`` / ``__qualname__`` the hand-written
    class had (``ExpandTile<node_label><Suffix>``) and is bound into
    ``module_globals`` so any transformation lookup by qualified name still
    resolves. The SDFG only ever serializes the implementation KEY (``"avx512"``)
    and re-resolves the class from the node's in-code ``implementations`` map, so
    the factory-built classes round-trip identically to the hand-written ones.

    :param node_label: The node's CamelCase tag, e.g. ``"Binop"`` / ``"MaskGen"``.
    :param maker: The op's tasklet builder ``(node, state, sdfg, key) -> Tasklet``.
    :param module_globals: The defining module's ``globals()`` (for name binding).
    :returns: ``{"scalar": cls, "avx512": cls, "avx2": cls, "neon": cls, "sve": cls}``
        ready to splice into the node's ``implementations`` mapping.
    """
    from dace import library
    from dace.transformation.transformation import ExpandTransformation
    from . import environments as _env
    out = {}
    for key, suffix, env_name in _ISA_BACKENDS:
        cls_name = f"ExpandTile{node_label}{suffix}"

        def _expansion(node, parent_state, parent_sdfg, _maker=maker, _key=key):
            return _maker(node, parent_state, parent_sdfg, _key)

        cls = type(
            cls_name, (ExpandTransformation, ), {
                "environments": [getattr(_env, env_name)],
                "expansion": staticmethod(_expansion),
                "__doc__": f"{key} ISA lowering of Tile{node_label} (calls "
                f"_isa_codegen.{maker.__name__}; the {env_name} environment pulls in the header).",
                "__module__": module_globals.get("__name__", __name__),
                "__qualname__": cls_name,
            })
        out[key] = library.expansion(cls)
        module_globals[cls_name] = out[key]
    return out


# TileBinop.op -> the single-char op code the backend headers template on
# (``dace::tileops::tile_binop<T, VLEN, Op, ...>``; legend in scalar.h).
_OP_TO_CHAR = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    # Python/NumPy modulo: the backend headers lower the ``%`` op char to
    # ``dace::math::py_mod`` (divisor-sign semantics), never C's truncated ``%``.
    # ``py_mod`` is the function-call spelling of the same op (emitted by the
    # ``RewriteModuloToPyMod`` cleaning step) and maps to the SAME backend char.
    "%": "%",
    "py_mod": "%",
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
    # A dtype-name cast op (``float64`` / ``int32`` / ...) changes dtype by definition;
    # there is no per-ISA ``tile_unop`` op char for it, so lower it via the pure
    # per-lane ``dace::<dtype>(x)`` cast (the operand != output fallback below also
    # catches the usual differing-dtype cast, but a no-op same-dtype cast must route
    # here too rather than KeyError on _UNOP_TO_CHAR).
    from dace.libraries.tileops.nodes.tile_unop import _CAST_OP_TO_CPP, ExpandTileUnopPure
    if node.op in _CAST_OP_TO_CPP:
        return ExpandTileUnopPure.expansion(node, parent_state, parent_sdfg)
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


def make_mask_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP tasklet calling ``dace::tileops::tile_mask_gen`` (K=1 iteration mask).

    The K=1 mask is ``_o[l] = (iter_var + l) < global_ub``; ``iter_var`` and
    ``global_ub`` are the surrounding-scope symbol expressions, rendered inline
    exactly as :class:`ExpandTileMaskGenPure` does (e.g. ``i`` / ``kfdia``). An
    ``int64`` index keeps array-sized bounds exact. K=1 only -- the selector
    routes K>=2 (the per-dim AND conjunction) to ``pure``.
    """
    node.validate(parent_sdfg, parent_state)
    vlen = _require_k1(node)
    base = str(node.iter_vars[0])
    ub = str(node.global_ubs[0])
    call = f"dace::tileops::tile_mask_gen<int64_t, {vlen}>(_o, ({base}), ({ub}));"
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={},
        outputs={"_o": None},
        code=call,
        language=dace.dtypes.Language.CPP,
    )


def _has_replicate_gt1(node) -> bool:
    """True if any per-dim replicate factor is (or may be) > 1.

    A REPLICATE factor ``k > 1`` (e.g. ``c[i // 2]``) means lanes share a source
    element, which the linear ``tile_load`` / ``tile_gather`` intrinsics (no
    per-lane replicate divisor) cannot express -- the caller routes such a node
    to the ``pure`` expansion. A symbolic factor is treated as ``>1`` (can't
    prove it is 1 at compile time, so route safely).
    """
    for r in (node.replicate_factor_per_dim or []):
        try:
            if int(r) > 1:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _try_make_gather_tasklet(node, parent_state, parent_sdfg, suffix: str):
    """Emit ``tile_gather`` for the clean 1D unit-stride gather (``a[idx[i]]``).

    Only the canonical case lowers to the gather intrinsic: K=1, a single gather
    dim on a 1-D source with unit stride, an index tile that depends on the one
    tile lane (shape ``(W,)``), and no replicate. Then the per-lane source offset
    is exactly ``_idx_<g>[l]`` and the runtime ``tile_gather`` (AVX-512
    ``_mm512_i64gather_pd``; scalar reference on the other backends) applies
    directly. Anything more complex -- a multi-dim source, a non-unit gather-dim
    stride, a replicate factor, or a lane-independent index -- returns ``None``
    so the caller keeps the per-lane ``pure`` expansion (the scalar fallback).
    """
    widths = list(node.widths)
    if len(widths) != 1 or len(node.gather_dims) != 1 or _has_replicate_gt1(node):
        return None
    g = int(node.gather_dims[0])
    if g != 0:
        return None
    src_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == "_src")
    src_arr = parent_sdfg.arrays[src_edge.data.data]
    if len(src_arr.shape) != 1:
        return None
    # Unit stride on the gather dim: ``_idx_0`` is then the direct element offset
    # into ``_src`` (the intrinsic does ``src[idx[l]]`` with no stride multiply).
    if not bool(dace.symbolic.simplify(src_arr.strides[0] == 1)):
        return None
    idx_conn = f"_idx_{g}"
    idx_edge = next((e for e in parent_state.in_edges(node) if e.dst_conn == idx_conn), None)
    if idx_edge is None:
        return None
    idx_arr = parent_sdfg.arrays[idx_edge.data.data]
    # The index tile must depend on the single tile lane (shape ``(W,)``).
    try:
        idx_shape = tuple(int(s) for s in idx_arr.shape)
    except (TypeError, ValueError):
        return None
    if idx_shape != (int(widths[0]), ):
        return None
    vlen = widths[0]
    dst_dtype = _out_ctype(node, parent_state, parent_sdfg, "_dst")
    idx_ctype = idx_arr.dtype.ctype
    masked = "true" if node.has_mask else "false"
    mask_arg = "_mask" if node.has_mask else "nullptr"
    call = (f"dace::tileops::tile_gather<{dst_dtype}, {idx_ctype}, {vlen}, {masked}>"
            f"(_dst, _src, {idx_conn}, {mask_arg});")
    inputs = {"_src", idx_conn} | ({"_mask"} if node.has_mask else set())
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in inputs},
        outputs={"_dst": None},
        code=call,
        language=dace.dtypes.Language.CPP,
    )


def make_load_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP tasklet calling ``dace::tileops::tile_load`` (contiguous / strided).

    The K=1 tile-dim linear stride into the source array is passed as the
    ``stride`` argument; the header SIMD-loads when it is 1 (``_mm512_loadu_pd``)
    and uses the gather intrinsic (``_mm512_i64gather_pd`` over a strided index
    vector) otherwise.

    ``src_kind != 'Tile'`` (a broadcast ``Symbol`` literal or a ``Scalar``
    length-1 read) has no per-lane ``_src`` tile pointer, so the K=1 runtime
    ``tile_load`` (which streams ``_src`` into ``_dst``) cannot express it; the
    pure expansion broadcasts the literal / scalar to every lane instead.
    Delegate to it -- symmetric to the ``src_kind != 'Tile'`` fallback in
    :func:`make_store_tasklet`.

    When ``gather_dims`` is set the clean 1D unit-stride case (``a[idx[i]]``)
    lowers to the ``tile_gather`` intrinsic via :func:`_try_make_gather_tasklet`;
    any richer gather (multi-dim source, non-unit stride, replicate) delegates to
    :class:`ExpandTileLoadPure`, which emits the per-lane indirect read using the
    ``_idx_<d>`` connectors (design section 9.3).
    """
    if node.src_kind == "Tile" and node.gather_dims:
        gather_tasklet = _try_make_gather_tasklet(node, parent_state, parent_sdfg, suffix)
        if gather_tasklet is not None:
            return gather_tasklet
    if node.src_kind != "Tile" or node.gather_dims:
        from dace.libraries.tileops.nodes.tile_load import ExpandTileLoadPure
        return ExpandTileLoadPure.expansion(node, parent_state, parent_sdfg)
    # REPLICATE codegen (user direction 2026-06-10): the K=1 intrinsic
    # ``tile_load<T, VLEN, Masked>(_dst, _src, mask, stride)`` does
    # ``dst[i] = src[i * stride]`` -- no per-lane replicate divisor. When a
    # ``replicate_factor_per_dim[d] > 1`` (e.g. ``c[i // 2]`` -> factor 2 means
    # lanes 0,1 share c[i//2], ...), the intrinsic produces wrong values. Fall
    # back to the pure expansion (``src[(__l/replicate) * stride]`` per lane).
    if _has_replicate_gt1(node):
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
