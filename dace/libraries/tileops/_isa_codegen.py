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

from ._pure_codegen import nested_loops, tile_offset

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


def make_merge_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP tasklet calling ``dace::tileops::tile_merge`` (per-lane select).

    ``_cond`` / ``_t`` / ``_e`` are all tile operands (Broadcast=false);
    ``CondT`` is the condition tile's element type.
    """
    node.validate(parent_sdfg, parent_state)
    vlen = _require_k1(node)
    out_dtype = _out_ctype(node, parent_state, parent_sdfg, "_o")
    cond_dtype = _in_ctype(node, parent_state, parent_sdfg, "_cond")
    masked = "true" if node.has_mask else "false"
    mask_arg = "_mask" if node.has_mask else "nullptr"
    call = (f"dace::tileops::tile_merge<{out_dtype}, {cond_dtype}, {vlen}, false, false, {masked}>"
            f"(_o, _cond, _t, _e, {mask_arg});")
    inputs = {"_cond", "_t", "_e"} | ({"_mask"} if node.has_mask else set())
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in inputs},
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

    """
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
    """
    if node.src_kind != "Tile":
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


def _stride_is_one(stride_expr: str) -> bool:
    """True iff the array stride C++ expression is statically the integer 1."""
    return symstr(stride_expr).strip() == "1"


def _num_idx_conns(node) -> int:
    """Count the per-source-dim index connectors (``_idx_0`` ..)."""
    return sum(1 for c in node.in_connectors if str(c).startswith("_idx_"))


def _strided_lane(stride: int, off: str) -> str:
    """Per-lane index into a (possibly ``c``-strided) index/value tile.

    :param stride: The lane stride ``c`` into the tile.
    :param off: The contiguous lane-offset expression (e.g. ``__l0``).
    :returns: ``off`` for ``c == 1`` (contiguous), else ``(c) * (off)``.
    """
    return off if stride == 1 else f"({stride}) * ({off})"


def make_gather_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP tasklet for ``TileGather``.

    The node carries one index tile per source dim, so the per-lane source
    offset is ``sum_k _idx_<k>[lane] * src_strides[k]``. The header
    ``dace::tileops::tile_gather`` models a single linear index (``src[idx[i]]``),
    so it is used only for a 1D contiguous source (one index tile, unit stride) —
    the common SpMV-style gather. Any multi-dim source or non-unit stride emits
    the per-lane scalar read (same as ``pure``), which is still correct (and on a
    non-contiguous source no ISA has a usable gather form anyway).
    """
    node.validate(parent_sdfg, parent_state)
    vlen = _require_k1(node)
    src_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == "_src")
    src_arr = parent_sdfg.arrays[src_edge.data.data]
    src_ndim = _num_idx_conns(node)
    src_strides = [src_arr.strides[d] for d in range(len(src_arr.strides) - src_ndim, len(src_arr.strides))]
    dst_dtype = _out_ctype(node, parent_state, parent_sdfg, "_dst")
    masked = "true" if node.has_mask else "false"
    idx_strides = node.index_strides or [1] * src_ndim
    inputs = ({"_src"} | {f"_idx_{k}" for k in range(src_ndim)} | ({"_mask"} if node.has_mask else set()))
    off = tile_offset([vlen])

    # The ``tile_gather`` header reads ``_idx_0[lane]`` internally, so the
    # contiguous-source fast path only applies to a unit-stride index tile;
    # a ``c``-strided index window (``b[idx[c*i]]``) falls to the explicit
    # per-lane form reading ``_idx_0[c*lane]``.
    # Per-idx subscripting rule (mirror of ``tile_gather.ExpandTileGatherPure``
    # ``_idx_subscript``): a Scalar source / single-element memlet lowers to a
    # by-value scalar connector that CANNOT be subscripted (``int64_t _idx_0 =
    # z1_lc[0];``); emit the bare name. Cloudsc snippet-one's
    # ``zqx[z1, j+1, i+1]`` lands here (z1 is a loop-invariant scalar param).
    def _idx_subscript(k: int) -> str:
        ie = next(e for e in parent_state.in_edges(node) if e.dst_conn == f"_idx_{k}")
        src_desc = parent_sdfg.arrays.get(ie.data.data) if ie.data is not None else None
        if isinstance(src_desc, dace.data.Scalar):
            return ""
        try:
            lane_count = ie.data.subset.num_elements_exact() if ie.data and ie.data.subset else None
        except Exception:
            lane_count = None
        if lane_count == 1:
            return ""
        return f"[{_strided_lane(idx_strides[k], off)}]"

    if src_ndim == 1 and _stride_is_one(src_strides[0]) and idx_strides[0] == 1:
        idx_dtype = _in_ctype(node, parent_state, parent_sdfg, "_idx_0")
        mask_arg = "_mask" if node.has_mask else "nullptr"
        code = (f"dace::tileops::tile_gather<{dst_dtype}, {idx_dtype}, {vlen}, {masked}>"
                f"(_dst, _src, _idx_0, {mask_arg});")
    else:
        soff = " + ".join(f"((std::ptrdiff_t)_idx_{k}{_idx_subscript(k)} * ({symstr(src_strides[k])}))"
                          for k in range(src_ndim))
        if node.has_mask:
            code = nested_loops([vlen], f"_dst[{off}] = _mask[{off}] ? _src[{soff}] : {dst_dtype}(0);")
        else:
            code = nested_loops([vlen], f"_dst[{off}] = _src[{soff}];")
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in inputs},
        outputs={"_dst": None},
        code=code,
        language=dace.dtypes.Language.CPP,
    )


def make_scatter_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP tasklet for ``TileScatter`` (symmetric to :func:`make_gather_tasklet`)."""
    node.validate(parent_sdfg, parent_state)
    vlen = _require_k1(node)
    dst_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == "_dst")
    dst_arr = parent_sdfg.arrays[dst_edge.data.data]
    dst_ndim = _num_idx_conns(node)
    dst_strides = [dst_arr.strides[d] for d in range(len(dst_arr.strides) - dst_ndim, len(dst_arr.strides))]
    dst_dtype = dst_arr.dtype.ctype
    masked = "true" if node.has_mask else "false"
    inputs = ({"_src"} | {f"_idx_{k}" for k in range(dst_ndim)} | ({"_mask"} if node.has_mask else set()))
    off = tile_offset([vlen])

    # Same Scalar / single-element broadcast rule as the gather.
    def _idx_subscript(k: int) -> str:
        ie = next(e for e in parent_state.in_edges(node) if e.dst_conn == f"_idx_{k}")
        src_desc = parent_sdfg.arrays.get(ie.data.data) if ie.data is not None else None
        if isinstance(src_desc, dace.data.Scalar):
            return ""
        try:
            lane_count = ie.data.subset.num_elements_exact() if ie.data and ie.data.subset else None
        except Exception:
            lane_count = None
        if lane_count == 1:
            return ""
        return f"[{off}]"

    if dst_ndim == 1 and _stride_is_one(dst_strides[0]):
        idx_dtype = _in_ctype(node, parent_state, parent_sdfg, "_idx_0")
        mask_arg = "_mask" if node.has_mask else "nullptr"
        code = (f"dace::tileops::tile_scatter<{dst_dtype}, {idx_dtype}, {vlen}, {masked}>"
                f"(_dst, _src, _idx_0, {mask_arg});")
    else:
        doff = " + ".join(f"((std::ptrdiff_t)_idx_{k}{_idx_subscript(k)} * ({symstr(dst_strides[k])}))"
                          for k in range(dst_ndim))
        if node.has_mask:
            code = nested_loops([vlen], f"if (_mask[{off}]) _dst[{doff}] = _src[{off}];")
        else:
            code = nested_loops([vlen], f"_dst[{doff}] = _src[{off}];")
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in inputs},
        outputs={"_dst": None},
        code=code,
        language=dace.dtypes.Language.CPP,
    )
