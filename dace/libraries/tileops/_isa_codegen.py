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
import warnings
from typing import Tuple

import dace
from dace.sdfg import nodes
from dace.symbolic import symstr

# ISA backends shared by every tile-op node: (implementation key, class-name
# suffix, environment attribute on ``..environments``). Each backend exposes the
# SAME ``dace::tileops::tile_<op>`` signature and differs ONLY in the header its
# environment pulls in -- so a single factory builds every expansion class.
_ISA_BACKENDS = (
    ("scalar", "Scalar", "TileOpsScalar"),
    ("avx512", "AVX512", "TileOpsAVX512"),
    ("avx2", "AVX2", "TileOpsAVX2"),
    ("neon", "Neon", "TileOpsNeon"),
    ("sve", "SVE", "TileOpsSVE"),
    ("cuda", "CUDA", "TileOpsCUDA"),
    ("cuda_warp", "CUDAWarp", "TileOpsCUDAWarp"),
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
    mapping = {"_a": node.expr_a, "_b": getattr(node, "expr_b", None), "_c": getattr(node, "expr_c", None)}
    return [mapping.get(conn) for _kind, conn in conns]


def _scalar_ref(conn: str, desc, subset) -> str:
    """C++ reference for a Scalar/broadcast operand connector.

    DaCe passes a tasklet input connector by value (``T conn``) for ANY
    single-element access -- a true :class:`dace.data.Scalar`, a single-element
    read of a larger array (``a[j]`` -> ``double conn = a[j]``), AND a length-1
    array sliced at its sole element (``a[0]`` -> ``bool conn = a[0]``). Only a
    genuine MULTI-element source is staged to a pointer connector (``T* conn``)
    and must be dereferenced ``conn[0]``. The discriminator is the access's
    element COUNT, not the descriptor shape: a length-1 array read at ``[0]`` is
    by value, so keying on ``isinstance(desc, Array)`` faulted (``conn[0]`` on a
    by-value scalar).

    :param conn: Connector name.
    :param desc: Source array/scalar descriptor (unused; kept for call-site symmetry).
    :param subset: The connector edge's subset (the access into ``desc``).
    :returns: ``conn`` (by value) or ``conn[0]`` (multi-element array pointer).
    """
    return conn if subset.num_elements() == 1 else f"{conn}[0]"


def make_binop_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP tasklet calling ``dace::tileops::tile_binop`` for ``node``.

    Maps ``op`` -> :data:`_OP_TO_CHAR`; each operand's ``kind`` -> the broadcast
    boolean (Tile reads the per-lane connector with ``Broadcast=false``; Scalar /
    Symbol are materialised into a length-1 ``T`` buffer read with
    ``Broadcast=true``, casting to the output type); ``has_mask`` -> ``Masked``.
    """
    node.validate(parent_sdfg, parent_state)
    # A scalar-shaped output (a volume-1 Scalar / length-1 Array, emitted by value as
    # ``T _c;``) cannot bind to ``tile_binop``'s ``T* out`` pointer argument. Delegate
    # to the pure expansion's ``out_is_scalar`` branch (mirrors ``make_unop_tasklet``
    # and the differing-dtype deferral below). Gated on both operands non-Tile, matching
    # the pure path's own ``out_is_scalar`` predicate.
    from dace.libraries.tileops.nodes.tile_binop import ExpandTileBinopPure, _is_scalar_shape
    out_desc = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node) if e.src_conn == "_c").data.data]
    if node.kind_a != _TILE and node.kind_b != _TILE and _is_scalar_shape(out_desc):
        return ExpandTileBinopPure.expansion(node, parent_state, parent_sdfg)
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
        pre.append(f"const {out_dtype} {buf}[1] = {{ {val} }};")
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


def make_fma_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP tasklet calling ``dace::tileops::tile_fma`` for ``node`` (K=1).

    Fused multiply-add ``_o = _a * _b + _c`` with a single rounding. Each
    operand's ``kind`` -> the broadcast boolean (Tile reads the per-lane
    connector with ``Broadcast=false``; Scalar / Symbol are materialised into a
    length-1 ``T`` buffer read with ``Broadcast=true``, casting to the output
    type); ``has_mask`` -> ``Masked``. Mirrors :func:`make_binop_tasklet` with a
    third operand (``_c``) and the ``_o`` output connector (no op char).
    """
    node.validate(parent_sdfg, parent_state)
    # A scalar-shaped output (volume-1 Scalar / length-1 Array, emitted by value as
    # ``T _o;``) cannot bind to ``tile_fma``'s ``T* out`` pointer argument. Delegate
    # to the pure expansion's ``out_is_scalar`` branch (mirrors make_binop_tasklet).
    # Gated on all operands non-Tile, matching the pure path's own predicate.
    from dace.libraries.tileops.nodes.tile_fma import ExpandTileFMAPure
    from dace.libraries.tileops.nodes.tile_binop import _is_scalar_shape
    out_desc = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node) if e.src_conn == "_o").data.data]
    no_tile = node.kind_a != _TILE and node.kind_b != _TILE and node.kind_c != _TILE
    if no_tile and _is_scalar_shape(out_desc):
        return ExpandTileFMAPure.expansion(node, parent_state, parent_sdfg)
    vlen = _require_k1(node)
    in_e = {e.dst_conn: e for e in parent_state.in_edges(node) if e.dst_conn is not None}
    out_dtype = _out_ctype(node, parent_state, parent_sdfg, "_o")
    # The single-``T`` ISA runtime cannot carry an op whose operands and output
    # differ in type; defer to the ``pure`` expansion rather than emitting a
    # value-truncating C-style cast (user direction 2026-06-15).
    operand_ctype = _resolve_operand_ctype(node, parent_state, parent_sdfg, [(node.kind_a, "_a"), (node.kind_b, "_b"),
                                                                             (node.kind_c, "_c")], out_dtype)
    if operand_ctype != out_dtype:
        return ExpandTileFMAPure.expansion(node, parent_state, parent_sdfg)
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
        pre.append(f"const {out_dtype} {buf}[1] = {{ {val} }};")
        return "true", buf

    a_bcast, a_ptr = operand(node.kind_a, "_a", node.expr_a)
    b_bcast, b_ptr = operand(node.kind_b, "_b", node.expr_b)
    c_bcast, c_ptr = operand(node.kind_c, "_c", node.expr_c)
    masked = "true" if node.has_mask else "false"
    mask_arg = "_mask" if node.has_mask else "nullptr"
    call = (f"dace::tileops::tile_fma<{out_dtype}, {vlen}, "
            f"{a_bcast}, {b_bcast}, {c_bcast}, {masked}>(_o, {a_ptr}, {b_ptr}, {c_ptr}, {mask_arg});")
    inputs = set()
    if node.kind_a in (_TILE, _SCALAR):
        inputs.add("_a")
    if node.kind_b in (_TILE, _SCALAR):
        inputs.add("_b")
    if node.kind_c in (_TILE, _SCALAR):
        inputs.add("_c")
    if node.has_mask:
        inputs.add("_mask")
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in inputs},
        outputs={"_o": None},
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
    # A scalar-shaped output (``_c`` a volume-1 Scalar / length-1 Array, which codegen
    # emits by value as ``T _c;``) cannot bind to ``tile_unop``'s ``T* out`` pointer
    # argument -- e.g. ``!`` of a scalar bool condition (s274). The pure expansion's
    # ``out_is_scalar`` branch emits the bare ``_c = <op>(...)`` assignment instead;
    # delegate to it (the same escape hatch the cast-op case above and the
    # differing-dtype case below use). Gated on a non-Tile operand to mirror the pure
    # path's own ``out_is_scalar`` predicate (a Tile operand always yields a tile output).
    from dace.libraries.tileops.nodes.tile_binop import _is_scalar_shape
    out_desc = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node) if e.src_conn == "_c").data.data]
    if node.kind_a != _TILE and _is_scalar_shape(out_desc):
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
        pre.append(f"const {out_dtype} {a_ptr}[1] = {{ {val} }};")
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
    in_e = {e.dst_conn: e for e in parent_state.in_edges(node) if e.dst_conn is not None}
    pre = []

    def arm(kind, conn, expr):
        """An arm lowers to a per-lane Tile read (``Broadcast=false``, via its
        connector) or a length-1 broadcast buffer (``Broadcast=true``) for a
        Scalar source / inline Symbol expression -- mirrors ``make_binop_tasklet``.
        Returns ``(broadcast_flag, ptr_expr)``."""
        if kind == _TILE:
            return "false", conn
        if kind == _SYMBOL:
            val = f"({out_dtype})({expr})"
        else:  # _SCALAR
            desc = parent_sdfg.arrays[in_e[conn].data.data]
            val = f"({out_dtype})({_scalar_ref(conn, desc, in_e[conn].data.subset)})"
        buf = f"_bc{conn}"
        pre.append(f"const {out_dtype} {buf}[1] = {{ {val} }};")
        return "true", buf

    t_bcast, t_ptr = arm(node.kind_t, "_t", node.expr_t)
    e_bcast, e_ptr = arm(node.kind_e, "_e", node.expr_e)
    call = (f"dace::tileops::tile_ite<{out_dtype}, {cond_dtype}, {vlen}, "
            f"{t_bcast}, {e_bcast}, false>(_o, _mask, {t_ptr}, {e_ptr}, nullptr);")
    # A Symbol arm embeds its expression inline (no connector); only Tile / Scalar
    # arms read through a connector. ``_mask`` is always a connector (the select
    # predicate is materialised to a tile by ``_convert_ite``).
    inputs = {"_mask"}
    if node.kind_t in (_TILE, _SCALAR):
        inputs.add("_t")
    if node.kind_e in (_TILE, _SCALAR):
        inputs.add("_e")
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in inputs},
        outputs={"_o": None},
        code="\n".join(pre + [call]),
        language=dace.dtypes.Language.CPP,
    )


# Byte alignment the allocator guarantees for the base of an array, per storage class. The GPU
# figure is the gpuMalloc/gpuMallocAsync contract (256 B). Anything else is unknown and gets no
# assumption. Register is NOT here on purpose -- see :func:`base_align_bytes`.
_BASE_ALIGN_BYTES = {
    dace.dtypes.StorageType.GPU_Global: 256,
    dace.dtypes.StorageType.CPU_Pinned: 256,
}


def base_align_bytes(arr) -> int:
    """Bytes the allocation guarantees for ``arr``'s base address.

    A Register array is a declared local, so its guarantee is whatever the descriptor asked the
    codegen to emit -- ``alignment == 0`` means no attribute at all, i.e. only the element's natural
    alignment. Reading the descriptor rather than assuming the old blanket ``DACE_ALIGN(64)`` is
    what keeps the widened reinterpret below a PROVEN fact: over-promising here is a misaligned
    device access, which faults, not merely slow code.
    """
    if arr.storage == dace.dtypes.StorageType.Register:
        return arr.alignment if arr.alignment > 0 else arr.dtype.bytes
    return _BASE_ALIGN_BYTES.get(arr.storage, 0)


def _enclosing_param_ranges(node, parent_state, parent_sdfg) -> Tuple[dict, list]:
    """``({param: (start, end, step)}, [(extent, width)])`` for the map scopes enclosing ``node``.

    Walks out through nested-SDFG boundaries, rewriting each level's params through the
    ``symbol_mapping`` that connects them, so a param of an OUTER map is reported under the name
    the INNER offset expression uses. A param whose range cannot be followed is simply absent,
    which leaves it un-substituted and therefore not provably divisible.

    The second list is the extents the vectorizer GUARANTEES divisible: a ``__tile_main`` map's
    tiled dim either had its end tightened to a whole number of tiles (the peeled-remainder path)
    or is the caller's ``assume_even`` promise, which that pass raises on when provably violated
    and otherwise guards with a host-side abort. Nothing here re-derives the guarantee; it reads
    the marker the guarantee is recorded under.
    """
    from dace.sdfg.sdfg import SDFG
    from dace.transformation.passes.vectorization.split_map_for_tile_remainder import TILE_MAIN_MARKER

    ranges, even, cur_node, state, sdfg = {}, [], node, parent_state, parent_sdfg
    while state is not None:
        entry = state.entry_node(cur_node)
        while entry is not None:
            for param, rng in zip(entry.map.params, entry.map.range):
                ranges.setdefault(str(param), (rng[0], rng[1], rng[2]))
                step = dace.symbolic.simplify(rng[2])
                # Only a TILED dim (step == its width > 1) carries a guarantee; a step-1 dim's
                # extent divides 1 and says nothing.
                if entry.map.label.endswith(TILE_MAIN_MARKER) and step.is_Integer and step > 1:
                    even.append((rng[1] - rng[0] + 1, int(step)))
            entry = state.entry_node(entry)
        nsdfg_node = sdfg.parent_nsdfg_node
        if nsdfg_node is None:
            break
        # Names change across the boundary: an inner symbol equal to a bare outer symbol can keep
        # that outer map's range, anything richer is not followed.
        renamed = {}
        for inner, outer in nsdfg_node.symbol_mapping.items():
            outer_sym = str(dace.symbolic.pystr_to_symbolic(outer))
            if outer_sym in ranges and str(inner) != outer_sym:
                renamed[str(inner)] = ranges[outer_sym]
        ranges.update(renamed)
        cur_node, state = nsdfg_node, sdfg.parent
        sdfg = sdfg.parent_sdfg
        if not isinstance(sdfg, SDFG):
            break
    return ranges, even


def _even_extent_substitutions(even: list) -> dict:
    """``{symbol: width*t - b}`` for every guaranteed-divisible extent of the form ``symbol + b``.

    This is what makes a SYMBOLIC row stride usable. An ``N``-column array tiled over ``1:N-1``
    has extent ``N - 2``, and a width-2 tile guarantees that is even, so ``N = 2t + 2`` -- which
    makes ``N*i`` provably even and a ``A[i, j+1]`` offset provably ODD instead of unknown. Without
    it every multi-dim access on a symbolic shape stays on the per-element path.

    ``t`` is non-negative because a tiled map runs at least one full tile. Only a bare ``symbol +
    constant`` extent is solved: a richer one (``N*M - 1``) has no single symbol to pin and is left
    alone, which simply leaves the offset undecidable.
    """
    subs = {}
    # Widest first: two tiled dims can constrain the same symbol, and the wider one is stronger.
    for n, (extent, width) in enumerate(sorted(even, key=lambda ew: -ew[1])):
        extent = dace.symbolic.simplify(extent)
        free = list(extent.free_symbols)
        if len(free) != 1 or free[0] in subs:
            continue
        sym = free[0]
        rest = dace.symbolic.simplify(extent - sym)
        if not rest.is_Integer:
            continue
        t = dace.symbolic.symbol(f"__dace_align_t{n}", nonnegative=True, integer=True)
        subs[sym] = width * t - int(rest)
    return subs


def _guarded_stride_divisors(parent_sdfg) -> dict:
    """``{symbol: modulus}`` for the stride-parity facts guarded anywhere up ``parent_sdfg``'s chain.

    The facts are read off the guard tasklets ``SplitMapForTileRemainder`` emitted, so every entry
    here is backed by a host-side ``abort`` that runs before the kernel: the proof below widens an
    access only on divisibility the program itself verifies. No guard -> no entry -> the symbolic
    stride stays undecidable and the caller keeps the per-element path.

    Guards live on the SDFG that owns the tiled MAP, which is usually an ancestor of the one that
    owns the tile ops, hence the walk. A name a NestedSDFG boundary REBINDS (``symbol_mapping``
    sends a different outer expression to it) means something else on the two sides, so it is
    refused from there outward rather than translated -- a wrong translation would be exactly the
    unchecked promise this is built to avoid.
    """
    from dace.sdfg.sdfg import SDFG
    from dace.transformation.passes.vectorization.split_map_for_tile_remainder import guarded_stride_divisors

    facts, sdfg, rebound = {}, parent_sdfg, set()
    while isinstance(sdfg, SDFG):
        for name, modulus in guarded_stride_divisors(sdfg).items():
            if name not in rebound:
                facts.setdefault(name, modulus)
        nsdfg_node = sdfg.parent_nsdfg_node
        if nsdfg_node is None:
            break
        for inner, outer in nsdfg_node.symbol_mapping.items():
            outer_name = str(dace.symbolic.pystr_to_symbolic(outer))
            if outer_name != str(inner):
                rebound.add(str(inner))
                rebound.add(outer_name)
        sdfg = sdfg.parent_sdfg
    return facts


def _base_offset_is_visible(sdfg, name: str) -> bool:
    """True if the memlet offset seen INSIDE ``sdfg`` is the whole offset from the allocation.

    A nested SDFG is handed a pointer that its enclosing memlet may already have shifted -- the
    body gets ``&A[N*i]`` and then indexes it by ``j`` alone. Reading only the inner subset would
    then miss the ``N*i`` and claim an alignment the pointer does not have, so every enclosing
    connection for this array has to start at element 0 for the inner view to be the whole story.
    """
    cur_sdfg, cur_name = sdfg, name
    while cur_sdfg is not None:
        nsdfg_node = cur_sdfg.parent_nsdfg_node
        state = cur_sdfg.parent
        if nsdfg_node is None or state is None:
            return True
        edges = ([e for e in state.in_edges(nsdfg_node) if e.dst_conn == cur_name] +
                 [e for e in state.out_edges(nsdfg_node) if e.src_conn == cur_name])
        if not edges:
            return False
        outer_names = set()
        for e in edges:
            if e.data.subset is None:
                return False
            if any(dace.symbolic.simplify(s) != 0 for s in e.data.subset.min_element()):
                return False
            outer_names.add(e.data.data)
        if len(outer_names) != 1:
            return False
        cur_sdfg, cur_name = cur_sdfg.parent_sdfg, outer_names.pop()
    return True


def _linear_base_offset(node, parent_state, parent_sdfg, edge):
    """``(offset_at_tile_base, offset_at_param_ends, allocated_elements)``, or ``None``.

    The first expression has every enclosing map param replaced by ``start + step*k`` (``k`` a
    fresh non-negative symbol), which is what makes a residue modulo a chunk decidable. The second
    puts every param at the END of its range, which upper-bounds the offset -- used to prove a
    widened window stays inside the allocation. A param whose coefficient is not provably
    non-negative makes the end substitution no bound at all, so the whole thing is refused.

    All three carry the guaranteed-divisible-extent rewrite (:func:`_even_extent_substitutions`),
    which is what a symbolic shape needs to decide anything at all -- and which the size has to
    carry too, or the bound is compared against the wrong allocation.
    """
    arr = parent_sdfg.arrays[edge.data.data]
    # A view starts wherever its source says, and a start_offset shifts the base out from under
    # the allocator's guarantee.
    if isinstance(arr, dace.data.View) or arr.start_offset != 0:
        return None
    if not _base_offset_is_visible(parent_sdfg, edge.data.data):
        return None
    subset = edge.data.subset
    if subset is None:
        return None
    offset = dace.symbolic.pystr_to_symbolic(sum(s * st for s, st in zip(subset.min_element(), arr.strides)))
    ranges, even = _enclosing_param_ranges(node, parent_state, parent_sdfg)
    at_base, at_end = offset, offset
    # Match params against the symbols the offset ACTUALLY holds, by name. A freshly built
    # ``pystr_to_symbolic(param)`` carries the default int32 dtype while the subset's symbol is
    # int64, and a DaCe symbol hashes on its dtype -- so ``sym in offset.free_symbols`` is silently
    # False and every param stays un-substituted, which leaves the residue undecidable for reasons
    # that have nothing to do with the shape.
    by_name = {str(s): s for s in offset.free_symbols}
    size = dace.symbolic.pystr_to_symbolic(arr.total_size)
    # Built BEFORE the coefficient test, not after: a stride of ``N - 2`` (the shape a transient
    # holding a stencil's interior gets) is not provably non-negative as written, while the very
    # extent fact that makes its parity decidable, ``N = 2t + 2``, also makes it ``2t``. Testing
    # the raw coefficient refused every such transient for a reason the substitution answers.
    subs = _even_extent_substitutions(even)
    _add_stride_substitutions(subs, _guarded_stride_divisors(parent_sdfg), (offset, size))
    for idx, (param, (start, end, step)) in enumerate(ranges.items()):
        sym = by_name.get(param)
        if sym is None:
            continue
        coeff = dace.symbolic.simplify(dace.symbolic.equalize_symbol(offset.diff(sym).subs(subs)))
        if coeff.is_nonnegative is not True:
            return None
        k = dace.symbolic.symbol(f"__dace_align_k{idx}", nonnegative=True, integer=True)
        start, end, step = (dace.symbolic.pystr_to_symbolic(x) for x in (start, end, step))
        at_base = at_base.subs(sym, start + step * k)
        at_end = at_end.subs(sym, end)
    return tuple(e.subs(subs) for e in (at_base, at_end, size))


def _add_stride_substitutions(subs: dict, facts: dict, exprs) -> None:
    """Fold the guarded stride-parity facts into ``subs`` as ``symbol -> modulus * t``.

    Same shape as :func:`_even_extent_substitutions` and for the same reason: a residue modulo a
    chunk is only decidable once the symbol carrying the row stride is written as a multiple of it.
    ``N -> 2*t`` is what turns heat3d's ``N**2*i + N*j + k`` from an unknown parity into a KNOWN
    odd one -- which is a widened load with a one-element shift, not a refusal.

    ``t`` is POSITIVE, not merely non-negative, because the guard checks ``stride >= modulus`` as
    well as the divisibility; that is what proves the widened window still ends inside the
    allocation. An extent fact already pinning the symbol wins: it carries an offset too, so it is
    strictly the stronger statement.
    """
    pinned = {str(s) for s in subs}
    by_name = {}
    for expr in exprs:
        for sym in expr.free_symbols:
            by_name.setdefault(str(sym), sym)
    for n, (name, modulus) in enumerate(sorted(facts.items())):
        sym = by_name.get(name)
        if sym is None or name in pinned:
            continue
        subs[sym] = modulus * dace.symbolic.symbol(f"__dace_align_s{n}", positive=True, integer=True)


def _declined(arr, edge, elem_bytes: int, allow_shift: bool) -> Tuple[int, int]:
    """The per-element result, plus the reason a SUB-32-bit access had to take it.

    Silence here is expensive and invisible: fp16 is a CLIFF, not a slope. Two elements make one
    32-bit word, which is both the minimum vector and the minimum aligned load, so an access the
    proof cannot place on an even element has NOTHING between ``half2`` and a per-element
    ``LDG.E.U16`` -- roughly six scalar loads where one wide one would do. A 4-byte or wider dtype
    merely degrades a step (128 -> 64 -> 32 bit) and is left unreported.

    The actionable half is the shape: every dimension of a sub-32-bit array has to be a multiple of
    the elements-per-word, or the row start's parity alternates and no access off it is decidable.

    Only the SHIFT-eligible side (the loads) reports. A store never widens off a boundary by
    design -- the widened word covers neighbours it must not overwrite -- so warning there would
    fire on every correct program.
    """
    if elem_bytes < 4 and allow_shift:
        warnings.warn(f'"{edge.data.data}" ({arr.dtype}) is loaded/stored per element: its access could not be '
                      f'placed on a {4 // elem_bytes}-element boundary. Sub-32-bit arrays need EVERY dimension '
                      f'to be a multiple of {4 // elem_bytes} for the vectorized path -- pad the shape, or pin '
                      f'the symbolic extents so their parity is decidable.')
    return elem_bytes, 0


def _array_align_shift(node, parent_state, parent_sdfg, edge, vlen: int, allow_shift: bool) -> Tuple[int, int]:
    """``(alignment bytes of the aligned base, element shift of the access from it)``.

    The tile side of a load/store is always DACE_ALIGN(64); the array side is a base pointer plus
    the memlet's linear element offset, so it is only as aligned as that offset is divisible.

    When the offset IS divisible the access itself is aligned and the shift is 0 -- the case
    ``c13a93c97`` already handles. When it is not, a ``±1`` stencil neighbour being the canonical
    example, the residue is often still a compile-time constant: ``A[i, j, k+1]`` on a 128-column
    array offsets by ``16384*i + 128*j + k + 16513`` with ``k`` even, which is odd for every
    iteration. Reporting that residue lets the caller load the ALIGNED window below the access and
    pick the elements out in registers, instead of declining to widen at all.

    The window reads ``chunk - shift`` elements past the tile, so the access must not be the last
    one in the allocation; ``at_end`` bounds that. It reads nothing BELOW element 0: the aligned
    base is ``offset - shift`` with ``offset >= shift`` by construction of the residue.

    A symbolic row stride is what defeats both: ``A[i, j]`` on an ``N``-column array offsets by
    ``N*i + j``, whose parity is unknown, so neither a divisibility nor a residue is decidable and
    the caller keeps the scalar path. Returns the element size and shift 0 when nothing is provable.
    """
    arr = parent_sdfg.arrays[edge.data.data]
    elem_bytes = arr.dtype.bytes
    base_bytes = base_align_bytes(arr)
    if base_bytes < elem_bytes:
        return _declined(arr, edge, elem_bytes, allow_shift)
    offsets = _linear_base_offset(node, parent_state, parent_sdfg, edge)
    if offsets is None:
        return _declined(arr, edge, elem_bytes, allow_shift)
    at_base, at_end, size = offsets
    for chunk in (8, 4, 2):
        if chunk * elem_bytes > base_bytes:
            continue
        if dace.symbolic.simplify(at_base % chunk) == 0:
            return chunk * elem_bytes, 0
    # One 32-bit word is the granularity the register extraction works at (__byte_perm), so the
    # window is chunked at that width and the shift has to leave the used elements inside two
    # consecutive words -- guaranteed by chunk | vlen.
    chunk = 4 // elem_bytes
    if not allow_shift or chunk < 2 or vlen % chunk != 0 or chunk * elem_bytes > base_bytes:
        return _declined(arr, edge, elem_bytes, allow_shift)
    shift = dace.symbolic.simplify(at_base % chunk)
    if not shift.is_Integer:
        return _declined(arr, edge, elem_bytes, allow_shift)
    if dace.symbolic.simplify(at_end + vlen + chunk - int(shift) - size).is_nonpositive is not True:
        return _declined(arr, edge, elem_bytes, allow_shift)
    return chunk * elem_bytes, int(shift)


def _align_template_arg(node,
                        parent_state,
                        parent_sdfg,
                        edge,
                        backend: str,
                        vlen: int,
                        allow_shift: bool = False) -> str:
    """``", <bytes>"`` or ``", <bytes>, <shift>"`` for a tile_load/tile_store template list.

    Only the CUDA header takes the trailing ``Align`` / ``Shift`` parameters, and only fp16 has a
    widened path behind them, so every other backend and dtype keeps the exact 3-argument call it
    emitted before.

    ``allow_shift`` is the load/store asymmetry: a load may read the aligned window around its
    elements and discard the extras, a store may not write them -- that would clobber the
    neighbours the widened word covers -- so a shifted store keeps the per-element loop.

    ``cuda_warp`` is excluded even though it reuses this header: there a tile is split across the
    lanes of a warp and each lane loads its own fragment, so the per-lane base is not the memlet
    offset this analysis reads and its alignment has not been established.
    """
    if backend != "cuda":
        return ""
    arr = parent_sdfg.arrays[edge.data.data]
    if arr.dtype != dace.float16:
        return ""
    align, shift = _array_align_shift(node, parent_state, parent_sdfg, edge, vlen, allow_shift)
    if shift:
        return f", {align}, {shift}"
    return f", {align}" if align > arr.dtype.bytes else ""


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
    # A non-dividing factor (``c[i // 3]`` with ``W % 3 != 0``, or a symbolic
    # divisor) likewise routes here: ``_has_replicate_gt1`` treats both as >1,
    # and the pure expansion emits the phase-aware per-lane index.
    if _has_replicate_gt1(node):
        from dace.libraries.tileops.nodes.tile_load import ExpandTileLoadPure
        return ExpandTileLoadPure.expansion(node, parent_state, parent_sdfg)
    vlen = _require_k1(node)
    src_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == "_src")
    dst_dtype = _out_ctype(node, parent_state, parent_sdfg, "_dst")
    stride = _k1_array_stride(node, parent_sdfg, src_edge, node.src_dims)
    masked = "true" if node.has_mask else "false"
    mask_arg = "_mask" if node.has_mask else "nullptr"
    align = _align_template_arg(node, parent_state, parent_sdfg, src_edge, suffix, vlen, allow_shift=True)
    call = f"dace::tileops::tile_load<{dst_dtype}, {vlen}, {masked}{align}>(_dst, _src, {mask_arg}, {stride});"
    inputs = {"_src"} | ({"_mask"} if node.has_mask else set())
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in inputs},
        outputs={"_dst": None},
        code=call,
        language=dace.dtypes.Language.CPP,
    )


def make_reduce_tasklet(node, parent_state, parent_sdfg, suffix: str) -> nodes.Tasklet:
    """CPP/CUDA tasklet calling ``dace::tileops::tile_reduce`` (full K=1 horizontal reduce).

    The ``tile_reduce`` intrinsic collapses a VLEN-lane tile to ONE scalar (an
    in-map / per-tile reduction ``acc = sum/prod/min/max over the tile``). Every
    backend header ships its own self-contained lowering (no shared dispatch
    header): AVX-512 accumulates the lane groups then collapses with the one-shot
    ``_mm512_reduce_<op>_p{s,d}``; scalar / avx2 / neon / sve use a portable
    balanced log-depth tree over the header's own ``tile_apply``; and the CUDA
    fp16 path folds pairwise via ``__hadd2`` / ... then combines the 2 surviving
    lanes into one ``__half`` (there is no native "reduce half2 -> half"
    intrinsic), with the generic CUDA template keeping the per-lane fold for every
    other element type. Only a shape the intrinsic cannot express -- a single-axis
    reduce (``axis is not None``), a mask, or K>=2 -- delegates to
    :class:`ExpandTileReducePure`.
    """
    from dace.libraries.tileops.nodes.tile_reduce import ExpandTileReducePure
    if node.axis is not None or node.has_mask or len(node.widths) != 1:
        return ExpandTileReducePure.expansion(node, parent_state, parent_sdfg)
    vlen = _require_k1(node)
    src_dtype = _in_ctype(node, parent_state, parent_sdfg, "_src")
    op_char = _OP_TO_CHAR[node.op]
    # Full reduction -> length-1 output. DaCe emits a volume-1 output by value
    # (``T _dst;``) so it is referenced bare; a genuine multi-element pointer
    # target is dereferenced ``_dst[0]`` (mirrors ExpandTileReducePure.writeback).
    out_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == "_dst")
    scalar_dst = out_edge.data.subset is None or out_edge.data.subset.num_elements() == 1
    dst_ref = "_dst" if scalar_dst else "_dst[0]"
    call = f"{dst_ref} = dace::tileops::tile_reduce<{src_dtype}, {vlen}, '{op_char}'>(_src);"
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={"_src": None},
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
    align = _align_template_arg(node, parent_state, parent_sdfg, dst_edge, suffix, vlen)
    call = f"dace::tileops::tile_store<{dst_dtype}, {vlen}, {masked}{align}>(_dst, _src, {mask_arg}, {stride});"
    inputs = {"_src"} | ({"_mask"} if node.has_mask else set())
    return nodes.Tasklet(
        label=f"{node.label}_{suffix}",
        inputs={c: None
                for c in inputs},
        outputs={"_dst": None},
        code=call,
        language=dace.dtypes.Language.CPP,
    )
