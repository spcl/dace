# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileBinop`` element-wise binary op on K-dim register tiles.

The lib node consumes two operands ``_a`` and ``_b`` and writes ``_c``.
Each operand carries a ``kind`` flag — ``Tile`` (a tile-shape array via
a connector) or ``Symbol`` (a free-symbol expression embedded inline in
the tasklet body). At least one operand must be ``Tile``; a Symbol /
Symbol pair belongs outside the tile path.

The pure expansion returns a CPP tasklet whose body is a single
``for``-loop over the flattened tile (correctness-only).
"""
from typing import Optional, Tuple

import numpy as np

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .._pure_codegen import nested_loops, tile_offset
from .. import _isa_codegen
from ..environments import TileOpsScalar, TileOpsAVX512, TileOpsAVX2, TileOpsNeon, TileOpsSVE


def _promotion_ok(src: dace.dtypes.typeclass, dst: dace.dtypes.typeclass) -> bool:
    """Whether a Tile operand of dtype ``src`` may be promoted to the output
    dtype ``dst`` before the op (a widening conversion).

    Allowed (widening): same dtype; integer -> wider-or-equal integer; integer
    -> float / double; float -> double. Disallowed (narrowing -> the caller must
    crash): float / double -> integer; double -> float; integer narrowing
    (e.g. int64 -> int32).

    :param src: The Tile operand's element dtype.
    :param dst: The output (``_c``) element dtype.
    :returns: ``True`` iff promoting ``src`` to ``dst`` is non-narrowing.
    """
    if src == dst:
        return True
    s_int = np.issubdtype(src.type, np.integer)
    d_int = np.issubdtype(dst.type, np.integer)
    s_flt = np.issubdtype(src.type, np.floating)
    d_flt = np.issubdtype(dst.type, np.floating)
    if s_int and d_flt:  # int -> float / double
        return True
    if s_int and d_int and dst.bytes >= src.bytes:  # integer widening
        return True
    if s_flt and d_flt and dst.bytes >= src.bytes:  # float -> double
        return True
    return False


_TILE = "Tile"
_SYMBOL = "Symbol"
_SCALAR = "Scalar"
_VALID_KINDS = (_TILE, _SYMBOL, _SCALAR)

_OP_CPP = {
    "+": ("(", " + ", ")"),
    "-": ("(", " - ", ")"),
    "*": ("(", " * ", ")"),
    "/": ("(", " / ", ")"),
    "%": ("(", " % ", ")"),
    "<": ("(", " < ", ")"),
    "<=": ("(", " <= ", ")"),
    ">": ("(", " > ", ")"),
    ">=": ("(", " >= ", ")"),
    "==": ("(", " == ", ")"),
    "!=": ("(", " != ", ")"),
    "&&": ("(", " && ", ")"),
    "||": ("(", " || ", ")"),
    "&": ("(", " & ", ")"),
    "|": ("(", " | ", ")"),
    "^": ("(", " ^ ", ")"),
    "min": ("std::min(", ", ", ")"),
    "max": ("std::max(", ", ", ")"),
}


def _binop_rhs(op: str, lhs: str, rhs: str) -> str:
    """Render the C++ expression for ``op`` on ``lhs`` and ``rhs``.

    :param op: The operator symbol (key of :data:`_OP_CPP`).
    :param lhs: Left-hand-side C++ expression.
    :param rhs: Right-hand-side C++ expression.
    :returns: A C++ expression string.
    """
    pre, sep, post = _OP_CPP[op]
    return f"{pre}{lhs}{sep}{rhs}{post}"


@library.expansion
class ExpandTileBinopPure(ExpandTransformation):
    """Correctness-only CPP tasklet lowering of ``TileBinop``."""

    environments = []

    @staticmethod
    def expansion(node: "TileBinop", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a single CPP tasklet that walks the flattened tile.

        :param node: The ``TileBinop`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A CPP tasklet replacing the lib node in place.
        """
        node.validate(parent_sdfg, parent_state)
        widths = list(node.widths)
        off = tile_offset(widths)
        in_e = {e.dst_conn: e for e in parent_state.in_edges(node) if e.dst_conn is not None}

        out_dtype = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node)
                                            if e.src_conn == "_c").data.data].dtype.ctype

        def _operand_ref(kind, conn, expr):
            """Return the per-lane C++ reference for one operand.

            A ``_SYMBOL`` / ``_SCALAR`` operand is cast to ``out_dtype`` so
            ``std::min`` / ``std::max`` (and any other type-strict overload) sees
            both operands at the same type — without this cast a ``min`` between
            an int literal ``(1)`` and a ``double _b[off]`` fails to resolve
            (``no matching function for call to 'min(int, double&)'``). The
            ``_TILE`` operand's dtype already matches ``out_dtype`` (validate()
            enforces it).
            """
            if kind == _SYMBOL:
                return f"({out_dtype})({expr})"
            if kind == _TILE:
                return f"{conn}[{off}]"
            # Scalar broadcast operand. DaCe passes a tasklet connector by
            # value (``T conn``) for a true ``dace.data.Scalar`` and for a
            # single-element access into a larger array (``a[j]`` ->
            # ``double conn = a[j]``); only a genuine length-1 *array*
            # connector is a pointer (``T* conn``) needing ``conn[0]``. Cast
            # to ``out_dtype`` so a typed binop (``std::min`` etc.) resolves.
            desc = parent_sdfg.arrays[in_e[conn].data.data]
            is_len1_array = (isinstance(desc, dace.data.Array)
                             and all(bool(dace.symbolic.simplify(s == 1)) for s in desc.shape))
            ref = f"{conn}[0]" if is_len1_array else conn
            return f"({out_dtype})({ref})"

        lhs = _operand_ref(node.kind_a, "_a", node.expr_a)
        rhs = _operand_ref(node.kind_b, "_b", node.expr_b)
        rhs_expr = _binop_rhs(node.op, lhs, rhs)
        if node.has_mask:
            body = f"_c[{off}] = _mask[{off}] ? ({rhs_expr}) : {out_dtype}(0);"
        else:
            body = f"_c[{off}] = {rhs_expr};"
        code = nested_loops(widths, body)
        inputs = {"_a", "_b", "_mask"}
        if node.kind_a == _SYMBOL:
            inputs.discard("_a")
        if node.kind_b == _SYMBOL:
            inputs.discard("_b")
        if not node.has_mask:
            inputs.discard("_mask")
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={c: None
                    for c in inputs},
            outputs={"_c": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


_CUTE_OP_EXPR = {
    "+": "{lhs} + {rhs}",
    "-": "{lhs} - {rhs}",
    "*": "{lhs} * {rhs}",
    "/": "{lhs} / {rhs}",
    "%": "{lhs} % {rhs}",
    "<": "{lhs} < {rhs}",
    "<=": "{lhs} <= {rhs}",
    ">": "{lhs} > {rhs}",
    ">=": "{lhs} >= {rhs}",
    "==": "{lhs} == {rhs}",
    "!=": "{lhs} != {rhs}",
    "&&": "{lhs} & {rhs}",
    "||": "{lhs} | {rhs}",
    "&": "{lhs} & {rhs}",
    "|": "{lhs} | {rhs}",
    "^": "{lhs} ^ {rhs}",
    "min": "ct.minimum({lhs}, {rhs})",
    "max": "ct.maximum({lhs}, {rhs})",
}


@library.expansion
class ExpandTileBinopCutile(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileBinop`.

    Emits the bare element-wise expression (e.g. ``a_tile + b_tile``)
    — matches the reference cuTile kernels, where the mask is applied
    at the ``ct.scatter`` store, not at the binop. Symbol-kind operands
    are embedded inline. ``min`` / ``max`` route to ``ct.minimum`` /
    ``ct.maximum``.
    """

    environments = []

    @staticmethod
    def expansion(node: "TileBinop", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet emitting the cuTile binop.

        :param node: The lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet with the element-wise body.
        """

        def _cutile_operand(kind, tile_conn, scalar_conn, expr):
            """cuTile operand reference: inline expr for Symbol, the
            tile connector for Tile, the scalar connector (broadcasts
            NumPy-style) for Scalar."""
            if kind == _SYMBOL:
                return expr
            if kind == _TILE:
                return tile_conn
            return scalar_conn

        lhs = _cutile_operand(node.kind_a, "__rhs1", "__const1", node.expr_a)
        rhs = _cutile_operand(node.kind_b, "__rhs2", "__const2", node.expr_b)
        rhs_expr = _CUTE_OP_EXPR[node.op].format(lhs=lhs, rhs=rhs)
        body = f"__output = {rhs_expr}"
        inputs = set()
        if node.kind_a == _TILE:
            inputs.add("__rhs1")
        elif node.kind_a == _SCALAR:
            inputs.add("__const1")
        if node.kind_b == _TILE:
            inputs.add("__rhs2")
        elif node.kind_b == _SCALAR:
            inputs.add("__const2")
        return nodes.Tasklet(
            label=f"{node.label}_cutile",
            inputs={c: None
                    for c in inputs},
            outputs={"__output": None},
            code=body,
            language=dace.dtypes.Language.Python,
        )


@library.expansion
class ExpandTileBinopScalar(ExpandTransformation):
    """K=1 scalar-backend lowering: a call to ``dace::tileops::tile_binop`` in
    ``dace/tile_ops/scalar.h`` (pulled in via :class:`TileOpsScalar`).

    The shared :func:`~dace.libraries.tileops._isa_codegen.make_binop_tasklet`
    maps ``op`` -> a one-char op code, each operand's ``kind`` -> a broadcast
    boolean, and ``has_mask`` -> the ``Masked`` boolean. K=1 only (the selector
    routes K>=2 to ``pure``). The avx512 / avx2 / neon / sve siblings emit the
    identical call and differ only in the backend header their environment pulls
    in.
    """

    environments = [TileOpsScalar]

    @staticmethod
    def expansion(node: "TileBinop", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _isa_codegen.make_binop_tasklet(node, parent_state, parent_sdfg, "scalar")


@library.expansion
class ExpandTileBinopAVX512(ExpandTransformation):
    """AVX-512 lowering of :class:`TileBinop` (``dace/tile_ops/avx512.h``)."""

    environments = [TileOpsAVX512]

    @staticmethod
    def expansion(node: "TileBinop", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _isa_codegen.make_binop_tasklet(node, parent_state, parent_sdfg, "avx512")


@library.expansion
class ExpandTileBinopAVX2(ExpandTransformation):
    """AVX2 lowering of :class:`TileBinop` (``dace/tile_ops/avx2.h``)."""

    environments = [TileOpsAVX2]

    @staticmethod
    def expansion(node: "TileBinop", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _isa_codegen.make_binop_tasklet(node, parent_state, parent_sdfg, "avx2")


@library.expansion
class ExpandTileBinopNeon(ExpandTransformation):
    """ARM NEON lowering of :class:`TileBinop` (``dace/tile_ops/arm_neon.h``)."""

    environments = [TileOpsNeon]

    @staticmethod
    def expansion(node: "TileBinop", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _isa_codegen.make_binop_tasklet(node, parent_state, parent_sdfg, "neon")


@library.expansion
class ExpandTileBinopSVE(ExpandTransformation):
    """ARM SVE lowering of :class:`TileBinop` (``dace/tile_ops/arm_sve.h``)."""

    environments = [TileOpsSVE]

    @staticmethod
    def expansion(node: "TileBinop", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _isa_codegen.make_binop_tasklet(node, parent_state, parent_sdfg, "sve")


@library.node
class TileBinop(nodes.LibraryNode):
    """Element-wise binary op on K-dim register tiles.

    Each operand has a ``kind``: ``Tile`` (read via the ``_a`` / ``_b``
    connector from a tile-shape array) or ``Symbol`` (a free-symbol
    expression embedded inline in the tasklet body, evaluated in the
    surrounding scope). At least one operand must be ``Tile``. With
    ``has_mask=True``, an additional ``_mask`` input gates the write
    per lane.

    :cvar implementations: Per-target expansions; ``"pure"`` is the
        flattened CPP-loop correctness fallback. ``"cutile"`` emits the
        :mod:`cuda.tile`-Python equivalent (opt-in; the orchestrator
        stays on ``"pure"`` for CPU).
    :cvar default_implementation: ``"pure"``.
    """

    implementations = {
        "pure": ExpandTileBinopPure,
        "cutile": ExpandTileBinopCutile,
        # K=1 ISA backends: a call into dace/tile_ops/<backend>.h (same call;
        # the backend's env pulls in the matching header).
        "scalar": ExpandTileBinopScalar,
        "avx512": ExpandTileBinopAVX512,
        "avx2": ExpandTileBinopAVX2,
        "neon": ExpandTileBinopNeon,
        "sve": ExpandTileBinopSVE,
    }
    default_implementation = "pure"

    target_isa = properties.Property(
        dtype=str,
        allow_none=False,
        default="SCALAR",
        desc="CPU target ISA the Auto-dispatch lowers to for K==1 "
        "(SCALAR | AVX512 | AVX2 | ARM_SVE | ARM_NEON | CUTILE); K>=2 is pure. "
        "Stamped by the VectorizeCPUMultiDim orchestrator before expansion.",
    )
    op = properties.Property(
        dtype=str,
        allow_none=False,
        default="+",
        desc="Binary op (one of: + - * / min max < <= > >= == != && ||).",
    )
    widths = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )
    has_mask = properties.Property(
        dtype=bool,
        allow_none=False,
        default=False,
        desc="When True, the ``_mask`` input connector is required.",
    )
    kind_a = properties.Property(
        dtype=str,
        allow_none=False,
        default=_TILE,
        desc="Operand kind for the left-hand side: 'Tile' or 'Symbol'.",
    )
    kind_b = properties.Property(
        dtype=str,
        allow_none=False,
        default=_TILE,
        desc="Operand kind for the right-hand side: 'Tile' or 'Symbol'.",
    )
    expr_a = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Symbolic expression embedded inline when kind_a == 'Symbol'; ignored otherwise.",
    )
    expr_b = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Symbolic expression embedded inline when kind_b == 'Symbol'; ignored otherwise.",
    )

    def __init__(self,
                 name: str,
                 widths: Tuple[int, ...],
                 op: str = "+",
                 has_mask: bool = False,
                 kind_a: str = _TILE,
                 kind_b: str = _TILE,
                 expr_a: Optional[str] = None,
                 expr_b: Optional[str] = None,
                 location: Optional[str] = None):
        """Construct a ``TileBinop`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param op: One of the keys of :data:`_OP_CPP`.
        :param has_mask: When True, declare the ``_mask`` input
            connector.
        :param kind_a: ``"Tile"`` (default — read via ``_a`` connector),
            ``"Symbol"`` (embed ``expr_a`` inline), or ``"Scalar"`` (read
            a length-1 / ``dace.data.Scalar`` via ``_a``, broadcast to
            every lane).
        :param kind_b: ``"Tile"``, ``"Symbol"`` or ``"Scalar"``.
        :param expr_a: Required when ``kind_a == "Symbol"``.
        :param expr_b: Required when ``kind_b == "Symbol"``.
        :param location: Optional DaCe node location override.
        :raises ValueError: On invalid ``op``, ``widths`` length, kind,
            missing expression for symbol kinds, or a no-Tile-operand
            pair (at least one operand must be a tile).
        """
        if op not in _OP_CPP:
            raise ValueError(f"TileBinop: unknown op {op!r}; allowed: {sorted(_OP_CPP)}")
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileBinop: widths must have length in {{1, 2, 3}}, got {widths!r}")
        for label, kind in (("kind_a", kind_a), ("kind_b", kind_b)):
            if kind not in _VALID_KINDS:
                raise ValueError(f"TileBinop: {label} must be one of {_VALID_KINDS}, got {kind!r}")
        if kind_a == _SYMBOL and not expr_a:
            raise ValueError("TileBinop: kind_a='Symbol' requires expr_a")
        if kind_b == _SYMBOL and not expr_b:
            raise ValueError("TileBinop: kind_b='Symbol' requires expr_b")

        inputs = set()
        if kind_a in (_TILE, _SCALAR):
            inputs.add("_a")
        if kind_b in (_TILE, _SCALAR):
            inputs.add("_b")
        if has_mask:
            inputs.add("_mask")
        super().__init__(name, location=location, inputs=inputs, outputs={"_c"})
        self.widths = list(widths)
        self.op = op
        self.has_mask = has_mask
        self.kind_a = kind_a
        self.kind_b = kind_b
        self.expr_a = expr_a
        self.expr_b = expr_b

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Validate connector counts at expansion time.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If a required connector is unconnected.
        :raises NotImplementedError: If Tile operand dtypes disagree
            with the output dtype (E2 lock).
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_c" not in out_e:
            raise ValueError(f"{self.label}: required output '_c' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
        c_arr = sdfg.arrays[out_e["_c"].data.data]
        for label, kind in (("_a", self.kind_a), ("_b", self.kind_b)):
            if kind in (_TILE, _SCALAR):
                if label not in in_e:
                    raise ValueError(f"{self.label}: kind={kind!r} but {label!r} not connected")
                # Each Tile / Scalar operand is promoted to the output dtype
                # before the op (the expansion casts on lowering). Widening
                # (int -> float/double, int -> wider int, float -> double) is
                # allowed; a narrowing conversion (e.g. double -> int) raises.
                if kind == _TILE:
                    src = sdfg.arrays[in_e[label].data.data].dtype
                    if not _promotion_ok(src, c_arr.dtype):
                        raise NotImplementedError(
                            f"{self.label}: Tile operand {label!r} dtype {src} cannot be promoted to output "
                            f"dtype {c_arr.dtype} (narrowing conversion); cast explicitly via a separate tasklet.")
