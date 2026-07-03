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


def _is_tile_shape(desc, widths) -> bool:
    """True iff ``desc`` is an :class:`dace.data.Array` whose shape equals ``widths``."""
    if not isinstance(desc, dace.data.Array):
        return False
    shape = tuple(desc.shape)
    if len(shape) != len(widths):
        return False
    return all(bool(dace.symbolic.simplify(s - w) == 0) for s, w in zip(shape, widths))


def _is_scalar_shape(desc) -> bool:
    """True iff ``desc`` is a :class:`dace.data.Scalar` or a length-1 :class:`Array`."""
    if isinstance(desc, dace.data.Scalar):
        return True
    if isinstance(desc, dace.data.Array):
        return all(bool(dace.symbolic.simplify(s - 1) == 0) for s in desc.shape)
    return False


def scalar_operand_ref(desc, conn: str, widths, off: str) -> Tuple[str, bool]:
    """Per-lane C++ reference for a ``Scalar``-kind tile-op operand.

    A ``Scalar``-kind operand (one classified as a broadcast because its source
    is read through a single-element ``"0"`` memlet) may be bound to one of two
    connector ABIs:

    * a **tile-shape** :class:`dace.data.Array` (``shape == widths``) -> a
      transient an upstream tile op widened to a register tile, then read here
      through a single-element memlet. The connector is a pointer (``T* conn``)
      carrying PER-LANE data, so it must be read ``conn[off]`` exactly like a
      Tile operand. Reading it as a broadcast would emit ``(T)conn`` -- an
      invalid pointer-to-value cast.
    * anything else (a true :class:`dace.data.Scalar`, a length-1 Array, or any
      single-element access) -> DaCe passes a volume-1 connector by value
      (``T conn = ...``), so the tasklet references the bare ``conn`` and
      broadcasts it. ``[0]`` is a *memlet* concern, never a tasklet-body one --
      a by-value ``conn`` is not a pointer.

    :param desc: The data descriptor bound to ``conn``.
    :param conn: The tasklet input connector name (e.g. ``"_a"``).
    :param widths: Per-dim tile widths (innermost-last).
    :param off: The flattened per-lane offset expression (from ``tile_offset``).
    :returns: ``(ref, broadcast)`` -- the C++ reference and whether it is a
        loop-invariant broadcast. The caller casts a broadcast to the operand
        dtype; a per-lane tile read (``broadcast == False``) keeps the tile
        dtype uncast, exactly like a Tile operand.
    """
    if isinstance(desc, dace.data.Array) and _is_tile_shape(desc, tuple(widths)):
        return f"{conn}[{off}]", False
    return conn, True


def _promotion_ok(src: dace.dtypes.typeclass, dst: dace.dtypes.typeclass) -> bool:
    """Whether a Tile operand of dtype ``src`` may be promoted to the output
    dtype ``dst`` before the op (a widening conversion).

    Allowed (widening): same dtype; integer -> wider-or-equal integer; integer
    -> float / double; float -> double; integer -> bool (truthiness cast,
    ``int != 0`` — well-defined in C++ for any integer operand, used by the
    merge-cond compound combine where a comparison result stored as int64
    flows into a bool-output combine tasklet). Disallowed (narrowing -> the
    caller must crash): float / double -> integer; double -> float; integer
    narrowing (e.g. int64 -> int32).

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
    s_bool = (src.type is np.bool_)
    d_bool = (dst.type is np.bool_)
    if s_int and d_flt:  # int -> float / double
        return True
    if s_int and d_int and dst.bytes >= src.bytes:  # integer widening
        return True
    if s_flt and d_flt and dst.bytes >= src.bytes:  # float -> double
        return True
    if (s_int or s_bool or s_flt) and d_bool:  # numeric -> bool (truthiness)
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
    # Python ``%`` differs from C ``%`` on negative operands (Python follows the divisor's
    # sign; C follows the dividend's). DaCe runtime ships ``dace::math::py_mod`` which
    # matches Python semantics; use it so the vectorised body matches the unvectorised
    # reference bit-for-bit. ``py_mod`` is the function-call spelling of the same op
    # (the ``RewriteModuloToPyMod`` cleaning step rewrites ``%`` to it); both lower here.
    # ``py_mod`` is a GLOBAL runtime function (it lives outside ``dace::math`` -- see
    # math.h "must reside outside of the DaCe namespace"); call it unqualified.
    "%": ("py_mod(", ", ", ")"),
    "py_mod": ("py_mod(", ", ", ")"),
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
    # Use ``std::`` for elemental functions: matches the K=1 ISA backend's
    # tile_binop_apply (which calls ``std::min`` / ``std::max``).
    "min": ("std::min(", ", ", ")"),
    "max": ("std::max(", ", ", ")"),
    # Python ``**`` -> ``std::pow``. ``PowerOperatorExpansion`` runs upstream in the
    # multi-dim pipeline and rewrites integer-constant exponents (``x**2`` -> ``x*x``);
    # only true runtime / non-constant exponents reach this lowering.
    "**": ("std::pow(", ", ", ")"),
    # Binary elemental math functions (``np.arctan2`` -> the frontend's bare
    # ``atan2(a, b)``; likewise ``hypot`` / ``fmod``). Emitted as a per-lane
    # ``std::atan2(a[i], b[i])`` inside the tile for-loop; the ``_dace_tile_vectorize``
    # pragma lets the compiler's vector-math library (libmvec) capture the call.
    "atan2": ("std::atan2(", ", ", ")"),
    "hypot": ("std::hypot(", ", ", ")"),
    "fmod": ("std::fmod(", ", ", ")"),
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

        # The dtype the VALUE operands share. A ``_SYMBOL`` / ``_SCALAR``
        # operand is cast to this so a type-strict binop (``std::min`` etc.)
        # resolves both operands at one type. This is the OPERAND dtype, NOT
        # ``out_dtype``: a comparison (``ZLI > RLMIN``) has a ``bool`` output
        # but ``double`` operands, so casting the symbol to ``out_dtype`` would
        # emit ``(bool)RLMIN`` — truncating ``RLMIN=1e-12`` to ``1`` and
        # corrupting the comparison. Prefer a data operand's descriptor dtype;
        # else the symbol's own declared dtype from ``sdfg.symbols``; else
        # fall back to ``out_dtype`` (all-Symbol case with no resolvable type).
        def _operand_dtype():
            for k, c in ((node.kind_a, "_a"), (node.kind_b, "_b")):
                if k in (_TILE, _SCALAR) and c in in_e:
                    return parent_sdfg.arrays[in_e[c].data.data].dtype.ctype
            for expr in (node.expr_a, node.expr_b):
                if not expr:
                    continue
                try:
                    for s in dace.symbolic.symlist(dace.symbolic.pystr_to_symbolic(expr)):
                        if str(s) in parent_sdfg.symbols:
                            return parent_sdfg.symbols[str(s)].ctype
                except Exception:  # noqa: BLE001
                    pass
            return out_dtype

        operand_dtype = _operand_dtype()
        # Never emit a ``(bool)X`` cast: a logical op's operands are already
        # bool tiles, and casting a value to bool truncates it. The cast only
        # exists to resolve type-strict overloads (``std::min(int, double)``),
        # which are never bool; so suppress it when the operand dtype is bool.
        _cast = "" if operand_dtype == "bool" else f"({operand_dtype})"

        def _operand_ref(kind, conn, expr):
            """Return the per-lane C++ reference for one operand.

            A ``_SYMBOL`` / ``_SCALAR`` operand is cast to ``operand_dtype``
            (see above) so ``std::min`` / ``std::max`` (and any other
            type-strict overload) sees both operands at the same type — and a
            comparison's symbol operand keeps its numeric type rather than
            being truncated to the ``bool`` output. The cast is suppressed when
            the operand dtype is bool (logical ops; no ``(bool)X`` is emitted).
            """
            if kind == _SYMBOL:
                return f"{_cast}({expr})"
            if kind == _TILE:
                return f"{conn}[{off}]"
            # Scalar operand. A tile-shape Array widened upstream is a pointer
            # read per lane (``conn[off]``); any volume-1 source (Scalar /
            # length-1 Array / single-element access) is passed by value and
            # read as the bare ``conn``. A per-lane tile read keeps the tile
            # dtype (no cast, like a Tile operand); a broadcast is cast to
            # ``operand_dtype`` so a typed binop (``std::min`` etc.) resolves.
            desc = parent_sdfg.arrays[in_e[conn].data.data]
            ref, broadcast = scalar_operand_ref(desc, conn, widths, off)
            return f"{_cast}({ref})" if broadcast else ref

        lhs = _operand_ref(node.kind_a, "_a", node.expr_a)
        rhs = _operand_ref(node.kind_b, "_b", node.expr_b)
        rhs_expr = _binop_rhs(node.op, lhs, rhs)
        # Output kind dispatch (design 6.2): when all inputs are non-Tile and ``_c`` is Scalar /
        # length-1, emit a single assignment with no lane loop. Otherwise emit the K-fold loop
        # ``_c[off] = ...`` over the tile.
        out_desc = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node) if e.src_conn == "_c").data.data]
        out_is_scalar = (node.kind_a != _TILE and node.kind_b != _TILE and _is_scalar_shape(out_desc))
        if out_is_scalar:
            # The Scalar output path: no lane loop; one assignment. A volume-1
            # output (Scalar or length-1 Array) is a by-value local (``T _c;``),
            # so it -- and the volume-1 ``_mask`` gating it -- are referenced
            # bare. ``[0]`` is a memlet concern, never a tasklet-body one (a
            # by-value connector is not a pointer).
            if node.has_mask:
                body = f"_c = _mask ? ({rhs_expr}) : {out_dtype}(0);"
            else:
                body = f"_c = {rhs_expr};"
            code = body
        else:
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
    # cuTile is Python-semantics, so a bare ``%`` already matches ``py_mod``.
    "%": "{lhs} % {rhs}",
    "py_mod": "{lhs} % {rhs}",
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
        raise NotImplementedError(
            "ExpandTileBinopCutile: cuTile expansion stubbed out during G3 step 3 migration; the unified `TileLoad` / `TileStore` (with `gather_dims`) cuTile path will be reinstated after the per-source-dim gather contract lands per design "
            "section 6.4. Pin a `pure` expansion via `sdfg.expand_library_nodes(implementation='pure')` to lower this node for now."
        )


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
        # K=1 ISA backends (scalar / avx512 / avx2 / neon / sve): a call into
        # dace/tile_ops/<backend>.h -- same call, the backend's env pulls in the
        # matching header. Built by the shared factory (selector routes K>=2 to
        # ``pure``).
        **_isa_codegen.make_isa_expansions("Binop", _isa_codegen.make_binop_tasklet, globals()),
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
        """Validate connector counts + output-kind rule at expansion time.

        Output-kind rule (design section 6.2, locked 2026-06-09):
        any Tile input -> ``_c`` must be tile-shape (``Array(shape=widths)``).
        All inputs Scalar / Symbol -> ``_c`` may be Scalar / length-1 Array
        (preferred) OR tile-shape (allowed for compositional flexibility).

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If a required connector is unconnected.
        :raises NotImplementedError: If Tile operand dtypes disagree with
            the output dtype (E2 lock) or the output-kind rule is violated.
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_c" not in out_e:
            raise ValueError(f"{self.label}: required output '_c' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
        c_arr = sdfg.arrays[out_e["_c"].data.data]
        # Output-kind rule (design 6.2): when any input is Tile, the output must be tile-shape.
        any_tile_input = (self.kind_a == _TILE or self.kind_b == _TILE)
        if any_tile_input and not _is_tile_shape(c_arr, tuple(self.widths)):
            raise NotImplementedError(f"{self.label}: output-kind rule violated -- kind_a={self.kind_a!r}, "
                                      f"kind_b={self.kind_b!r} (has Tile input) but '_c' descriptor is not tile-shape "
                                      f"{tuple(self.widths)!r}. Per design section 6.2: any Tile input -> Tile output.")
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
