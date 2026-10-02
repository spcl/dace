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

import dace
from dace import library, properties
from dace.codegen.cppunparse import pyexpr2cpp
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from ..kinds import SCALAR, SYMBOL, TILE, VALID_KINDS
from ..ops import BINARY_OPS
from .. import _isa_codegen
from ..lanes import half_disambiguated, lane_invariant_assign, nested_loops, tile_offset
from ..operands import scalar_operand_ref
from ..validation import edge_moves_a_tile, edge_moves_one_element, is_tile_shape, promotion_ok


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

        # The dtype the VALUE operands share. A ``SYMBOL`` / ``SCALAR``
        # operand is cast to this so a type-strict binop (``std::min`` etc.)
        # resolves both operands at one type. This is the OPERAND dtype, NOT
        # ``out_dtype``: a comparison (``ZLI > RLMIN``) has a ``bool`` output
        # but ``double`` operands, so casting the symbol to ``out_dtype`` would
        # emit ``(bool)RLMIN`` — truncating ``RLMIN=1e-12`` to ``1`` and
        # corrupting the comparison. Prefer a data operand's descriptor dtype;
        # else the symbol's own declared dtype from ``sdfg.symbols``; else
        # fall back to ``out_dtype`` (all-Symbol case with no resolvable type).
        def _operand_dtype() -> str:
            for k, c in ((node.kind_a, "_a"), (node.kind_b, "_b")):
                if k in (TILE, SCALAR) and c in in_e:
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
        cast = "" if operand_dtype == "bool" else f"({operand_dtype})"

        def _effective_ctype(kind: str, conn: str) -> str:
            """The C++ type ``conn`` is actually emitted as (post any cast)."""
            if kind == SYMBOL:
                return operand_dtype
            if kind == TILE:
                return parent_sdfg.arrays[in_e[conn].data.data].dtype.ctype
            desc = parent_sdfg.arrays[in_e[conn].data.data]
            _, broadcast = scalar_operand_ref(desc, conn, widths, off)
            return operand_dtype if broadcast else desc.dtype.ctype

        ctype_a = _effective_ctype(node.kind_a, "_a")
        ctype_b = _effective_ctype(node.kind_b, "_b")

        def _operand_ref(kind: str, conn: str, expr: str | None, meets_ctype: str) -> str:
            """Return the per-lane C++ reference for one operand.

            A ``SYMBOL`` / ``SCALAR`` operand is cast to ``operand_dtype``
            (see above) so ``std::min`` / ``std::max`` (and any other
            type-strict overload) sees both operands at the same type — and a
            comparison's symbol operand keeps its numeric type rather than
            being truncated to the ``bool`` output. The cast is suppressed when
            the operand dtype is bool (logical ops; no ``(bool)X`` is emitted).
            A ``TILE`` / per-lane ``SCALAR`` operand keeps its own dtype
            uncast (like before) UNLESS it is ``dace::float16`` meeting a
            differently-typed sibling operand: ``__half`` (what
            ``dace::float16`` is on GPU) exposes several simultaneously
            implicit conversions, so handing it bare to a mixed-type infix
            operator is a compile-time ambiguity, not a truncation risk --
            ``half_disambiguated`` routes it through one explicit, lossless
            ``(float)`` hop first (see its docstring).
            """
            if kind == SYMBOL:
                return f"{cast}({pyexpr2cpp(expr)})"
            if kind == TILE:
                src = parent_sdfg.arrays[in_e[conn].data.data].dtype.ctype
                return half_disambiguated(f"{conn}[{off}]", src, meets_ctype)
            # Scalar operand. A tile-shape Array widened upstream is a pointer
            # read per lane (``conn[off]``); any volume-1 source (Scalar /
            # length-1 Array / single-element access) is passed by value and
            # read as the bare ``conn``. A per-lane tile read keeps the tile
            # dtype (no cast, like a Tile operand); a broadcast is cast to
            # ``operand_dtype`` so a typed binop (``std::min`` etc.) resolves.
            desc = parent_sdfg.arrays[in_e[conn].data.data]
            ref, broadcast = scalar_operand_ref(desc, conn, widths, off)
            if broadcast:
                return f"{cast}({ref})"
            return half_disambiguated(ref, desc.dtype.ctype, meets_ctype)

        lhs = _operand_ref(node.kind_a, "_a", node.expr_a, ctype_b)
        rhs = _operand_ref(node.kind_b, "_b", node.expr_b, ctype_a)
        rhs_expr = BINARY_OPS[node.op].cpp(lhs, rhs)
        # Output kind dispatch (design 6.2): when all inputs are non-Tile and the ``_c`` memlet moves
        # one element, emit a single assignment with no lane loop. Otherwise emit the K-fold loop
        # ``_c[off] = ...`` over the tile.
        out_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == "_c")
        out_is_scalar = (node.kind_a != TILE and node.kind_b != TILE and edge_moves_one_element(out_edge))
        if out_is_scalar:
            # No lane loop: a one-element output is a by-value local (``T _c;``) assigned once.
            mask_elements = in_e["_mask"].data.subset.num_elements() if node.has_mask else None
            code = lane_invariant_assign("_c", rhs_expr, out_dtype, widths, mask_elements)
        else:
            if node.has_mask:
                body = f"_c[{off}] = _mask[{off}] ? ({rhs_expr}) : {out_dtype}(0);"
            else:
                body = f"_c[{off}] = {rhs_expr};"
            code = nested_loops(widths, body)
        inputs = {"_a", "_b", "_mask"}
        if node.kind_a == SYMBOL:
            inputs.discard("_a")
        if node.kind_b == SYMBOL:
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
        flattened CPP-loop correctness fallback.
    :cvar default_implementation: ``"pure"``.
    """

    # The backend below is chosen from the vectorizer's ``target_isa``, not from the target
    # device, so device auto-selection must not overwrite it.
    auto_select_implementation = False
    implementations = {
        "pure": ExpandTileBinopPure,
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
        "(SCALAR | AVX512 | AVX2 | ARM_SVE | ARM_NEON | CUDA); K>=2 is pure. "
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
        default=TILE,
        desc="Operand kind for the left-hand side: 'Tile' or 'Symbol'.",
    )
    kind_b = properties.Property(
        dtype=str,
        allow_none=False,
        default=TILE,
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
                 kind_a: str = TILE,
                 kind_b: str = TILE,
                 expr_a: Optional[str] = None,
                 expr_b: Optional[str] = None,
                 location: Optional[str] = None):
        """Construct a ``TileBinop`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param op: One of the keys of :data:`~dace.libraries.tileops.ops.BINARY_OPS`.
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
        if op not in BINARY_OPS:
            raise ValueError(f"TileBinop: unknown op {op!r}; allowed: {sorted(BINARY_OPS)}")
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileBinop: widths must have length in {{1, 2, 3}}, got {widths!r}")
        for label, kind in (("kind_a", kind_a), ("kind_b", kind_b)):
            if kind not in VALID_KINDS:
                raise ValueError(f"TileBinop: {label} must be one of {VALID_KINDS}, got {kind!r}")
        if kind_a == SYMBOL and not expr_a:
            raise ValueError("TileBinop: kind_a='Symbol' requires expr_a")
        if kind_b == SYMBOL and not expr_b:
            raise ValueError("TileBinop: kind_b='Symbol' requires expr_b")

        inputs = set()
        if kind_a in (TILE, SCALAR):
            inputs.add("_a")
        if kind_b in (TILE, SCALAR):
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
        any_tile_input = (self.kind_a == TILE or self.kind_b == TILE)
        if any_tile_input and not (is_tile_shape(c_arr, tuple(self.widths))
                                   or edge_moves_a_tile(out_e["_c"], tuple(self.widths))):
            raise NotImplementedError(f"{self.label}: output-kind rule violated -- kind_a={self.kind_a!r}, "
                                      f"kind_b={self.kind_b!r} (has Tile input) but '_c' descriptor is not tile-shape "
                                      f"{tuple(self.widths)!r}. Per design section 6.2: any Tile input -> Tile output.")
        for label, kind in (("_a", self.kind_a), ("_b", self.kind_b)):
            if kind in (TILE, SCALAR):
                if label not in in_e:
                    raise ValueError(f"{self.label}: kind={kind!r} but {label!r} not connected")
                # Each Tile / Scalar operand is promoted to the output dtype
                # before the op (the expansion casts on lowering). Widening
                # (int -> float/double, int -> wider int, float -> double) is
                # allowed; a narrowing conversion (e.g. double -> int) raises.
                # A comparison is exempt: the expansion compares a Tile operand at its own dtype and
                # stores the ``bool`` it answers, which every numeric output holds exactly. The
                # operand never meets the output, so ``b_index > 0.0`` into an int8 mask narrows nothing.
                if kind == TILE and not BINARY_OPS[self.op].comparison:
                    src = sdfg.arrays[in_e[label].data.data].dtype
                    if not promotion_ok(src, c_arr.dtype):
                        raise NotImplementedError(
                            f"{self.label}: Tile operand {label!r} dtype {src} cannot be promoted to output "
                            f"dtype {c_arr.dtype} (narrowing conversion); cast explicitly via a separate tasklet.")
