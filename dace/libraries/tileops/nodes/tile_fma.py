# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileFMA`` fused multiply-add on K-dim register tiles (``_o = _a * _b + _c``).

The lib node consumes three operands ``_a`` / ``_b`` / ``_c`` and writes ``_o``
(mirroring :class:`TileITE`'s output connector). Each operand carries a ``kind``
flag -- ``Tile`` (a tile-shape array via a connector), ``Scalar`` (a length-1 /
:class:`dace.data.Scalar` broadcast via a connector) or ``Symbol`` (a free-symbol
expression embedded inline in the tasklet body). At least one operand must be
``Tile``; an all-Symbol / all-Scalar triple is loop-invariant and belongs outside
the tile path.

The op is the FUSED multiply-add with a SINGLE rounding: the pure expansion emits
``std::fma((a), (b), (c))`` (NOT ``a*b + c``) and every ISA backend lowers to the
native fused FMA (``_mm*_fmadd`` / ``vfmaq`` / ``svmla`` / ``__hfma2`` / the
scalar ``std::fma``), so the pure and ISA lowerings agree bit-for-bit. The caller
opts into FMA's single-rounded result over a separate ``*`` then ``+``.

The pure expansion returns a CPP tasklet whose body is a single ``for``-loop over
the flattened tile (correctness-only).
"""
from typing import Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .._pure_codegen import nested_loops, tile_offset
from .. import _isa_codegen
from .tile_binop import (_TILE, _SYMBOL, _SCALAR, _VALID_KINDS, _is_tile_shape, _is_scalar_shape, scalar_operand_ref,
                         _promotion_ok)


@library.expansion
class ExpandTileFMAPure(ExpandTransformation):
    """Correctness-only CPP tasklet lowering of ``TileFMA`` (fused ``a*b + c``)."""

    environments = []

    @staticmethod
    def expansion(node: "TileFMA", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a single CPP tasklet that walks the flattened tile.

        :param node: The ``TileFMA`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A CPP tasklet replacing the lib node in place.
        """
        node.validate(parent_sdfg, parent_state)
        widths = list(node.widths)
        off = tile_offset(widths)
        in_e = {e.dst_conn: e for e in parent_state.in_edges(node) if e.dst_conn is not None}

        out_dtype = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node)
                                            if e.src_conn == "_o").data.data].dtype.ctype

        # The dtype the VALUE operands share (a ``_SYMBOL`` / ``_SCALAR`` operand
        # is cast to this). Prefer a data operand's descriptor dtype; else a
        # symbol's own declared dtype; else fall back to ``out_dtype``. Mirrors
        # ``ExpandTileBinopPure._operand_dtype``.
        def _operand_dtype():
            for k, c in ((node.kind_a, "_a"), (node.kind_b, "_b"), (node.kind_c, "_c")):
                if k in (_TILE, _SCALAR) and c in in_e:
                    return parent_sdfg.arrays[in_e[c].data.data].dtype.ctype
            for expr in (node.expr_a, node.expr_b, node.expr_c):
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
        _cast = "" if operand_dtype == "bool" else f"({operand_dtype})"

        def _operand_ref(kind, conn, expr):
            """Return the per-lane C++ reference for one FMA operand.

            A ``_SYMBOL`` / broadcast ``_SCALAR`` operand is cast to
            ``operand_dtype`` so ``std::fma`` resolves all three operands at one
            type; a per-lane Tile read (or a tile-shape Scalar widened upstream)
            keeps the tile dtype uncast, exactly like a Tile operand.
            """
            if kind == _SYMBOL:
                return f"{_cast}({expr})"
            if kind == _TILE:
                return f"{conn}[{off}]"
            # Scalar operand: descriptor-aware (a tile-shape Array widened upstream
            # is read per lane ``conn[off]``; a by-value Scalar / length-1 Array is
            # broadcast and cast to ``operand_dtype``).
            desc = parent_sdfg.arrays[in_e[conn].data.data]
            ref, broadcast = scalar_operand_ref(desc, conn, widths, off)
            return f"{_cast}({ref})" if broadcast else ref

        a_ref = _operand_ref(node.kind_a, "_a", node.expr_a)
        b_ref = _operand_ref(node.kind_b, "_b", node.expr_b)
        c_ref = _operand_ref(node.kind_c, "_c", node.expr_c)
        # Fused single-rounding multiply-add (NOT ``a*b + c``): keeps the pure path
        # bit-for-bit with every ISA backend's native FMA.
        rhs_expr = f"std::fma({a_ref}, {b_ref}, {c_ref})"
        # Output-kind dispatch (design 6.2): all inputs non-Tile and ``_o`` Scalar /
        # length-1 -> a single assignment (no lane loop); otherwise the K-fold loop.
        out_desc = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node) if e.src_conn == "_o").data.data]
        out_is_scalar = (node.kind_a != _TILE and node.kind_b != _TILE and node.kind_c != _TILE
                         and _is_scalar_shape(out_desc))
        if out_is_scalar:
            # Scalar output: no lane loop; one assignment. A volume-1 output (Scalar
            # or length-1 Array) is a by-value local (``T _o;``), so it -- and the
            # volume-1 ``_mask`` gating it -- are referenced bare (``[0]`` is a
            # memlet concern, never a tasklet-body one).
            if node.has_mask:
                body = f"_o = _mask ? ({rhs_expr}) : {out_dtype}(0);"
            else:
                body = f"_o = {rhs_expr};"
            code = body
        else:
            if node.has_mask:
                body = f"_o[{off}] = _mask[{off}] ? ({rhs_expr}) : {out_dtype}(0);"
            else:
                body = f"_o[{off}] = {rhs_expr};"
            code = nested_loops(widths, body)
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
            label=f"{node.label}_pure",
            inputs={c: None
                    for c in inputs},
            outputs={"_o": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandTileFMACutile(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileFMA` (stubbed out)."""

    environments = []

    @staticmethod
    def expansion(node: "TileFMA", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        raise NotImplementedError(
            "ExpandTileFMACutile: cuTile expansion stubbed out during G3 step 3 migration; the unified `TileLoad` / `TileStore` (with `gather_dims`) cuTile path will be reinstated after the per-source-dim gather contract lands per design "
            "section 6.4. Pin a `pure` expansion via `sdfg.expand_library_nodes(implementation='pure')` to lower this node for now."
        )


@library.node
class TileFMA(nodes.LibraryNode):
    """Fused multiply-add ``_o = _a * _b + _c`` on K-dim register tiles.

    Each operand has a ``kind``: ``Tile`` (read via the ``_a`` / ``_b`` / ``_c``
    connector from a tile-shape array), ``Scalar`` (a length-1 /
    :class:`dace.data.Scalar` broadcast via its connector) or ``Symbol`` (a
    free-symbol expression embedded inline in the tasklet body). At least one
    operand must be ``Tile``. With ``has_mask=True``, an additional ``_mask``
    input gates the write per lane.

    The op is the FUSED multiply-add with a SINGLE rounding (``std::fma`` /
    native FMA), so the pure and every ISA lowering agree bit-for-bit.

    :cvar implementations: Per-target expansions; ``"pure"`` is the flattened
        CPP-loop correctness fallback. ``"cutile"`` is the (stubbed)
        :mod:`cuda.tile`-Python equivalent.
    :cvar default_implementation: ``"pure"``.
    """

    implementations = {
        "pure": ExpandTileFMAPure,
        "cutile": ExpandTileFMACutile,
        # K=1 ISA backends (scalar / avx512 / avx2 / neon / sve / cuda): a call
        # into dace/tile_ops/<backend>.h -- the backend's env pulls in the matching
        # header. Built by the shared factory (selector routes K>=2 to ``pure``).
        **_isa_codegen.make_isa_expansions("Fma", _isa_codegen.make_fma_tasklet, globals()),
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
        desc="Operand kind for the multiplicand ``a``: 'Tile', 'Scalar' or 'Symbol'.",
    )
    kind_b = properties.Property(
        dtype=str,
        allow_none=False,
        default=_TILE,
        desc="Operand kind for the multiplier ``b``: 'Tile', 'Scalar' or 'Symbol'.",
    )
    kind_c = properties.Property(
        dtype=str,
        allow_none=False,
        default=_TILE,
        desc="Operand kind for the addend ``c``: 'Tile', 'Scalar' or 'Symbol'.",
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
    expr_c = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Symbolic expression embedded inline when kind_c == 'Symbol'; ignored otherwise.",
    )

    def __init__(self,
                 name: str,
                 widths: Tuple[int, ...],
                 has_mask: bool = False,
                 kind_a: str = _TILE,
                 kind_b: str = _TILE,
                 kind_c: str = _TILE,
                 expr_a: Optional[str] = None,
                 expr_b: Optional[str] = None,
                 expr_c: Optional[str] = None,
                 location: Optional[str] = None):
        """Construct a ``TileFMA`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param has_mask: When True, declare the ``_mask`` input connector.
        :param kind_a: ``"Tile"`` (default -- read via ``_a`` connector),
            ``"Scalar"`` (length-1 / ``dace.data.Scalar`` via ``_a``, broadcast)
            or ``"Symbol"`` (embed ``expr_a`` inline).
        :param kind_b: ``"Tile"``, ``"Scalar"`` or ``"Symbol"``.
        :param kind_c: ``"Tile"``, ``"Scalar"`` or ``"Symbol"``.
        :param expr_a: Required when ``kind_a == "Symbol"``.
        :param expr_b: Required when ``kind_b == "Symbol"``.
        :param expr_c: Required when ``kind_c == "Symbol"``.
        :param location: Optional DaCe node location override.
        :raises ValueError: On invalid ``widths`` length, kind, missing
            expression for a symbol kind, or a no-Tile-operand triple (at least
            one operand must be a tile).
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileFMA: widths must have length in {{1, 2, 3}}, got {widths!r}")
        for label, kind in (("kind_a", kind_a), ("kind_b", kind_b), ("kind_c", kind_c)):
            if kind not in _VALID_KINDS:
                raise ValueError(f"TileFMA: {label} must be one of {_VALID_KINDS}, got {kind!r}")
        if kind_a == _SYMBOL and not expr_a:
            raise ValueError("TileFMA: kind_a='Symbol' requires expr_a")
        if kind_b == _SYMBOL and not expr_b:
            raise ValueError("TileFMA: kind_b='Symbol' requires expr_b")
        if kind_c == _SYMBOL and not expr_c:
            raise ValueError("TileFMA: kind_c='Symbol' requires expr_c")
        if _TILE not in (kind_a, kind_b, kind_c):
            raise ValueError("TileFMA: at least one operand must be a Tile "
                             f"(got kind_a={kind_a!r}, kind_b={kind_b!r}, kind_c={kind_c!r})")

        inputs = set()
        if kind_a in (_TILE, _SCALAR):
            inputs.add("_a")
        if kind_b in (_TILE, _SCALAR):
            inputs.add("_b")
        if kind_c in (_TILE, _SCALAR):
            inputs.add("_c")
        if has_mask:
            inputs.add("_mask")
        super().__init__(name, location=location, inputs=inputs, outputs={"_o"})
        self.widths = list(widths)
        self.has_mask = has_mask
        self.kind_a = kind_a
        self.kind_b = kind_b
        self.kind_c = kind_c
        self.expr_a = expr_a
        self.expr_b = expr_b
        self.expr_c = expr_c

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Validate connector counts + output-kind rule at expansion time.

        Output-kind rule (design section 6.2): any Tile input -> ``_o`` must be
        tile-shape (``Array(shape=widths)``).

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If a required connector is unconnected.
        :raises NotImplementedError: If a Tile operand dtype cannot be promoted to
            the output dtype (narrowing) or the output-kind rule is violated.
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_o" not in out_e:
            raise ValueError(f"{self.label}: required output '_o' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
        o_arr = sdfg.arrays[out_e["_o"].data.data]
        # Output-kind rule (design 6.2): when any input is Tile, ``_o`` must be tile-shape.
        any_tile_input = _TILE in (self.kind_a, self.kind_b, self.kind_c)
        if any_tile_input and not _is_tile_shape(o_arr, tuple(self.widths)):
            raise NotImplementedError(f"{self.label}: output-kind rule violated -- a Tile input is present but "
                                      f"'_o' descriptor is not tile-shape {tuple(self.widths)!r}. Per design "
                                      f"section 6.2: any Tile input -> Tile output.")
        for label, kind in (("_a", self.kind_a), ("_b", self.kind_b), ("_c", self.kind_c)):
            if kind in (_TILE, _SCALAR):
                if label not in in_e:
                    raise ValueError(f"{self.label}: kind={kind!r} but {label!r} not connected")
                # Each Tile operand is promoted to the output dtype before the op
                # (the expansion casts on lowering). Widening (int -> float/double,
                # int -> wider int, float -> double) is allowed; a narrowing
                # conversion (e.g. double -> int) raises.
                if kind == _TILE:
                    src = sdfg.arrays[in_e[label].data.data].dtype
                    if not _promotion_ok(src, o_arr.dtype):
                        raise NotImplementedError(
                            f"{self.label}: Tile operand {label!r} dtype {src} cannot be promoted to output "
                            f"dtype {o_arr.dtype} (narrowing conversion); cast explicitly via a separate tasklet.")
