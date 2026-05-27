# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileUnop`` element-wise unary op on K-dim register tiles.

The lib node consumes one operand ``_a`` and writes ``_c`` (``_mask``
optional). The operand carries a ``kind`` flag — ``Tile`` (per-lane via a
connector), ``Scalar`` (length-1 / ``dace.data.Scalar`` broadcast), or
``Symbol`` (a free-symbol expression embedded inline). Covers the unary ops
the legacy 1D path emitted: ``neg`` (``-a``), ``abs``, ``exp``, ``log``,
``sqrt``, ``sin``, ``cos``, ``floor``, ``ceil``, ``tanh``.

The pure expansion returns a CPP tasklet whose body is a single ``for``-loop
over the flattened tile (correctness-only); the K=1 ISA backends call
``dace::tileops::tile_unop`` in ``dace/tile_ops/<backend>.h``.
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

_TILE = "Tile"
_SYMBOL = "Symbol"
_SCALAR = "Scalar"
_VALID_KINDS = (_TILE, _SYMBOL, _SCALAR)

# op -> (prefix, suffix) for the pure (K>=2) inline C++ form ``<pre>operand<suf>``.
_UNOP_CPP = {
    "neg": ("(-", ")"),
    "abs": ("std::abs(", ")"),
    "exp": ("std::exp(", ")"),
    "log": ("std::log(", ")"),
    "sqrt": ("std::sqrt(", ")"),
    "sin": ("std::sin(", ")"),
    "cos": ("std::cos(", ")"),
    "floor": ("std::floor(", ")"),
    "ceil": ("std::ceil(", ")"),
    "tanh": ("std::tanh(", ")"),
}

# op -> the cuTile-Python expression (operand placeholder ``{a}``).
_CUTE_UNOP_EXPR = {
    "neg": "-{a}",
    "abs": "ct.abs({a})",
    "exp": "ct.exp({a})",
    "log": "ct.log({a})",
    "sqrt": "ct.sqrt({a})",
    "sin": "ct.sin({a})",
    "cos": "ct.cos({a})",
    "floor": "ct.floor({a})",
    "ceil": "ct.ceil({a})",
    "tanh": "ct.tanh({a})",
}


def _promotion_ok(src: dace.dtypes.typeclass, dst: dace.dtypes.typeclass) -> bool:
    """Whether a Tile operand of dtype ``src`` may be promoted to the output
    dtype ``dst`` (a widening conversion) before the op.

    Same rule as :func:`dace.libraries.tileops.nodes.tile_binop._promotion_ok`:
    int -> float/double, int -> wider int, float -> double, or equal; a
    narrowing conversion raises.

    :param src: The operand's element dtype.
    :param dst: The output element dtype.
    :returns: ``True`` iff promoting ``src`` to ``dst`` is non-narrowing.
    """
    if src == dst:
        return True
    s_int = np.issubdtype(src.type, np.integer)
    d_int = np.issubdtype(dst.type, np.integer)
    s_flt = np.issubdtype(src.type, np.floating)
    d_flt = np.issubdtype(dst.type, np.floating)
    if s_int and d_flt:
        return True
    if s_int and d_int and dst.bytes >= src.bytes:
        return True
    if s_flt and d_flt and dst.bytes >= src.bytes:
        return True
    return False


@library.expansion
class ExpandTileUnopPure(ExpandTransformation):
    """Correctness-only CPP tasklet lowering of ``TileUnop``."""

    environments = []

    @staticmethod
    def expansion(node: "TileUnop", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a single CPP tasklet that walks the flattened tile.

        :param node: The ``TileUnop`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A CPP tasklet replacing the lib node in place.
        """
        node.validate(parent_sdfg, parent_state)
        widths = list(node.widths)
        off = tile_offset(widths)
        in_e = {e.dst_conn: e for e in parent_state.in_edges(node) if e.dst_conn is not None}

        if node.kind_a == _SYMBOL:
            operand = f"({node.expr_a})"
        elif node.kind_a == _TILE:
            operand = f"_a[{off}]"
        else:  # Scalar: length-1 Array reads ``_a[0]``, a dace.data.Scalar is ``_a``.
            desc = parent_sdfg.arrays[in_e["_a"].data.data]
            operand = "_a" if isinstance(desc, dace.data.Scalar) else "_a[0]"

        pre, post = _UNOP_CPP[node.op]
        rhs_expr = f"{pre}{operand}{post}"
        out_dtype = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node)
                                            if e.src_conn == "_c").data.data].dtype.ctype
        if node.has_mask:
            body = f"_c[{off}] = _mask[{off}] ? ({rhs_expr}) : {out_dtype}(0);"
        else:
            body = f"_c[{off}] = {rhs_expr};"
        code = nested_loops(widths, body)
        inputs = set()
        if node.kind_a in (_TILE, _SCALAR):
            inputs.add("_a")
        if node.has_mask:
            inputs.add("_mask")
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={c: None
                    for c in inputs},
            outputs={"_c": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandTileUnopCute(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileUnop`."""

    environments = []

    @staticmethod
    def expansion(node: "TileUnop", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet emitting the cuTile unary op.

        :param node: The lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet with the element-wise body.
        """
        if node.kind_a == _SYMBOL:
            operand = node.expr_a
        elif node.kind_a == _TILE:
            operand = "__rhs1"
        else:
            operand = "__const1"
        body = f"__output = {_CUTE_UNOP_EXPR[node.op].format(a=operand)}"
        inputs = set()
        if node.kind_a == _TILE:
            inputs.add("__rhs1")
        elif node.kind_a == _SCALAR:
            inputs.add("__const1")
        return nodes.Tasklet(
            label=f"{node.label}_cute",
            inputs={c: None
                    for c in inputs},
            outputs={"__output": None},
            code=body,
            language=dace.dtypes.Language.Python,
        )


@library.expansion
class ExpandTileUnopScalar(ExpandTransformation):
    """K=1 scalar backend lowering (``dace/tile_ops/scalar.h``)."""

    environments = [TileOpsScalar]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_unop_tasklet(node, parent_state, parent_sdfg, "scalar")


@library.expansion
class ExpandTileUnopAVX512(ExpandTransformation):
    """K=1 avx512 backend lowering (``dace/tile_ops/avx512.h``)."""

    environments = [TileOpsAVX512]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_unop_tasklet(node, parent_state, parent_sdfg, "avx512")


@library.expansion
class ExpandTileUnopAVX2(ExpandTransformation):
    """K=1 avx2 backend lowering (``dace/tile_ops/avx2.h``)."""

    environments = [TileOpsAVX2]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_unop_tasklet(node, parent_state, parent_sdfg, "avx2")


@library.expansion
class ExpandTileUnopNeon(ExpandTransformation):
    """K=1 neon backend lowering (``dace/tile_ops/arm_neon.h``)."""

    environments = [TileOpsNeon]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_unop_tasklet(node, parent_state, parent_sdfg, "neon")


@library.expansion
class ExpandTileUnopSVE(ExpandTransformation):
    """K=1 sve backend lowering (``dace/tile_ops/arm_sve.h``)."""

    environments = [TileOpsSVE]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_unop_tasklet(node, parent_state, parent_sdfg, "sve")


@library.node
class TileUnop(nodes.LibraryNode):
    """Element-wise unary op on a K-dim register tile.

    Connectors: ``_a`` (operand, omitted when ``kind_a == 'Symbol'``),
    ``_mask`` (optional), ``_c`` (output tile).

    :cvar implementations: ``pure`` (portable lane loop), ``cute`` (cuTile),
        and the four K=1 ISA backends.
    :cvar default_implementation: ``"pure"``.
    """

    implementations = {
        "pure": ExpandTileUnopPure,
        "cute": ExpandTileUnopCute,
        "scalar": ExpandTileUnopScalar,
        "avx512": ExpandTileUnopAVX512,
        "avx2": ExpandTileUnopAVX2,
        "neon": ExpandTileUnopNeon,
        "sve": ExpandTileUnopSVE,
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
        default="abs",
        desc="Unary op (one of: neg abs exp log sqrt sin cos floor ceil tanh).",
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
        desc="Operand kind: 'Tile', 'Scalar' or 'Symbol'.",
    )
    expr_a = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Symbolic expression embedded inline when kind_a == 'Symbol'; ignored otherwise.",
    )

    def __init__(self,
                 name: str,
                 widths: Tuple[int, ...],
                 op: str = "abs",
                 has_mask: bool = False,
                 kind_a: str = _TILE,
                 expr_a: Optional[str] = None,
                 location: Optional[str] = None):
        """Construct a ``TileUnop`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param op: One of the keys of :data:`_UNOP_CPP`.
        :param has_mask: When True, declare the ``_mask`` input connector.
        :param kind_a: ``"Tile"`` (default), ``"Scalar"`` or ``"Symbol"``.
        :param expr_a: Required when ``kind_a == "Symbol"``.
        :param location: Optional DaCe node location override.
        :raises ValueError: On invalid ``op``, ``widths`` length, kind, or a
            missing expression for the symbol kind.
        """
        if op not in _UNOP_CPP:
            raise ValueError(f"TileUnop: unknown op {op!r}; allowed: {sorted(_UNOP_CPP)}")
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileUnop: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if kind_a not in _VALID_KINDS:
            raise ValueError(f"TileUnop: kind_a must be one of {_VALID_KINDS}, got {kind_a!r}")
        if kind_a == _SYMBOL and not expr_a:
            raise ValueError("TileUnop: kind_a='Symbol' requires expr_a")

        inputs = set()
        if kind_a in (_TILE, _SCALAR):
            inputs.add("_a")
        if has_mask:
            inputs.add("_mask")
        super().__init__(name, location=location, inputs=inputs, outputs={"_c"})
        self.widths = list(widths)
        self.op = op
        self.has_mask = has_mask
        self.kind_a = kind_a
        self.expr_a = expr_a

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Validate connectors + the operand promotion at expansion time.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If a required connector is unconnected.
        :raises NotImplementedError: If a Tile operand dtype would narrow to
            the output dtype.
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_c" not in out_e:
            raise ValueError(f"{self.label}: required output '_c' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
        if self.kind_a in (_TILE, _SCALAR) and "_a" not in in_e:
            raise ValueError(f"{self.label}: kind_a={self.kind_a!r} but '_a' not connected")
        if self.kind_a == _TILE:
            c_arr = sdfg.arrays[out_e["_c"].data.data]
            src = sdfg.arrays[in_e["_a"].data.data].dtype
            if not _promotion_ok(src, c_arr.dtype):
                raise NotImplementedError(
                    f"{self.label}: Tile operand '_a' dtype {src} cannot be promoted to output dtype "
                    f"{c_arr.dtype} (narrowing conversion); cast explicitly via a separate tasklet.")
