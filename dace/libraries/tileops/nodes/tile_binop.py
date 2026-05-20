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
from functools import reduce
from operator import mul

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

_TILE = "Tile"
_SYMBOL = "Symbol"
_VALID_KINDS = (_TILE, _SYMBOL)

_OP_CPP = {
    "+": ("(", " + ", ")"),
    "-": ("(", " - ", ")"),
    "*": ("(", " * ", ")"),
    "/": ("(", " / ", ")"),
    "<": ("(", " < ", ")"),
    "<=": ("(", " <= ", ")"),
    ">": ("(", " > ", ")"),
    ">=": ("(", " >= ", ")"),
    "==": ("(", " == ", ")"),
    "!=": ("(", " != ", ")"),
    "&&": ("(", " && ", ")"),
    "||": ("(", " || ", ")"),
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
        n = reduce(mul, widths, 1)

        lhs = node.expr_a if node.kind_a == _SYMBOL else "_a[__k]"
        rhs = node.expr_b if node.kind_b == _SYMBOL else "_b[__k]"
        rhs_expr = _binop_rhs(node.op, lhs, rhs)

        if node.has_mask:
            body_inner = (
                f"_c[__k] = _mask[__k] ? ({rhs_expr}) : "
                f"static_cast<std::remove_reference_t<decltype(_c[__k])>>(0);"
            )
        else:
            body_inner = f"_c[__k] = {rhs_expr};"

        code = (
            f"for (std::size_t __k = 0; __k < {n}; ++__k) {{\n"
            f"    {body_inner}\n"
            f"}}"
        )
        inputs = {"_a", "_b", "_mask"}
        if node.kind_a == _SYMBOL:
            inputs.discard("_a")
        if node.kind_b == _SYMBOL:
            inputs.discard("_b")
        if not node.has_mask:
            inputs.discard("_mask")
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={c: None for c in inputs},
            outputs={"_c": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


_CUTE_OP_EXPR = {
    "+": "{lhs} + {rhs}",
    "-": "{lhs} - {rhs}",
    "*": "{lhs} * {rhs}",
    "/": "{lhs} / {rhs}",
    "<": "{lhs} < {rhs}",
    "<=": "{lhs} <= {rhs}",
    ">": "{lhs} > {rhs}",
    ">=": "{lhs} >= {rhs}",
    "==": "{lhs} == {rhs}",
    "!=": "{lhs} != {rhs}",
    "&&": "{lhs} & {rhs}",
    "||": "{lhs} | {rhs}",
    "min": "ct.minimum({lhs}, {rhs})",
    "max": "ct.maximum({lhs}, {rhs})",
}


@library.expansion
class ExpandTileBinopCute(ExpandTransformation):
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
        lhs = node.expr_a if node.kind_a == _SYMBOL else "__rhs1"
        rhs = node.expr_b if node.kind_b == _SYMBOL else "__rhs2"
        rhs_expr = _CUTE_OP_EXPR[node.op].format(lhs=lhs, rhs=rhs)
        body = f"__output = {rhs_expr}"
        inputs = set()
        if node.kind_a == _TILE:
            inputs.add("__rhs1")
        if node.kind_b == _TILE:
            inputs.add("__rhs2")
        return nodes.Tasklet(
            label=f"{node.label}_cute",
            inputs={c: None for c in inputs},
            outputs={"__output": None},
            code=body,
            language=dace.dtypes.Language.Python,
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
        flattened CPP-loop correctness fallback. ``"cute"`` emits the
        :mod:`cuda.tile`-Python equivalent (opt-in; the orchestrator
        stays on ``"pure"`` for CPU).
    :cvar default_implementation: ``"pure"``.
    """

    implementations = {"pure": ExpandTileBinopPure, "cute": ExpandTileBinopCute}
    default_implementation = "pure"

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
        :param kind_a: ``"Tile"`` (default — read via ``_a`` connector)
            or ``"Symbol"`` (embed ``expr_a`` inline).
        :param kind_b: ``"Tile"`` or ``"Symbol"``.
        :param expr_a: Required when ``kind_a == "Symbol"``.
        :param expr_b: Required when ``kind_b == "Symbol"``.
        :param location: Optional DaCe node location override.
        :raises ValueError: On invalid ``op``, ``widths`` length, kind,
            missing expression for symbol kinds, or symbol/symbol
            pairs.
        """
        if op not in _OP_CPP:
            raise ValueError(f"TileBinop: unknown op {op!r}; allowed: {sorted(_OP_CPP)}")
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileBinop: widths must have length in {{1, 2, 3}}, got {widths!r}")
        for label, kind in (("kind_a", kind_a), ("kind_b", kind_b)):
            if kind not in _VALID_KINDS:
                raise ValueError(f"TileBinop: {label} must be one of {_VALID_KINDS}, got {kind!r}")
        if kind_a == _SYMBOL and kind_b == _SYMBOL:
            raise ValueError(
                "TileBinop: at least one operand must be 'Tile'; a Symbol/Symbol pair "
                "belongs outside the tile path."
            )
        if kind_a == _SYMBOL and not expr_a:
            raise ValueError("TileBinop: kind_a='Symbol' requires expr_a")
        if kind_b == _SYMBOL and not expr_b:
            raise ValueError("TileBinop: kind_b='Symbol' requires expr_b")

        inputs = set()
        if kind_a == _TILE:
            inputs.add("_a")
        if kind_b == _TILE:
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

    def validate(self,
                 sdfg: dace.SDFG,
                 state: dace.SDFGState) -> None:
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
        dtypes_ = {c_arr.dtype}
        if self.kind_a == _TILE:
            if "_a" not in in_e:
                raise ValueError(f"{self.label}: kind_a='Tile' but '_a' not connected")
            dtypes_.add(sdfg.arrays[in_e["_a"].data.data].dtype)
        if self.kind_b == _TILE:
            if "_b" not in in_e:
                raise ValueError(f"{self.label}: kind_b='Tile' but '_b' not connected")
            dtypes_.add(sdfg.arrays[in_e["_b"].data.data].dtype)
        if len(dtypes_) > 1:
            raise NotImplementedError(
                f"{self.label}: TileBinop requires uniform dtype across Tile operands and _c "
                f"(got {dtypes_}); cast via separate tasklet first."
            )
