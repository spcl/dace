# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileBinop`` element-wise binary op on K-dim register tiles.

The lib node consumes two operands ``_a`` and ``_b`` and writes ``_c``.
Each operand carries a ``kind`` flag — ``Tile`` (a tile-shape array via
a connector) or ``Symbol`` (a free-symbol expression embedded inline in
the tasklet body). At least one operand must be ``Tile``; a Symbol /
Symbol pair belongs outside the tile path. The ``Symbol`` kind replaces
a standalone broadcast lib node: outer-scope symbols flow through
``TileBinop`` directly with no intermediate tile transient.

The pure expansion lowers to a K-fold nested Map with one scalar
tasklet per lane — correctness-only; cuTile lowering follows in T9.
"""
from typing import Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

_TILE = "Tile"
_SYMBOL = "Symbol"
_VALID_KINDS = (_TILE, _SYMBOL)

_PY_OP_TEMPLATES = {
    "+": "({lhs}) + ({rhs})",
    "-": "({lhs}) - ({rhs})",
    "*": "({lhs}) * ({rhs})",
    "/": "({lhs}) / ({rhs})",
    "min": "min(({lhs}), ({rhs}))",
    "max": "max(({lhs}), ({rhs}))",
    "<": "({lhs}) < ({rhs})",
    "<=": "({lhs}) <= ({rhs})",
    ">": "({lhs}) > ({rhs})",
    ">=": "({lhs}) >= ({rhs})",
    "==": "({lhs}) == ({rhs})",
    "!=": "({lhs}) != ({rhs})",
    "&&": "({lhs}) and ({rhs})",
    "||": "({lhs}) or ({rhs})",
}


@library.expansion
class ExpandTileBinopPure(ExpandTransformation):
    """Correctness-only K-fold nested-Map lowering of ``TileBinop``."""

    environments = []

    @staticmethod
    def expansion(node: "TileBinop", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        """Build a nested SDFG that loops over every lane and applies
        ``op`` to the corresponding lanes (per operand kind), gated by
        ``_mask`` when present.

        :param node: The ``TileBinop`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: Nested SDFG that replaces the lib node in place.
        """
        tile_dtype, mask_arr, a_arr, b_arr = node.validate(parent_sdfg, parent_state)
        widths = list(node.widths)

        sdfg = dace.SDFG(f"{node.label}_pure")
        sdfg.add_array("_c", widths, tile_dtype, transient=False)
        if node.kind_a == _TILE:
            sdfg.add_array("_a", widths, a_arr.dtype, transient=False)
        if node.kind_b == _TILE:
            sdfg.add_array("_b", widths, b_arr.dtype, transient=False)
        if node.has_mask:
            sdfg.add_array("_mask", widths, dace.bool_, transient=False)

        state = sdfg.add_state(f"{node.label}_state")
        K = len(widths)
        map_params = [f"__l{k}" for k in range(K)]
        map_rng = {p: f"0:{w}" for p, w in zip(map_params, widths)}
        access = ", ".join(map_params)

        inputs = {}
        lhs = node.expr_a if node.kind_a == _SYMBOL else "__rhs1"
        rhs = node.expr_b if node.kind_b == _SYMBOL else "__rhs2"
        if node.kind_a == _TILE:
            inputs["__rhs1"] = dace.Memlet(f"_a[{access}]")
        if node.kind_b == _TILE:
            inputs["__rhs2"] = dace.Memlet(f"_b[{access}]")
        if node.has_mask:
            inputs["__mask"] = dace.Memlet(f"_mask[{access}]")

        rhs_expr = _PY_OP_TEMPLATES[node.op].format(lhs=lhs, rhs=rhs)
        if node.has_mask:
            body = f"__output = ({rhs_expr}) if __mask else 0"
        else:
            body = f"__output = {rhs_expr}"
        outputs = {"__output": dace.Memlet(f"_c[{access}]")}

        state.add_mapped_tasklet(
            f"{node.label}_tasklet",
            map_rng,
            inputs,
            body,
            outputs,
            external_edges=True,
        )
        return sdfg


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
        K-fold nested-Map correctness fallback. cuTile expansion lands
        in T9.
    :cvar default_implementation: ``"pure"``.
    """

    implementations = {"pure": ExpandTileBinopPure}
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
        :param op: One of the keys of ``_PY_OP_TEMPLATES``.
        :param has_mask: When True, declare the ``_mask`` input
            connector.
        :param kind_a: ``"Tile"`` (default — read via ``_a`` connector)
            or ``"Symbol"`` (embed ``expr_a`` inline).
        :param kind_b: ``"Tile"`` or ``"Symbol"`` (same semantics as
            ``kind_a``).
        :param expr_a: Required when ``kind_a == "Symbol"``; the
            in-scope expression to evaluate per lane.
        :param expr_b: Required when ``kind_b == "Symbol"``.
        :param location: Optional DaCe node location override.
        :raises ValueError: On invalid ``op``, ``widths`` length, kind,
            missing ``expr_a`` / ``expr_b`` for symbol kinds, or
            symbol/symbol pairs (use a scalar op outside the tile path).
        """
        if op not in _PY_OP_TEMPLATES:
            raise ValueError(f"TileBinop: unknown op {op!r}; allowed: {sorted(_PY_OP_TEMPLATES)}")
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

    def validate(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
    ) -> Tuple[dace.dtypes.typeclass, Optional[dace.data.Data], Optional[dace.data.Data], Optional[dace.data.Data]]:
        """Look up data descriptors, infer the output dtype from the
        connected Tile operand, and check the homogeneous-dtype
        invariant across every Tile operand.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :returns: ``(tile_dtype, mask_arr_or_None, a_arr_or_None,
            b_arr_or_None)``. The two operand entries are ``None`` for
            ``Symbol``-kind sides (no connector → no edge).
        :raises ValueError: If a required connector is unconnected.
        :raises NotImplementedError: If Tile operand dtypes disagree
            (E2 lock).
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_c" not in out_e:
            raise ValueError(f"{self.label}: required output '_c' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
        a_arr = b_arr = None
        if self.kind_a == _TILE:
            if "_a" not in in_e:
                raise ValueError(f"{self.label}: kind_a='Tile' but '_a' not connected")
            a_arr = sdfg.arrays[in_e["_a"].data.data]
        if self.kind_b == _TILE:
            if "_b" not in in_e:
                raise ValueError(f"{self.label}: kind_b='Tile' but '_b' not connected")
            b_arr = sdfg.arrays[in_e["_b"].data.data]
        c_arr = sdfg.arrays[out_e["_c"].data.data]
        tile_descs = [d for d in (a_arr, b_arr, c_arr) if d is not None]
        dtypes = {d.dtype for d in tile_descs}
        if len(dtypes) != 1:
            raise NotImplementedError(
                f"{self.label}: TileBinop requires uniform dtype across Tile operands and _c "
                f"(got {dtypes}); cast via separate tasklet first."
            )
        mask_arr = sdfg.arrays[in_e["_mask"].data.data] if self.has_mask else None
        return c_arr.dtype, mask_arr, a_arr, b_arr
