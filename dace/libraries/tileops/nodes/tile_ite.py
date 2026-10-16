# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileITE``: a per-lane select on K-dim register tiles."""

import dace
from dace import library, properties
from dace.codegen.cppunparse import pyexpr2cpp
from dace.libraries.tileops.environments import (
    TileOpsAVX2,
    TileOpsAVX512,
    TileOpsCUDA,
    TileOpsNeon,
    TileOpsScalar,
    TileOpsSVE,
)
from dace.libraries.tileops.expansions import ExpandTileIsa, ExpandTilePure
from dace.libraries.tileops.isa import IsaCall, broadcast_ref, call_text, element_code, template_flag
from dace.libraries.tileops.kinds import SYMBOL, TILE
from dace.libraries.tileops.lanes import nested_loops, tile_offset
from dace.libraries.tileops.nodes.tile_binop import _SCALAR, _SYMBOL, _TILE
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.libraries.tileops.operands import (
    Operand,
    check_operands,
    connected_edges,
    edge_ctype,
    input_connectors,
    output_edge,
    scalar_operand_ref,
    validate_elementwise,
)
from dace.optionals import required
from dace.sdfg import nodes
from dace.tile_functions import TileFunction
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandTileITEPure(ExpandTilePure):
    pass


@library.expansion
class ExpandTileITEScalar(ExpandTileIsa):
    environments = [TileOpsScalar]
    backend = "scalar"


@library.expansion
class ExpandTileITEAVX512(ExpandTileIsa):
    environments = [TileOpsAVX512]
    backend = "avx512"


@library.expansion
class ExpandTileITEAVX2(ExpandTileIsa):
    environments = [TileOpsAVX2]
    backend = "avx2"


@library.expansion
class ExpandTileITENeon(ExpandTileIsa):
    environments = [TileOpsNeon]
    backend = "neon"


@library.expansion
class ExpandTileITESVE(ExpandTileIsa):
    environments = [TileOpsSVE]
    backend = "sve"


@library.expansion
class ExpandTileITECUDA(ExpandTileIsa):
    environments = [TileOpsCUDA]
    backend = "cuda"


@library.expansion
class ExpandTileITECutile(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileITE`.

    Emits ``_o = ct.where(<mask>, <then>, <else>)`` — the cuTile
    per-lane select primitive.  Each operand (mask, then-arm, else-arm)
    can be a Tile (per-lane connector), a Scalar (length-1 connector;
    cuTile broadcasts automatically), or a Symbol (loop-invariant
    expression embedded inline).  The surrounding iteration mask is
    applied at the downstream ``ct.scatter`` store, not at the select
    (matching the reference cuTile kernels).
    """

    environments = []

    @staticmethod
    def expansion(node: "TileITE", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet emitting the cuTile ``ct.where`` select.

        :param node: The ``TileITE`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet replacing the lib node.
        """
        node.validate(parent_sdfg, parent_state)

        def _cutile_operand(kind: str, conn: str, expr: str) -> str:
            """cuTile operand reference: inline expr for Symbol, the
            connector for Tile or Scalar (cuTile broadcasts automatically)."""
            if kind == _SYMBOL:
                from dace.symbolic import symstr

                return symstr(expr)
            return conn  # Tile or Scalar — cuTile handles broadcasting

        mask_ref = _cutile_operand(node.kind_mask, "_mask", node.expr_mask)
        t_ref = _cutile_operand(node.kind_t, "_t", node.expr_t)
        e_ref = _cutile_operand(node.kind_e, "_e", node.expr_e)

        body = f"_o = ct.where({mask_ref}, {t_ref}, {e_ref})"

        inputs = set()
        if node.kind_mask in (_TILE, _SCALAR):
            inputs.add("_mask")
        if node.kind_t in (_TILE, _SCALAR):
            inputs.add("_t")
        if node.kind_e in (_TILE, _SCALAR):
            inputs.add("_e")

        return nodes.Tasklet(
            label=f"{node.label}_cutile",
            inputs={c: None for c in inputs},
            outputs={"_o": None},
            code=body,
            language=dace.dtypes.Language.Python,
        )


@library.node
class TileITE(TileOp):
    """``_o = _mask ? _t : _e`` per lane: the tile form of a branch-normalized ``merge(cond, then, else)``.

    The condition and both arms have a kind (see :mod:`dace.libraries.tileops.kinds`), so a loop-invariant condition
    is a ``Symbol`` or a ``Scalar`` and the node has no ``_mask`` connector then. The condition may have any dtype; the
    arms and the output share one. Masking the write is the job of the :class:`TileScatter` that consumes ``_o``.
    """

    lanes_independent = True
    implementations = {
        "cutile": ExpandTileITECutile,
        "pure": ExpandTileITEPure,
        "scalar": ExpandTileITEScalar,
        "avx512": ExpandTileITEAVX512,
        "avx2": ExpandTileITEAVX2,
        "neon": ExpandTileITENeon,
        "sve": ExpandTileITESVE,
        "cuda": ExpandTileITECUDA,
    }
    default_implementation = "pure"

    kind_t = properties.Property(
        dtype=str,
        allow_none=False,
        default=TILE,
        desc="How the then arm is read: 'Tile', 'Scalar' or 'Symbol'.",
    )
    kind_e = properties.Property(
        dtype=str,
        allow_none=False,
        default=TILE,
        desc="How the else arm is read: 'Tile', 'Scalar' or 'Symbol'.",
    )
    expr_t = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="The expression of the then arm when it is a 'Symbol'.",
    )
    expr_e = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="The expression of the else arm when it is a 'Symbol'.",
    )
    kind_mask = properties.Property(
        dtype=str,
        allow_none=False,
        default=TILE,
        desc="How the condition is read: 'Tile', 'Scalar' or 'Symbol'.",
    )
    expr_mask = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="The expression of the condition when it is a 'Symbol'.",
    )

    def __init__(
        self,
        name: str,
        widths: tuple[int, ...],
        kind_t: str = TILE,
        kind_e: str = TILE,
        expr_t: str | None = None,
        expr_e: str | None = None,
        kind_mask: str = TILE,
        expr_mask: str | None = None,
        location: str | None = None,
    ):
        if not 1 <= len(widths) <= 3:
            raise ValueError(f"TileITE: widths must have length in {{1, 2, 3}}, got {widths!r}")
        operands = [
            Operand("_mask", kind_mask, expr_mask),
            Operand("_t", kind_t, expr_t),
            Operand("_e", kind_e, expr_e),
        ]
        check_operands("TileITE", operands)
        super().__init__(
            name, location=location, inputs=dict.fromkeys(input_connectors(operands, False)), outputs={"_o"}
        )
        self.widths = list(widths)
        self.kind_t = kind_t
        self.kind_e = kind_e
        self.expr_t = expr_t
        self.expr_e = expr_e
        self.kind_mask = kind_mask
        self.expr_mask = expr_mask

    def operands(self) -> list[Operand]:
        return [
            Operand("_mask", self.kind_mask, self.expr_mask),
            Operand("_t", self.kind_t, self.expr_t),
            Operand("_e", self.kind_e, self.expr_e),
        ]

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        # The condition is not promoted to the output dtype; the arms are.
        validate_elementwise(self, state, sdfg, self.operands(), "_o", False, unpromoted=("_mask",))

    def pure_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        self.validate(sdfg, state)
        operands = self.operands()
        widths = list(self.widths)
        offset = tile_offset(widths)
        in_edges = connected_edges(state, self)
        out_ctype = edge_ctype(sdfg, output_edge(state, self, "_o"))

        def lane_reference(operand: Operand, cast: str | None) -> str:
            """A symbol and a broadcast scalar are cast to ``cast`` when it is given; a tile read keeps its dtype."""
            if operand.kind == SYMBOL:
                cpp = pyexpr2cpp(operand.expr)
                return f"({cast})({cpp})" if cast else f"({cpp})"
            if operand.kind == TILE:
                return f"{operand.conn}[{offset}]"
            reference, broadcast = scalar_operand_ref(
                sdfg.arrays[required(in_edges[operand.conn].data.data)], operand.conn, widths, offset
            )
            if not broadcast:
                return reference
            return f"({cast})({reference})" if cast else f"({reference})"

        condition, then, otherwise = operands
        body = (
            f"_o[{offset}] = ({lane_reference(condition, None)} ? {lane_reference(then, out_ctype)} : "
            f"{lane_reference(otherwise, out_ctype)});"
        )
        return nodes.Tasklet(
            label=f"{self.label}_pure",
            inputs=dict.fromkeys(input_connectors(operands, False)),
            outputs={"_o": None},
            code=nested_loops(widths, body),
            language=dace.dtypes.Language.CPP,
        )

    def isa_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG, backend: str) -> nodes.Tasklet:
        self.validate(sdfg, state)
        operands = self.operands()
        condition, then, otherwise = operands
        call = IsaCall.of(self, state, sdfg, "_o", backend)
        # The function reads the condition per lane, so a loop-invariant one is splatted across the tile.
        if condition.kind == TILE:
            condition_ctype = edge_ctype(sdfg, call.in_edges["_mask"])
            condition_pointer = call.input("_mask")
        else:
            condition_ctype = dace.bool_.ctype
            if condition.kind == SYMBOL:
                value = pyexpr2cpp(condition.expr)
            else:
                value = broadcast_ref("_mask", call.in_edges["_mask"].data.subset)
            condition_pointer = "_bcmask"
            call.pre.append(f"{condition_ctype} {condition_pointer}[{call.vlen}];")
            call.pre.append(
                f"for (int _mi = 0; _mi < {call.vlen}; ++_mi) {condition_pointer}[_mi] = ({condition_ctype})({value});"
            )
        (then_broadcast, then_pointer), (else_broadcast, else_pointer) = (
            call.operand(arm, promote_tile=False) for arm in (then, otherwise)
        )
        if call.functions:
            flags = then_broadcast + else_broadcast
            name = TileFunction(
                "ite", backend, call.element, call.vlen, "", flags, False, element_code(condition_ctype)
            ).name
        else:
            flags = f"{template_flag(then_broadcast)}, {template_flag(else_broadcast)}"
            name = f"dace::tileops::tile_ite<{call.ctype}, {condition_ctype}, {call.vlen}, {flags}, false>"
        arguments = [call.output("_o"), condition_pointer, then_pointer, else_pointer, *call.mask(False)]
        return call.tasklet(self, backend, operands, "_o", call_text(name, arguments), False)
