# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileBinop``: a binary op on K-dim register tiles."""

import dace
from dace import library, properties
from dace.libraries.tileops.environments import (
    TileOpsAVX2,
    TileOpsAVX512,
    TileOpsCUDA,
    TileOpsNeon,
    TileOpsScalar,
    TileOpsSVE,
)
from dace.libraries.tileops.expansions import ExpandTileIsa, ExpandTilePure
from dace.libraries.tileops.isa import IsaCall, call_text, template_flag
from dace.libraries.tileops.kinds import TILE
from dace.libraries.tileops.lanes import SEQUENTIAL, TileLowering
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.libraries.tileops.operands import (
    LaneOperands,
    Operand,
    check_operands,
    edge_ctype,
    elementwise_tasklet,
    has_lane_invariant_output,
    input_connectors,
    operands_share_output_type,
    output_edge,
    validate_elementwise,
)
from dace.libraries.tileops.ops import BINARY_OPS
from dace.libraries.tileops.validation import promotion_ok
from dace.sdfg import nodes
from dace.tile_functions import TileFunction


@library.expansion
class ExpandTileBinopPure(ExpandTilePure):
    pass


@library.expansion
class ExpandTileBinopScalar(ExpandTileIsa):
    environments = [TileOpsScalar]
    backend = "scalar"


@library.expansion
class ExpandTileBinopAVX512(ExpandTileIsa):
    environments = [TileOpsAVX512]
    backend = "avx512"


@library.expansion
class ExpandTileBinopAVX2(ExpandTileIsa):
    environments = [TileOpsAVX2]
    backend = "avx2"


@library.expansion
class ExpandTileBinopNeon(ExpandTileIsa):
    environments = [TileOpsNeon]
    backend = "neon"


@library.expansion
class ExpandTileBinopSVE(ExpandTileIsa):
    environments = [TileOpsSVE]
    backend = "sve"


@library.expansion
class ExpandTileBinopCUDA(ExpandTileIsa):
    environments = [TileOpsCUDA]
    backend = "cuda"


@library.node
class TileBinop(TileOp):
    """``_c = _a <op> _b`` over a tile.

    Each operand has a kind (see :mod:`dace.libraries.tileops.kinds`): ``Tile`` and ``Scalar`` read the connector
    ``_a`` / ``_b``, ``Symbol`` embeds ``expr_a`` / ``expr_b`` inline. With ``has_mask`` the lanes where ``_mask`` is
    false are zero.
    """

    lanes_independent = True
    implementations = {
        "pure": ExpandTileBinopPure,
        "scalar": ExpandTileBinopScalar,
        "avx512": ExpandTileBinopAVX512,
        "avx2": ExpandTileBinopAVX2,
        "neon": ExpandTileBinopNeon,
        "sve": ExpandTileBinopSVE,
        "cuda": ExpandTileBinopCUDA,
    }
    default_implementation = "pure"

    op = properties.Property(
        dtype=str,
        allow_none=False,
        default="+",
        desc="The op, a key of ``dace.libraries.tileops.ops.BINARY_OPS``.",
    )
    has_mask = properties.Property(
        dtype=bool,
        allow_none=False,
        default=False,
        desc="Whether the ``_mask`` input connector gates the lanes.",
    )
    kind_a = properties.Property(
        dtype=str,
        allow_none=False,
        default=TILE,
        desc="How the left operand is read: 'Tile', 'Scalar' or 'Symbol'.",
    )
    kind_b = properties.Property(
        dtype=str,
        allow_none=False,
        default=TILE,
        desc="How the right operand is read: 'Tile', 'Scalar' or 'Symbol'.",
    )
    expr_a = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="The expression of the left operand when it is a 'Symbol'.",
    )
    expr_b = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="The expression of the right operand when it is a 'Symbol'.",
    )

    def __init__(
        self,
        name: str,
        widths: tuple[int, ...],
        op: str = "+",
        has_mask: bool = False,
        kind_a: str = TILE,
        kind_b: str = TILE,
        expr_a: str | None = None,
        expr_b: str | None = None,
        location: str | None = None,
    ):
        if op not in BINARY_OPS:
            raise ValueError(f"TileBinop: unknown op {op!r}; allowed: {sorted(BINARY_OPS)}")
        if not 1 <= len(widths) <= 3:
            raise ValueError(f"TileBinop: widths must have length in {{1, 2, 3}}, got {widths!r}")
        operands = [Operand("_a", kind_a, expr_a), Operand("_b", kind_b, expr_b)]
        check_operands("TileBinop", operands)
        super().__init__(
            name, location=location, inputs=dict.fromkeys(input_connectors(operands, has_mask)), outputs={"_c"}
        )
        self.widths = list(widths)
        self.op = op
        self.has_mask = has_mask
        self.kind_a = kind_a
        self.kind_b = kind_b
        self.expr_a = expr_a
        self.expr_b = expr_b

    def operands(self) -> list[Operand]:
        return [Operand("_a", self.kind_a, self.expr_a), Operand("_b", self.kind_b, self.expr_b)]

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        # A comparison compares a tile operand at its own dtype and stores the ``bool`` it answers, which every numeric
        # output holds: ``b_index > 0.0`` into an int8 mask narrows nothing, so its operands are not promoted.
        validate_elementwise(
            self,
            state,
            sdfg,
            self.operands(),
            "_c",
            self.has_mask,
            promotion=None if BINARY_OPS[self.op].comparison else promotion_ok,
        )

    def can_lower_to_isa(self, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        operands = self.operands()
        return (
            BINARY_OPS[self.op].isa_code is not None
            and not has_lane_invariant_output(operands, output_edge(state, self, "_c"))
            and operands_share_output_type(self, state, sdfg, operands, "_c")
        )

    def pure_tasklet(
        self, state: dace.SDFGState, sdfg: dace.SDFG, lowering: TileLowering = SEQUENTIAL
    ) -> nodes.Tasklet:
        self.validate(sdfg, state)
        operands = self.operands()
        out_ctype = edge_ctype(sdfg, output_edge(state, self, "_c"))
        lanes = LaneOperands.of(self, state, sdfg, operands, out_ctype)
        lhs, rhs = (
            lanes.reference(operand, [other for other in operands if other is not operand]) for operand in operands
        )
        return elementwise_tasklet(
            self, state, operands, "_c", BINARY_OPS[self.op].cpp(lhs, rhs), out_ctype, lanes.in_edges, lowering=lowering
        )

    def isa_tasklet(
        self, state: dace.SDFGState, sdfg: dace.SDFG, backend: str, lowering: TileLowering = SEQUENTIAL
    ) -> nodes.Tasklet:
        self.validate(sdfg, state)
        operands = self.operands()
        call = IsaCall.of(self, state, sdfg, "_c", backend, lowering)
        (a_broadcast, a_pointer), (b_broadcast, b_pointer) = (call.operand(operand) for operand in operands)
        op = BINARY_OPS[self.op].isa_code
        if call.functions:
            function = TileFunction(
                "binop", backend, call.element, call.vlen, op, a_broadcast + b_broadcast, self.has_mask
            )
            name = function.name
        else:
            masked = "true" if self.has_mask else "false"
            flags = f"{template_flag(a_broadcast)}, {template_flag(b_broadcast)}"
            name = f"dace::tileops::tile_binop<{call.ctype}, {call.vlen}, '{op}', {flags}, {masked}>"
        text = call_text(name, [call.output("_c"), a_pointer, b_pointer, *call.mask(self.has_mask)])
        return call.tasklet(self, backend, operands, "_c", text, self.has_mask)
