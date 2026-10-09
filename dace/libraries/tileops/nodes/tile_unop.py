# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileUnop``: a unary op on K-dim register tiles."""

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
from dace.libraries.tileops.isa import IsaCall
from dace.libraries.tileops.kinds import SYMBOL, TILE
from dace.libraries.tileops.lanes import half_disambiguated, tile_offset
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.libraries.tileops.operands import (
    Operand,
    check_operands,
    connected_edges,
    edge_ctype,
    elementwise_tasklet,
    has_lane_invariant_output,
    input_connectors,
    operands_share_output_type,
    output_edge,
    scalar_operand_ref,
    validate_elementwise,
)
from dace.libraries.tileops.ops import CAST_OPS, UNARY_OPS
from dace.libraries.tileops.validation import promotion_ok
from dace.optionals import required
from dace.sdfg import nodes


@library.expansion
class ExpandTileUnopPure(ExpandTilePure):
    pass


@library.expansion
class ExpandTileUnopScalar(ExpandTileIsa):
    environments = [TileOpsScalar]
    backend = "scalar"


@library.expansion
class ExpandTileUnopAVX512(ExpandTileIsa):
    environments = [TileOpsAVX512]
    backend = "avx512"


@library.expansion
class ExpandTileUnopAVX2(ExpandTileIsa):
    environments = [TileOpsAVX2]
    backend = "avx2"


@library.expansion
class ExpandTileUnopNeon(ExpandTileIsa):
    environments = [TileOpsNeon]
    backend = "neon"


@library.expansion
class ExpandTileUnopSVE(ExpandTileIsa):
    environments = [TileOpsSVE]
    backend = "sve"


@library.expansion
class ExpandTileUnopCUDA(ExpandTileIsa):
    environments = [TileOpsCUDA]
    backend = "cuda"


@library.node
class TileUnop(TileOp):
    """``_c = <op> _a`` over a tile.

    The operand has a kind (see :mod:`dace.libraries.tileops.kinds`). With ``has_mask`` the lanes where ``_mask`` is
    false are zero. An op named after a dtype is the explicit conversion to it.
    """

    implementations = {
        "pure": ExpandTileUnopPure,
        "scalar": ExpandTileUnopScalar,
        "avx512": ExpandTileUnopAVX512,
        "avx2": ExpandTileUnopAVX2,
        "neon": ExpandTileUnopNeon,
        "sve": ExpandTileUnopSVE,
        "cuda": ExpandTileUnopCUDA,
    }
    default_implementation = "pure"

    op = properties.Property(
        dtype=str,
        allow_none=False,
        default="abs",
        desc="The op, a key of ``dace.libraries.tileops.ops.UNARY_OPS`` or a dtype name for a conversion.",
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
        desc="How the operand is read: 'Tile', 'Scalar' or 'Symbol'.",
    )
    expr_a = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="The expression of the operand when it is a 'Symbol'.",
    )

    def __init__(
        self,
        name: str,
        widths: tuple[int, ...],
        op: str = "abs",
        has_mask: bool = False,
        kind_a: str = TILE,
        expr_a: str | None = None,
        location: str | None = None,
    ):
        if op not in UNARY_OPS and op not in CAST_OPS:
            raise ValueError(f"TileUnop: unknown op {op!r}; allowed: {sorted(UNARY_OPS) + sorted(CAST_OPS)}")
        if not 1 <= len(widths) <= 3:
            raise ValueError(f"TileUnop: widths must have length in {{1, 2, 3}}, got {widths!r}")
        operands = [Operand("_a", kind_a, expr_a)]
        check_operands("TileUnop", operands)
        super().__init__(
            name, location=location, inputs=dict.fromkeys(input_connectors(operands, has_mask)), outputs={"_c"}
        )
        self.widths = list(widths)
        self.op = op
        self.has_mask = has_mask
        self.kind_a = kind_a
        self.expr_a = expr_a

    def operands(self) -> list[Operand]:
        return [Operand("_a", self.kind_a, self.expr_a)]

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:

        def promotes(src: dace.typeclass, dst: dace.typeclass) -> bool:
            # ``abs`` of a complex operand is its real magnitude, so a complex to real result is no narrowing.
            return (self.op == "abs" and src in (dace.dtypes.complex64, dace.dtypes.complex128)) or promotion_ok(
                src, dst
            )

        # A conversion is the one op that may narrow; every other op keeps the operand dtype, so a narrowing is a bug.
        validate_elementwise(
            self, state, sdfg, self.operands(), "_c", self.has_mask, promotion=None if self.op in CAST_OPS else promotes
        )

    def can_lower_to_isa(self, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        operands = self.operands()
        return (
            self.op in UNARY_OPS
            and UNARY_OPS[self.op].isa_code is not None
            and not has_lane_invariant_output(operands, output_edge(state, self, "_c"))
            and operands_share_output_type(self, state, sdfg, operands, "_c")
        )

    def pure_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        self.validate(sdfg, state)
        operand = self.operands()[0]
        in_edges = connected_edges(state, self)
        out_ctype = edge_ctype(sdfg, output_edge(state, self, "_c"))
        offset = tile_offset(self.widths)
        # A value cast to bool truncates it, and ``not`` has a bool operand and output.
        cast = "" if out_ctype == dace.bool_.ctype else f"({out_ctype})"
        if operand.kind == SYMBOL:
            source = f"({pyexpr2cpp(operand.expr)})"
            operand_ctype, reference = out_ctype, f"{cast}{source}"
        elif operand.kind == TILE:
            source = reference = f"_a[{offset}]"
            operand_ctype = edge_ctype(sdfg, in_edges["_a"])
        else:
            source, broadcast = scalar_operand_ref(
                sdfg.arrays[required(in_edges["_a"].data.data)], "_a", self.widths, offset
            )
            operand_ctype = out_ctype if broadcast else edge_ctype(sdfg, in_edges["_a"])
            reference = f"{cast}({source})" if broadcast else source
        if self.op in CAST_OPS:
            rhs = f"{CAST_OPS[self.op]}({source})"
        else:
            if self.op not in ("neg", "not") and operand_ctype == dace.float16.ctype:
                # Every other op is an overloaded ``std::`` function, none of which has a ``__half`` overload, so
                # ``dace::float16`` takes the explicit hop through ``float``.
                reference = half_disambiguated(reference, operand_ctype, "float")
            rhs = UNARY_OPS[self.op].cpp(reference)
        return elementwise_tasklet(self, state, [operand], "_c", rhs, out_ctype, in_edges)

    def isa_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG, backend: str) -> nodes.Tasklet:
        self.validate(sdfg, state)
        operands = self.operands()
        call = IsaCall.of(self, state, sdfg, "_c")
        broadcast, pointer = call.operand(operands[0])
        masked = "true" if self.has_mask else "false"
        mask_argument = "_mask" if self.has_mask else "nullptr"
        text = (
            f"dace::tileops::tile_unop<{call.ctype}, {call.vlen}, '{UNARY_OPS[self.op].isa_code}', "
            f"{broadcast}, {masked}>(_c, {pointer}, {mask_argument});"
        )
        return call.tasklet(self, backend, operands, "_c", text, self.has_mask)
