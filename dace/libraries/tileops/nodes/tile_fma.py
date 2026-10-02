# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileFMA``: a fused multiply-add on K-dim register tiles."""
from typing import Final

import numpy as np

import dace
from dace import library, properties
from dace.sdfg import nodes

from ..environments import TileOpsAVX2, TileOpsAVX512, TileOpsCUDA, TileOpsNeon, TileOpsScalar, TileOpsSVE
from ..expansions import ExpandTileIsa, ExpandTilePure
from ..isa import IsaCall
from ..kinds import TILE
from ..operands import (LaneOperands, Operand, check_operands, edge_ctype, elementwise_tasklet,
                        has_lane_invariant_output, input_connectors, operands_share_output_type, output_edge,
                        validate_elementwise)
from .tile_op import TileOp

#: The registered dtypes narrower than ``float``, which take the ``double``-widened spelling of the fma; every other
#: operand type calls ``std::fma`` as it is. Read off the dtype registry, which includes the fp8 types, whose CUDA
#: classes convert implicitly to several types like ``__half``. ``bool`` is excluded though it is one byte: a widened
#: bool would read ``bool(std::fma(...))``, a truncating conversion to bool.
NARROW_OPERAND_CTYPES: Final[frozenset[str]] = frozenset(
    dtype.ctype for dtype in dace.dtypes.TYPECLASS_TO_STRING
    if dtype.bytes < dace.float32.bytes and dtype.type is not np.bool_)


@library.expansion
class ExpandTileFMAPure(ExpandTilePure):
    pass


@library.expansion
class ExpandTileFMAScalar(ExpandTileIsa):
    environments = [TileOpsScalar]
    backend = "scalar"


@library.expansion
class ExpandTileFMAAVX512(ExpandTileIsa):
    environments = [TileOpsAVX512]
    backend = "avx512"


@library.expansion
class ExpandTileFMAAVX2(ExpandTileIsa):
    environments = [TileOpsAVX2]
    backend = "avx2"


@library.expansion
class ExpandTileFMANeon(ExpandTileIsa):
    environments = [TileOpsNeon]
    backend = "neon"


@library.expansion
class ExpandTileFMASVE(ExpandTileIsa):
    environments = [TileOpsSVE]
    backend = "sve"


@library.expansion
class ExpandTileFMACUDA(ExpandTileIsa):
    environments = [TileOpsCUDA]
    backend = "cuda"


@library.node
class TileFMA(TileOp):
    """``_o = _a * _b + _c`` over a tile, rounded once.

    Every lowering is the fused multiply-add (``std::fma``, ``_mm*_fmadd``, ``vfmaq``, ``svmla``, ``__hfma2``), not a
    multiply followed by an add, so they agree bit for bit. Each operand has a kind (see
    :mod:`dace.libraries.tileops.kinds`), and at least one is a ``Tile``. With ``has_mask`` the lanes where ``_mask``
    is false are zero.
    """

    implementations = {
        "pure": ExpandTileFMAPure,
        "scalar": ExpandTileFMAScalar,
        "avx512": ExpandTileFMAAVX512,
        "avx2": ExpandTileFMAAVX2,
        "neon": ExpandTileFMANeon,
        "sve": ExpandTileFMASVE,
        "cuda": ExpandTileFMACUDA,
    }
    default_implementation = "pure"

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
        desc="How the multiplicand is read: 'Tile', 'Scalar' or 'Symbol'.",
    )
    kind_b = properties.Property(
        dtype=str,
        allow_none=False,
        default=TILE,
        desc="How the multiplier is read: 'Tile', 'Scalar' or 'Symbol'.",
    )
    kind_c = properties.Property(
        dtype=str,
        allow_none=False,
        default=TILE,
        desc="How the addend is read: 'Tile', 'Scalar' or 'Symbol'.",
    )
    expr_a = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="The expression of the multiplicand when it is a 'Symbol'.",
    )
    expr_b = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="The expression of the multiplier when it is a 'Symbol'.",
    )
    expr_c = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="The expression of the addend when it is a 'Symbol'.",
    )

    def __init__(self,
                 name: str,
                 widths: tuple[int, ...],
                 has_mask: bool = False,
                 kind_a: str = TILE,
                 kind_b: str = TILE,
                 kind_c: str = TILE,
                 expr_a: str | None = None,
                 expr_b: str | None = None,
                 expr_c: str | None = None,
                 location: str | None = None):
        if not 1 <= len(widths) <= 3:
            raise ValueError(f"TileFMA: widths must have length in {{1, 2, 3}}, got {widths!r}")
        operands = [Operand("_a", kind_a, expr_a), Operand("_b", kind_b, expr_b), Operand("_c", kind_c, expr_c)]
        check_operands("TileFMA", operands)
        if TILE not in (kind_a, kind_b, kind_c):
            raise ValueError("TileFMA: at least one operand must be a Tile "
                             f"(got kind_a={kind_a!r}, kind_b={kind_b!r}, kind_c={kind_c!r})")
        super().__init__(name,
                         location=location,
                         inputs=dict.fromkeys(input_connectors(operands, has_mask)),
                         outputs={"_o"})
        self.widths = list(widths)
        self.has_mask = has_mask
        self.kind_a = kind_a
        self.kind_b = kind_b
        self.kind_c = kind_c
        self.expr_a = expr_a
        self.expr_b = expr_b
        self.expr_c = expr_c

    def operands(self) -> list[Operand]:
        return [
            Operand("_a", self.kind_a, self.expr_a),
            Operand("_b", self.kind_b, self.expr_b),
            Operand("_c", self.kind_c, self.expr_c)
        ]

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        validate_elementwise(self, state, sdfg, self.operands(), "_o", self.has_mask)

    def can_lower_to_isa(self, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        operands = self.operands()
        return (not has_lane_invariant_output(operands, output_edge(state, self, "_o"))
                and operands_share_output_type(self, state, sdfg, operands, "_o"))

    def pure_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        self.validate(sdfg, state)
        operands = self.operands()
        out_ctype = edge_ctype(sdfg, output_edge(state, self, "_o"))
        lanes = LaneOperands.of(self, state, sdfg, operands, out_ctype)
        a, b, c = (lanes.reference(operand, [other for other in operands if other is not operand])
                   for operand in operands)
        if lanes.shared in NARROW_OPERAND_CTYPES:
            # ``std::fma`` has no half overload: on the device the overload is ambiguous and on the host it is an
            # out-of-line ``fmaf``. A half product is exact in double, so widening keeps the single rounding that two
            # chained half ops would lose, and matches a native ``__hfma``.
            rhs = f"{lanes.shared}(std::fma(double({a}), double({b}), double({c})))"
        else:
            rhs = f"std::fma({a}, {b}, {c})"
        return elementwise_tasklet(self, state, operands, "_o", rhs, out_ctype, lanes.in_edges)

    def isa_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG, backend: str) -> nodes.Tasklet:
        self.validate(sdfg, state)
        operands = self.operands()
        call = IsaCall.of(self, state, sdfg, "_o")
        (a_broadcast, a_pointer), (b_broadcast, b_pointer), (c_broadcast, c_pointer) = (call.operand(operand)
                                                                                        for operand in operands)
        masked = "true" if self.has_mask else "false"
        mask_argument = "_mask" if self.has_mask else "nullptr"
        text = (f"dace::tileops::tile_fma<{call.ctype}, {call.vlen}, {a_broadcast}, {b_broadcast}, {c_broadcast}, "
                f"{masked}>(_o, {a_pointer}, {b_pointer}, {c_pointer}, {mask_argument});")
        return call.tasklet(self, backend, operands, "_o", text, self.has_mask)
