# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileIota``: a per-lane affine or indirect fill of an integer tile."""
from collections.abc import Sequence

import dace
from dace import library, properties
from dace.sdfg import nodes

from ..expansions import ExpandTilePure
from ..lanes import nested_loops, tile_offset
from .tile_op import TileOp


@library.expansion
class ExpandTileIotaPure(ExpandTilePure):
    pass


@library.node
class TileIota(TileOp):
    """``_dst[l_0, ..., l_{K-1}] = expr`` over a tile, the lane indices being spelled ``__l0, ..., __l{K-1}``.

    * An affine index tile: ``i + __l0`` is contiguous along dim 0, ``i + 2 * __l0`` strided.
    * An indirect index tile: ``extra_inputs = ("_idx", )`` and ``expr = "_idx[__l0]"``.
    * A multi-dim indirect one: ``extra_inputs = ("_src", )`` and ``expr = "_src[<flat offset of the lanes>]"``.
    """

    implementations = {"pure": ExpandTileIotaPure}
    default_implementation = "pure"

    expr = properties.Property(
        dtype=str,
        default="",
        desc="The per-lane value assigned to ``_dst``, over the lane indices ``__l0, ...`` and the extra inputs.",
    )
    extra_inputs = properties.ListProperty(
        element_type=str,
        default=[],
        desc="Input connectors the expression reads, such as ``_idx`` for a tile lookup or ``_src`` for a view of an "
        "outer array.",
    )

    def __init__(self,
                 name: str,
                 widths: tuple[int, ...],
                 expr: str,
                 extra_inputs: Sequence[str] = (),
                 location: str | None = None):
        if not 1 <= len(widths) <= 3:
            raise ValueError(f"TileIota: widths length {len(widths)} not in {{1, 2, 3}}")
        if not expr:
            raise ValueError("TileIota: expr is required (non-empty per-lane body)")
        super().__init__(name, location=location, inputs=dict.fromkeys(extra_inputs), outputs={"_dst"})
        self.widths = list(widths)
        self.expr = expr
        self.extra_inputs = list(extra_inputs)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        in_conns = {edge.dst_conn for edge in state.in_edges(self)}
        if "_dst" not in {edge.src_conn for edge in state.out_edges(self)}:
            raise ValueError(f"{self.label}: required output '_dst' not connected")
        for conn in self.extra_inputs:
            if conn not in in_conns:
                raise ValueError(f"{self.label}: declared extra input {conn!r} not connected")

    def pure_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        widths = list(self.widths)
        if all(width == 1 for width in widths):
            # DaCe collapses a ``Register`` ``Array(shape=(1,))`` to a plain scalar, which ``_dst[__l0]`` would index
            # into: the single lane has index 0 and the extra inputs are bare scalars as well.
            expr = self.expr
            for dim in range(len(widths)):
                expr = expr.replace(f"__l{dim}", "0")
            for conn in self.extra_inputs:
                expr = expr.replace(f"{conn}[0]", conn)
            code = f"_dst = {expr};"
        else:
            code = nested_loops(widths, f"_dst[{tile_offset(widths)}] = {self.expr};")
        return nodes.Tasklet(
            label=f"{self.label}_pure",
            inputs=dict.fromkeys(self.extra_inputs),
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
