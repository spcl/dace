# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileScan``: the inclusive scan of a one-dim tile."""

import math

import dace
from dace import library, properties
from dace.libraries.tileops.environments import TileOpsCUDABlock
from dace.libraries.tileops.expansions import THREAD_INDEX, ExpandTilePure
from dace.libraries.tileops.lanes import SEQUENTIAL, TileLowering
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.libraries.tileops.nodes.tile_reduce import collective_group, combine_expr, identity_literal
from dace.libraries.tileops.operands import connected_edges, edge_ctype
from dace.libraries.tileops.ops import BINARY_OPS, REDUCE_OPS
from dace.sdfg import nodes


@library.expansion
class ExpandTileScanPure(ExpandTilePure):
    pass


@library.node
class TileScan(TileOp):
    """Lane ``l`` of the tile ``_dst`` combines lanes ``0..l`` of the one-dim tile ``_src`` with ``op``.

    One thread runs the lanes in order. A ``BLOCK`` node in a GPU kernel scans with ``cub::BlockScan``, each thread
    holding consecutive lanes; on a CPU it runs as one thread does.
    """

    implementations = {"pure": ExpandTileScanPure}
    default_implementation = "pure"

    op = properties.Property(
        dtype=str,
        allow_none=False,
        default="+",
        desc="The combine: '+', '*', 'min' or 'max'.",
    )

    def __init__(self, name: str, widths: tuple[int, ...], op: str = "+", location: str | None = None):
        if op not in REDUCE_OPS:
            raise ValueError(f"TileScan: unknown op {op!r}; allowed: {REDUCE_OPS}")
        if len(widths) != 1:
            raise ValueError(f"TileScan: scans a one-dim tile, got the widths {widths!r}")
        super().__init__(name, location=location, inputs={"_src"}, outputs={"_dst"})
        self.widths = list(widths)
        self.op = op

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        if "_src" not in connected_edges(state, self):
            raise ValueError(f"{self.label}: required input '_src' not connected")
        if "_dst" not in {edge.src_conn for edge in state.out_edges(self)}:
            raise ValueError(f"{self.label}: required output '_dst' not connected")

    def can_lower_to_isa(self, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        return False

    def pure_tasklet(
        self, state: dace.SDFGState, sdfg: dace.SDFG, lowering: TileLowering = SEQUENTIAL
    ) -> nodes.Tasklet:
        ctype = edge_ctype(sdfg, connected_edges(state, self)["_src"])
        width = self.widths[0]
        code = (
            f"{ctype} __acc = {identity_literal(self.op, ctype)};\n"
            f"for (std::size_t __l0 = 0; __l0 < {width}; ++__l0) {{\n"
            f"    __acc = {combine_expr(self.op, '__acc', '_src[__l0]', ctype)};\n"
            "    _dst[__l0] = __acc;\n"
            "}"
        )
        return nodes.Tasklet(
            label=f"{self.label}_pure",
            inputs={"_src": None},
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )

    def block_collective(
        self, state: dace.SDFGState, sdfg: dace.SDFG, lowering: TileLowering = SEQUENTIAL
    ) -> nodes.Tasklet:
        """Thread ``t`` loads lanes ``t * items`` to ``(t + 1) * items``, the lanes past the tile being the identity,
        ``cub::BlockScan`` scans them, and the thread stores the ones in the tile."""
        threads = collective_group(self, "TileScan")
        ctype = edge_ctype(sdfg, connected_edges(state, self)["_src"])
        width = self.widths[0]
        items = math.ceil(width / threads)
        scan = f"dace::tileops::BlockScan<{ctype}, {threads}>"
        combine = f"dace::tileops::tile_combine<{ctype}, '{BINARY_OPS[self.op].isa_code}'>()"
        lane = f"{THREAD_INDEX} * {items} + __j"
        # The storage is reused by the next run of the node, after every thread has left this one
        code = (
            f"{ctype} __items[{items}];\n"
            f"for (int __j = 0; __j < {items}; ++__j) "
            f"__items[__j] = ({lane} < {width}) ? _src[{lane}] : {identity_literal(self.op, ctype)};\n"
            f"__shared__ typename {scan}::TempStorage __scan_storage;\n"
            f"{scan}(__scan_storage).InclusiveScan(__items, __items, {combine});\n"
            f"for (int __j = 0; __j < {items}; ++__j) if ({lane} < {width}) _dst[{lane}] = __items[__j];\n"
            "__syncthreads();"
        )
        tasklet = nodes.Tasklet(
            label=f"{self.label}_block",
            inputs={"_src": None},
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
        tasklet.environments = {TileOpsCUDABlock.full_class_path()}
        return tasklet
