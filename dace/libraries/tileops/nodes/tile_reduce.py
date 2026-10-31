# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileReduce``: a reduction inside a tile, along one axis or over all of it."""

import dataclasses

import dace
from dace import cpf_lowering, library, properties, tile_functions
from dace.libraries.tileops.dispatch import TileGroup
from dace.libraries.tileops.environments import (
    TileOpsAVX2,
    TileOpsAVX512,
    TileOpsCUDA,
    TileOpsCUDABlock,
    TileOpsNeon,
    TileOpsScalar,
    TileOpsSVE,
)
from dace.libraries.tileops.expansions import THREAD_INDEX, ExpandTileIsa, ExpandTilePure
from dace.libraries.tileops.isa import element_code, pointer, require_k1
from dace.libraries.tileops.lanes import SEQUENTIAL, LaneDistribution, TileLowering, nested_loops, tile_offset
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.libraries.tileops.operands import connected_edges, edge_ctype, output_edge
from dace.libraries.tileops.ops import BINARY_OPS, REDUCE_OPS
from dace.sdfg import nodes
from dace.tile_functions import TileFunction


def identity_literal(op: str, ctype: str) -> str:
    """The C++ identity of a reduction at ``ctype``."""
    if op == "+":
        return f"{ctype}(0)"
    if op == "*":
        return f"{ctype}(1)"
    if op == "min":
        return f"std::numeric_limits<{ctype}>::max()"
    if op == "max":
        return f"std::numeric_limits<{ctype}>::lowest()"
    raise ValueError(f"unknown op {op!r}")


def collective_group(node: TileOp, what: str) -> int:
    """The threads of the group a thread-block collective of ``node`` runs on, which a ``WARP`` node has not yet."""
    if node.group is TileGroup.WARP:
        raise NotImplementedError(f"{what}: a WARP group has no lowering yet, use BLOCK or THREAD")
    return node.group_threads()


def combine_expr(op: str, acc: str, value: str, ctype: str) -> str:
    """The C++ expression combining the accumulator and a value; ``ctype`` names ``std::min<T>`` for the C dialect."""
    if op == "+":
        return f"{acc} + {value}"
    if op == "*":
        return f"{acc} * {value}"
    if op in ("min", "max"):
        template = f"<{ctype}>" if cpf_lowering.standalone_c() else ""
        return f"std::{op}{template}({acc}, {value})"
    raise ValueError(f"unknown op {op!r}")


@library.expansion
class ExpandTileReducePure(ExpandTilePure):
    pass


@library.expansion
class ExpandTileReduceScalar(ExpandTileIsa):
    environments = [TileOpsScalar]
    backend = "scalar"


@library.expansion
class ExpandTileReduceAVX512(ExpandTileIsa):
    environments = [TileOpsAVX512]
    backend = "avx512"


@library.expansion
class ExpandTileReduceAVX2(ExpandTileIsa):
    environments = [TileOpsAVX2]
    backend = "avx2"


@library.expansion
class ExpandTileReduceNeon(ExpandTileIsa):
    environments = [TileOpsNeon]
    backend = "neon"


@library.expansion
class ExpandTileReduceSVE(ExpandTileIsa):
    environments = [TileOpsSVE]
    backend = "sve"


@library.expansion
class ExpandTileReduceCUDA(ExpandTileIsa):
    environments = [TileOpsCUDA]
    backend = "cuda"


@library.node
class TileReduce(TileOp):
    """Reduces the tile ``_src`` along ``axis``, or fully to one element when ``axis`` is ``None``.

    The result ``_dst`` has the shape of the kept dims, or is one element. With ``has_mask`` the lanes where ``_mask``
    is false contribute the identity of ``op``. Accumulating across tiles is the caller's job, typically a WCR memlet
    on the output edge.

    Only the unmasked full reduction of a K=1 tile has an ISA lowering: the CPU backends fold a balanced tree over the
    lanes, CUDA folds ``half2`` pairs.
    """

    implementations = {
        "pure": ExpandTileReducePure,
        "scalar": ExpandTileReduceScalar,
        "avx512": ExpandTileReduceAVX512,
        "avx2": ExpandTileReduceAVX2,
        "neon": ExpandTileReduceNeon,
        "sve": ExpandTileReduceSVE,
        "cuda": ExpandTileReduceCUDA,
    }
    default_implementation = "pure"

    op = properties.Property(
        dtype=str,
        allow_none=False,
        default="+",
        desc="The reduction: '+', '*', 'min' or 'max'.",
    )
    axis = properties.Property(
        dtype=int,
        allow_none=True,
        default=None,
        desc="The tile dim to reduce along; ``None`` reduces the whole tile to one element.",
    )
    has_mask = properties.Property(
        dtype=bool,
        allow_none=False,
        default=False,
        desc="Whether the ``_mask`` input connector gates the lanes.",
    )

    def __init__(
        self,
        name: str,
        widths: tuple[int, ...],
        op: str = "+",
        axis: int | None = None,
        has_mask: bool = False,
        location: str | None = None,
    ):
        if op not in REDUCE_OPS:
            raise ValueError(f"TileReduce: unknown op {op!r}; allowed: {REDUCE_OPS}")
        if not 1 <= len(widths) <= 3:
            raise ValueError(f"TileReduce: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if axis is not None and not 0 <= axis < len(widths):
            raise ValueError(f"TileReduce: axis {axis} out of range for K={len(widths)}")
        inputs = ["_src", "_mask"] if has_mask else ["_src"]
        super().__init__(name, location=location, inputs=dict.fromkeys(inputs), outputs={"_dst"})
        self.widths = list(widths)
        self.op = op
        self.axis = axis
        self.has_mask = has_mask

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        in_edges = connected_edges(state, self)
        out_conns = {edge.src_conn for edge in state.out_edges(self)}
        if "_src" not in in_edges:
            raise ValueError(f"{self.label}: required input '_src' not connected")
        if "_dst" not in out_conns:
            raise ValueError(f"{self.label}: required output '_dst' not connected")
        if self.has_mask and "_mask" not in in_edges:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")

    def can_lower_to_isa(self, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        return self.axis is None and not self.has_mask and len(self.widths) == 1

    def pure_tasklet(
        self, state: dace.SDFGState, sdfg: dace.SDFG, lowering: TileLowering = SEQUENTIAL
    ) -> nodes.Tasklet:
        widths = list(self.widths)
        ctype = edge_ctype(sdfg, connected_edges(state, self)["_src"])
        identity = identity_literal(self.op, ctype)
        source_offset = tile_offset(widths)
        # The mask gates each lane's contribution: an inactive lane adds nothing.
        gate = f"if (_mask[{source_offset}]) " if self.has_mask else ""
        if self.axis is None:
            out_edge = output_edge(state, self, "_dst")
            scalar_destination = out_edge.data.subset is None or out_edge.data.subset.num_elements() == 1
            writeback = "_dst = __acc;" if scalar_destination else "_dst[0] = __acc;"
            body = f"{gate}__acc = {combine_expr(self.op, '__acc', f'_src[{source_offset}]', ctype)};"
            code = f"{ctype} __acc = {identity};\n{nested_loops(widths, body, lowering=lowering)}\n{writeback}"
        else:
            kept = [dim for dim in range(len(widths)) if dim != self.axis]
            kept_widths = [widths[dim] for dim in kept]
            # The reduction loops over all dims under their own lane names, the init loop over the kept dims alone,
            # which renumbers its lane names from ``__l0``.
            reduce_offset = tile_offset(kept_widths, [f"__l{dim}" for dim in kept])
            init_offset = tile_offset(kept_widths)
            combined = combine_expr(self.op, f"_dst[{reduce_offset}]", f"_src[{source_offset}]", ctype)
            code = (
                f"{nested_loops(kept_widths, f'_dst[{init_offset}] = {identity};', lowering=lowering)}\n"
                f"{nested_loops(widths, f'{gate}_dst[{reduce_offset}] = {combined};', lowering=lowering)}"
            )
        return nodes.Tasklet(
            label=f"{self.label}_pure",
            inputs=dict.fromkeys(["_src", "_mask"] if self.has_mask else ["_src"]),
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )

    def block_collective(
        self, state: dace.SDFGState, sdfg: dace.SDFG, lowering: TileLowering = SEQUENTIAL
    ) -> nodes.Tasklet | None:
        """A full reduction: each thread folds its share of the lanes, ``cub::BlockReduce`` folds the threads, and
        every thread reads the result from shared memory. A reduction along an axis runs on one thread."""
        if self.axis is not None:
            return None
        threads = collective_group(self, "TileReduce")
        widths = list(self.widths)
        ctype = edge_ctype(sdfg, connected_edges(state, self)["_src"])
        gate = f"if (_mask[{tile_offset(widths)}]) " if self.has_mask else ""
        body = f"{gate}__acc = {combine_expr(self.op, '__acc', f'_src[{tile_offset(widths)}]', ctype)};"
        spread = dataclasses.replace(lowering, distribution=LaneDistribution(THREAD_INDEX, threads))
        partial = nested_loops(widths, body, lowering=spread)
        reduce = f"dace::tileops::BlockReduce<{ctype}, {threads}>"
        combine = f"dace::tileops::tile_combine<{ctype}, '{BINARY_OPS[self.op].isa_code}'>()"
        out_edge = output_edge(state, self, "_dst")
        scalar_destination = out_edge.data.subset is None or out_edge.data.subset.num_elements() == 1
        destination = "_dst" if scalar_destination else "_dst[0]"
        # The storage is reused by the next run of the node, after every thread has left this one
        code = (
            f"{ctype} __acc = {identity_literal(self.op, ctype)};\n{partial}\n"
            f"__shared__ typename {reduce}::TempStorage __reduce_storage;\n"
            f"__shared__ {ctype} __reduced;\n"
            f"const {ctype} __total = {reduce}(__reduce_storage).Reduce(__acc, {combine});\n"
            f"if ({THREAD_INDEX} == 0) __reduced = __total;\n"
            "__syncthreads();\n"
            f"{destination} = __reduced;\n"
            "__syncthreads();"
        )
        tasklet = nodes.Tasklet(
            label=f"{self.label}_block",
            inputs=dict.fromkeys(["_src", "_mask"] if self.has_mask else ["_src"]),
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
        tasklet.environments = {TileOpsCUDABlock.full_class_path()}
        return tasklet

    def isa_tasklet(
        self, state: dace.SDFGState, sdfg: dace.SDFG, backend: str, lowering: TileLowering = SEQUENTIAL
    ) -> nodes.Tasklet:
        vlen = require_k1(self, lowering=lowering)
        ctype = edge_ctype(sdfg, connected_edges(state, self)["_src"])
        # A one-element output is bound by value, so it is assigned bare; a pointer target is dereferenced.
        out_edge = output_edge(state, self, "_dst")
        scalar_destination = out_edge.data.subset is None or out_edge.data.subset.num_elements() == 1
        destination = "_dst" if scalar_destination else "_dst[0]"
        op = BINARY_OPS[self.op].isa_code
        if backend in tile_functions.ISAS:
            source = pointer("_src", connected_edges(state, self)["_src"])
            call = f"{TileFunction('reduce', backend, element_code(ctype), vlen, op).name}({source})"
        else:
            call = f"dace::tileops::tile_reduce<{ctype}, {vlen}, '{op}'>(_src)"
        code = f"{destination} = {call};"
        return nodes.Tasklet(
            label=f"{self.label}_{backend}",
            inputs={"_src": None},
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
