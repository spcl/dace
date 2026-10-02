# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileReduce``: a reduction inside a tile, along one axis or over all of it."""
import dace
from dace import cpf_lowering, library, properties
from dace.sdfg import nodes

from ..environments import TileOpsAVX2, TileOpsAVX512, TileOpsCUDA, TileOpsNeon, TileOpsScalar, TileOpsSVE
from ..expansions import ExpandTileIsa, ExpandTilePure
from ..isa import require_k1
from ..lanes import nested_loops, tile_offset
from ..operands import connected_edges, edge_ctype, output_edge
from ..ops import BINARY_OPS, REDUCE_OPS
from .tile_op import TileOp


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

    Only the unmasked full reduction of a K=1 tile has a header lowering. Each backend header implements it its own
    way: AVX-512 collapses with ``_mm512_reduce_<op>``, the other CPU backends with a balanced tree over the lanes, and
    CUDA folds ``half2`` pairs.
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

    def __init__(self,
                 name: str,
                 widths: tuple[int, ...],
                 op: str = "+",
                 axis: int | None = None,
                 has_mask: bool = False,
                 location: str | None = None):
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

    def pure_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
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
            code = f"{ctype} __acc = {identity};\n{nested_loops(widths, body)}\n{writeback}"
        else:
            kept = [dim for dim in range(len(widths)) if dim != self.axis]
            kept_widths = [widths[dim] for dim in kept]
            # The reduction loops over all dims under their own lane names, the init loop over the kept dims alone,
            # which renumbers its lane names from ``__l0``.
            reduce_offset = tile_offset(kept_widths, [f"__l{dim}" for dim in kept])
            init_offset = tile_offset(kept_widths)
            combined = combine_expr(self.op, f"_dst[{reduce_offset}]", f"_src[{source_offset}]", ctype)
            code = (f"{nested_loops(kept_widths, f'_dst[{init_offset}] = {identity};')}\n"
                    f"{nested_loops(widths, f'{gate}_dst[{reduce_offset}] = {combined};')}")
        return nodes.Tasklet(
            label=f"{self.label}_pure",
            inputs=dict.fromkeys(["_src", "_mask"] if self.has_mask else ["_src"]),
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )

    def isa_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG, backend: str) -> nodes.Tasklet:
        vlen = require_k1(self)
        ctype = edge_ctype(sdfg, connected_edges(state, self)["_src"])
        # A one-element output is bound by value, so it is assigned bare; a pointer target is dereferenced.
        out_edge = output_edge(state, self, "_dst")
        scalar_destination = out_edge.data.subset is None or out_edge.data.subset.num_elements() == 1
        destination = "_dst" if scalar_destination else "_dst[0]"
        code = f"{destination} = dace::tileops::tile_reduce<{ctype}, {vlen}, '{BINARY_OPS[self.op].isa_code}'>(_src);"
        return nodes.Tasklet(
            label=f"{self.label}_{backend}",
            inputs={"_src": None},
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
