# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Give every GPU kernel an explicit thread-block map along its coalesced axis, and each thread several elements of
an outer axis.

The offload leaves a kernel as one ``GPU_Device`` map, and ``AddThreadBlockMaps`` later cuts it by the configured
block size along its LAST parameters, whatever the accesses say. This pass decides the block from the accesses
instead::

    GPU_Device map (o, c):                       GPU_Device map (g_o, g_c):            # one block per tile
        body(o, c)                        ->         GPU_ThreadBlock map (t_o, t_c):   # lanes along c
                                                         map o in t_o : t_o + F (Sequential):
                                                             body(o, c = t_c)

``c`` is the coalesced axis: the parameter that indexes the unit-stride dimension of the most accesses. It becomes
``threadIdx.x``, so a wavefront's lanes touch adjacent elements on every step of the thread's loop. ``o`` is the
outermost other parameter (``threadIdx.y``); each thread walks ``factor`` consecutive points of it, so the block
reuses what neighbouring points of ``o`` read (a stencil's rows). The block keeps the domain-matched shape
:mod:`~dace.transformation.passes.gpu_block_size_selection` chose (``x`` on ``c``, ``y`` on ``o``, ``z`` on the
next parameter); without one it is ``LANES x 256 / LANES``.

A one-parameter kernel is not coarsened (folding a streaming map measured no gain on MI300A, see
:mod:`~dace.transformation.passes.gpu_specialization.grid_stride_kernels`): it only gets a ``block_threads``-wide
block, which ``AddThreadBlockMaps`` then makes explicit.
"""

from typing import Any

from dace import SDFG, SDFGState, data, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.passes.gpu_specialization.block_tile_kernels import is_kernel

#: Threads along the coalesced axis of a multi-dimensional kernel: one MI300A wavefront.
LANES = 64


def coalesced_param(state: SDFGState, entry: nodes.MapEntry, sdfg: SDFG) -> str:
    """The parameter of ``entry`` that indexes the unit-stride dimension of the most accesses in its scope; the last
    parameter on a tie (the order the frontend and canonicalization already chose)."""
    params = entry.map.params
    counts = dict.fromkeys(params, 0)
    for edge in state.scope_subgraph(entry, include_entry=False, include_exit=False).edges():
        memlet = edge.data
        desc = sdfg.arrays.get(memlet.data) if memlet.data else None
        if not isinstance(desc, data.Array) or memlet.subset is None or len(desc.strides) == 0:
            continue
        unit = [d for d, stride in enumerate(desc.strides) if symbolic.issymbolic(stride) is False and stride == 1]
        for dim in unit:
            begin = memlet.subset.ndrange()[dim][0]
            for name in {str(s) for s in symbolic.pystr_to_symbolic(begin).free_symbols} & set(params):
                counts[name] += 1
    best = max(counts.values())
    return [p for p in params if counts[p] == best][-1]


@properties.make_properties
class TileKernelsCoalesced(ppl.Pass):
    """See the module docstring."""

    CATEGORY: str = "Device Specialization"

    factor = properties.Property(dtype=int, default=4, desc="Points of the outer axis each thread computes.")
    block_threads = properties.Property(dtype=int, default=256, desc="Threads per block.")

    def __init__(self, factor: int = 4, block_threads: int = 256) -> None:
        super().__init__()
        self.factor = factor
        self.block_threads = block_threads

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> dict[str, tuple] | None:
        """:returns: per kernel label, ``(coalesced, outer, factor)``, or None if no kernel was tiled."""
        kernels = [
            (state, node)
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, nodes.MapEntry) and is_kernel(state, node)
        ]
        tiled = {}
        for state, kernel in kernels:
            if xfh.gpu_map_has_explicit_threadblocks(state, kernel):
                continue
            params = kernel.map.params
            if len(params) == 1:
                kernel.map.gpu_block_size = [self.block_threads, 1, 1]
                tiled[kernel.map.label] = (params[0], None, 1)
                continue
            coalesced = coalesced_param(state, kernel, state.sdfg)
            outer = next(p for p in params if p != coalesced)
            self.tile(state, kernel, coalesced, outer)
            tiled[kernel.map.label] = (coalesced, outer, self.factor)
        return tiled or None

    def tile(self, state: SDFGState, kernel: nodes.MapEntry, coalesced: str, outer: str) -> None:
        """Grid over tiles, an explicit thread-block map (``coalesced`` is ``threadIdx.x``, ``outer`` ``.y``) and a
        sequential loop of ``factor`` points of ``outer`` per thread."""
        # threadIdx.x is the last parameter of the thread-block map, .y the one before it.
        rest = [p for p in kernel.map.params if p not in (coalesced, outer)]
        bx, by, bz = [int(b) for b in (kernel.map.gpu_block_size or [LANES, max(1, self.block_threads // LANES), 1])]
        perm = [kernel.map.params.index(p) for p in rest + [outer, coalesced]]
        ranges = kernel.map.range.ranges
        kernel.map.params = [kernel.map.params[i] for i in perm]
        kernel.map.range = subsets.Range([ranges[i] for i in perm])
        factor = max(1, self.factor)
        sizes = {coalesced: bx, outer: by * factor}
        if rest and bz > 1:
            sizes[rest[-1]] = bz
        else:
            bz = 1
        MapTiling.apply_to(
            state.sdfg,
            options={
                "prefix": "g",
                "tile_sizes": [sizes.get(p, 1) for p in kernel.map.params],
                "tile_trivial": False,
                "divides_evenly": False,
            },
            map_entry=kernel,
        )
        grid, *_ = xfh.get_parent_map(state, kernel)
        MapTiling.apply_to(
            state.sdfg,
            options={
                "prefix": "t",
                "tile_sizes": [factor if p == outer else 1 for p in kernel.map.params],
                "tile_trivial": True,
                "divides_evenly": False,
            },
            map_entry=kernel,
        )
        block, *_ = xfh.get_parent_map(state, kernel)
        grid.map.schedule = dtypes.ScheduleType.GPU_Device
        grid.map.gpu_block_size = [bx, by, bz]
        for name in ("gpu_launch_bounds", "gpu_min_warps_per_eu", "gpu_maxnreg", "instrument"):
            setattr(grid.map, name, getattr(kernel.map, name))
        block.map.schedule = dtypes.ScheduleType.GPU_ThreadBlock
        block.map.gpu_block_size = None
        block.map.instrument = dtypes.InstrumentationType.No_Instrumentation
        kernel.map.schedule = dtypes.ScheduleType.Sequential
        kernel.map.gpu_block_size = None
        kernel.map.instrument = dtypes.InstrumentationType.No_Instrumentation
