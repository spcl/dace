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
next parameter); without one it is ``block_min_lanes x block_threads / block_min_lanes``.

A one-parameter kernel is not coarsened (folding a streaming map measured no gain on MI300A, see
:mod:`~dace.transformation.passes.gpu_specialization.grid_stride_kernels`): it only gets a ``block_threads``-wide
block, which ``AddThreadBlockMaps`` then makes explicit.
"""

from typing import Any

from dace import SDFG, SDFGState, dtypes, properties, subsets
from dace.config import Config
from dace.sdfg import nodes
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.passes.gpu_block_size_selection import unit_stride_counts
from dace.transformation.passes.gpu_specialization.block_tile_kernels import is_kernel

def coalesced_param(state: SDFGState, entry: nodes.MapEntry, sdfg: SDFG) -> str:
    """The parameter of ``entry`` that indexes the unit-stride dimension of the most accesses in its scope (counted
    as :func:`~dace.transformation.passes.gpu_block_size_selection.unit_stride_counts` counts them for the block
    shape); the last parameter on a tie (the order the frontend and canonicalization already chose)."""
    counts = unit_stride_counts(sdfg, state, entry)[::-1]
    best = max(counts)
    return [p for p, count in zip(entry.map.params, counts) if count == best][-1]


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
        lanes = Config.get("compiler", "cuda", "block_min_lanes")
        bx, by, bz = [int(b) for b in (kernel.map.gpu_block_size or [lanes, max(1, self.block_threads // lanes), 1])]
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
