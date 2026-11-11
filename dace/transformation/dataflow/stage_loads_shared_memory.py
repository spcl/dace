# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stage the global arrays a GPU kernel's threads reuse through per-block shared memory."""

import sympy

from dace import SDFG, Memlet, SDFGState, properties
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap
from dace.transformation.passes.gpu_specialization.block_tile_kernels import is_kernel
from dace.transformation.passes.gpu_specialization.stage_thread_block_data import (
    StageThreadBlockData,
    thread_block_map,
    written_arrays,
)

#: The value every free symbol takes when deciding reuse: sizes are assumed large.
LARGE = 2**20


def has_reuse(memlet: Memlet) -> bool:
    """Whether the scope reads some element of ``memlet``'s region more than once: its access count (``volume``)
    exceeds the region's size, for large symbol values."""
    extra = sympy.sympify(memlet.volume - memlet.subset.num_elements())
    extra = extra.subs(dict.fromkeys(extra.free_symbols, LARGE))
    return bool(extra.is_number and extra > 0)


@properties.make_properties
class StageLoadsThroughSharedMemory(transformation.SingleStateTransformation):
    """Stage the global arrays a ``GPU_Device`` map reads with reuse across its threads through ``GPU_Shared``
    buffers, one per thread-block tile, halo included.

    The map is tiled into thread blocks first if it has no ``GPU_ThreadBlock`` map (``AddThreadBlockMap``, sized
    by ``block_size``). Each staged tile is loaded by a block-collective copy, the copies close with one
    ``__syncthreads()``, and the body reads the shared copy (``StageThreadBlockData.stage_reads``). An array is
    staged if the kernel does not write it, some element is read more than once, and every thread's footprint is
    a compile-time constant (so the tile's is).
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    arrays = properties.ListProperty(element_type=str, default=[], desc="Arrays to stage; empty: every candidate.")
    block_size = properties.ListProperty(
        element_type=int, default=[], desc="Thread-block size when the map is tiled; empty: the map's or default."
    )
    shared_memory_bytes = properties.Property(
        dtype=int, default=48 * 1024, desc="Shared memory one block may use for the staged buffers."
    )

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def staged_arrays(self, state: SDFGState, sdfg: SDFG) -> list[str]:
        """The arrays the transformation stages, in the order the kernel reads them."""
        entry = self.map_entry
        stager = StageThreadBlockData()
        written = written_arrays(state, entry)
        names = OrderedSet()
        for edge in state.in_edges(entry):
            name = edge.data.data
            if (
                edge.data.is_empty()
                or edge.data.dynamic
                or name in written
                or (self.arrays and name not in self.arrays)
            ):
                continue
            # Each thread's footprint: the memlets leaving the map through the edge's connector.
            inner = [e.data for e in state.out_edges_by_connector(entry, "OUT_" + edge.dst_conn[3:])]
            if inner and has_reuse(edge.data) and all(stager.region(sdfg, m) is not None for m in inner):
                names.add(name)
        return list(names)

    def can_be_applied(self, state: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        entry = self.map_entry
        if not is_kernel(state, entry):
            return False
        if thread_block_map(state, entry) is None and not AddThreadBlockMap.can_be_applied_to(sdfg, map_entry=entry):
            return False
        return bool(self.staged_arrays(state, sdfg))

    def apply(self, state: SDFGState, sdfg: SDFG) -> list[str]:
        names = self.staged_arrays(state, sdfg)
        entry = self.map_entry
        if self.block_size and thread_block_map(state, entry) is None:
            entry.map.gpu_block_size = list(self.block_size)
        stager = StageThreadBlockData(shared_memory_bytes=self.shared_memory_bytes)
        kernel, block = stager.ensure_thread_block_map(state, entry)
        return stager.stage_reads(state, kernel, block, [self.shared_memory_bytes], names)
