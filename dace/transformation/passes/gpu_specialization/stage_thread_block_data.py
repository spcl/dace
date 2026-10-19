# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stage the tile a thread block reads (and, on request, writes) through shared memory.

A kernel whose block computes on a tile reads, through the memlets between the ``GPU_Device`` map and its
``GPU_ThreadBlock`` map, exactly the region of each array that tile touches::

    GPU_Device map (b_i, b_j):                      # one block per tile
        GPU_ThreadBlock map (i, j) in tile:
            out[i, j] = f(A[i, 0:3], B[j])          # every thread of the block reads A[tile_i, 0:3]

This pass gives each such region a ``GPU_Shared`` buffer of its shape and fills it with one block-collective
``CopyLibraryNode`` (``SharedMemoryCollective``, rows moved in aligned 16-byte words), so the block reads global
memory once, coalesced, and every thread reads its operands from shared memory::

    GPU_Device map (b_i, b_j):
        copy A[tile] -> shm_A ; copy B[tile] -> shm_B    # chained, no barrier in between
        __syncthreads()                                  # one barrier for the whole stage
        GPU_ThreadBlock map (i, j) in tile:
            shm_out[i, j] = f(shm_A[i, 0:3], shm_B[j])
        __syncthreads()                                  # with stage_stores
        copy shm_out -> out[tile]

A kernel without a thread-block map gets one first (:class:`~dace.transformation.dataflow.add_threadblock_map.
AddThreadBlockMap`). The buffer of a region is sized by the region's static bound (a partial edge tile copies
only the cells inside the array); a region whose bound is not a constant -- a data-dependent gather, an axis the
tile does not cut -- is not staged, nor is one whose buffer would overflow ``shared_memory_bytes``. A region is
staged for reading only if the kernel does not also write the array, and for writing only if every thread writes
its cell (no dynamic or write-conflict-resolved memlet), so the copy back never overwrites a cell with stale data.
"""

import copy
import numbers
from typing import Any

import sympy

from dace import SDFG, Memlet, SDFGState, data, dtypes, properties, subsets, symbolic
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.sdfg import nodes
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap
from dace.transformation.dataflow.local_storage import InLocalStorage, OutLocalStorage
from dace.transformation.passes.gpu_specialization.block_tile_kernels import is_kernel
from dace.transformation.passes.shared_memory_synchronization import BARRIER

#: Prefix of the shared buffers the pass creates.
SHARED_PREFIX = "shm_"


def static_bound(extent: Any) -> int | None:
    """A constant upper bound of a region's ``extent``: the extent itself, or the smallest constant argument of a
    ``Min`` (a tile clipped at the array's end); None when there is none."""
    extent = sympy.sympify(extent)
    if extent.is_Integer:
        return int(extent)
    if isinstance(extent, sympy.Min):
        bounds = [b for b in (static_bound(arg) for arg in extent.args) if b is not None]
        return min(bounds) if bounds else None
    approx = sympy.sympify(symbolic.overapproximate(extent))
    return int(approx) if approx.is_Integer else None


def exact(subset: subsets.Range) -> subsets.Range:
    """``subset`` with every bound its exact expression: the memlets of a tiled kernel carry ``SymExpr`` bounds
    (``Min(N - 1, g + 63)`` over-approximated as ``g + 63``), and the copy moves exactly the cells read."""
    return subsets.Range(
        [tuple(b.expr if isinstance(b, symbolic.SymExpr) else b for b in bounds) for bounds in subset.ranges]
    )


def clipped(subset: subsets.Range, shape: list[Any]) -> subsets.Range:
    """``subset`` (exact) with every range ended at the array's last index, so a partial edge tile copies only the
    cells inside the array."""
    return subsets.Range(
        [(begin, sympy.Min(end, extent - 1), step) for (begin, end, step), extent in zip(exact(subset).ranges, shape)]
    )


def add_barrier(state: SDFGState, label: str) -> nodes.Tasklet:
    """A ``__syncthreads()`` tasklet in ``state``."""
    return state.add_tasklet(label, {}, {}, BARRIER, language=dtypes.Language.CPP, side_effects=True)


def thread_block_map(state: SDFGState, kernel: nodes.MapEntry) -> nodes.MapEntry | None:
    """The one ``GPU_ThreadBlock`` map directly inside ``kernel`` that every data edge into the kernel's body
    enters, or None."""
    children = [
        node
        for node in state.scope_children()[kernel]
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    ]
    return children[0] if len(children) == 1 else None


def written_arrays(state: SDFGState, entry: nodes.MapEntry) -> set[str]:
    """The arrays the scope of ``entry`` writes."""
    exit_node = state.exit_node(entry)
    return {edge.data.data for edge in state.in_edges(exit_node) if not edge.data.is_empty()}


@properties.make_properties
class StageThreadBlockData(ppl.Pass):
    """Stage every bounded region a kernel's thread-block map reads -- and, with ``stage_stores``, writes --
    through shared memory with block-collective copies and one barrier per direction (see the module docstring).
    """

    CATEGORY: str = "Device Specialization"

    stage_loads = properties.Property(dtype=bool, default=True, desc="Stage the regions the thread blocks read.")
    stage_stores = properties.Property(dtype=bool, default=False, desc="Stage the regions the thread blocks write.")
    aligned_words = properties.Property(
        dtype=bool, default=True, desc="Move contiguous rows in aligned 16-byte words (CopyLibraryNode.aligned_words)."
    )
    shared_memory_bytes = properties.Property(
        dtype=int, default=48 * 1024, desc="Shared memory one block may use for the staged buffers."
    )

    def __init__(
        self,
        stage_loads: bool = True,
        stage_stores: bool = False,
        aligned_words: bool = True,
        shared_memory_bytes: int = 48 * 1024,
    ) -> None:
        super().__init__()
        self.stage_loads = stage_loads
        self.stage_stores = stage_stores
        self.aligned_words = aligned_words
        self.shared_memory_bytes = shared_memory_bytes

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> dict[str, list[str]] | None:
        """Stage the data of every kernel of ``sdfg`` and its nested SDFGs.

        :returns: per kernel label, the arrays staged (``load:<name>`` / ``store:<name>``); None if none were.
        """
        kernels = [
            (state, node)
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, nodes.MapEntry) and is_kernel(state, node)
        ]
        staged: dict[str, list[str]] = {}
        for state, kernel in kernels:
            kernel, block = self.ensure_thread_block_map(state, kernel)
            if block is None:
                continue
            budget = [self.shared_memory_bytes]
            done = []
            if self.stage_loads:
                done += [f"load:{name}" for name in self.stage_reads(state, kernel, block, budget)]
            if self.stage_stores:
                done += [f"store:{name}" for name in self.stage_writes(state, kernel, block, budget)]
            if done:
                staged[kernel.map.label] = done
        return staged or None

    def ensure_thread_block_map(
        self, state: SDFGState, kernel: nodes.MapEntry
    ) -> tuple[nodes.MapEntry, nodes.MapEntry | None]:
        """``(kernel, its thread-block map)``, adding the thread-block map when the kernel has none."""
        block = thread_block_map(state, kernel)
        if block is not None:
            return kernel, block
        if xfh.gpu_map_has_explicit_threadblocks(state, kernel):
            return kernel, None
        # AddThreadBlockMap tiles the kernel: the original entry becomes the thread-block map, a new one the grid.
        AddThreadBlockMap.apply_to(state.sdfg, map_entry=kernel, verify=False)
        if kernel.map.schedule != dtypes.ScheduleType.GPU_ThreadBlock:
            return kernel, None
        return state.entry_node(kernel), kernel

    def region(self, sdfg: SDFG, memlet: Memlet) -> tuple[list[int], int] | None:
        """``(buffer shape bound, bytes)`` of a stageable region of a global array, or None."""
        if memlet.is_empty() or memlet.dynamic or memlet.wcr is not None:
            return None
        desc = sdfg.arrays[memlet.data]
        if not isinstance(desc, data.Array) or isinstance(desc, data.View):
            return None
        if desc.storage != dtypes.StorageType.GPU_Global:
            return None
        # The buffer takes the shape LocalStorage gives it, which a nested SDFG inside the tile is rebased onto; it
        # must be a compile-time constant, a specialized symbol (``sdfg.specialize``) included.
        constants = {
            symbolic.symbol(name): int(value)
            for name, value in sdfg.constants.items()
            if isinstance(value, numbers.Integral)
        }
        extents = [symbolic.overapproximate(extent).simplify() for extent in memlet.bounding_box_size()]
        shape = [static_bound(sympy.sympify(extent).subs(constants)) for extent in extents]
        if any(extent is None or extent < 1 for extent in shape):
            return None
        size = 1
        for extent in shape:
            size *= extent
        return shape, size * desc.dtype.bytes

    def shared_buffer(self, sdfg: SDFG, name: str) -> None:
        """Make the transient ``name`` (of a compile-time constant shape) a static shared buffer."""
        desc = sdfg.arrays[name]
        desc.storage = dtypes.StorageType.GPU_Shared
        desc.lifetime = dtypes.AllocationLifetime.Scope

    def stage_reads(self, state: SDFGState, kernel: nodes.MapEntry, block: nodes.MapEntry, budget: list[int]) -> list:
        """Stage the regions ``block`` reads; returns the arrays staged."""
        sdfg = state.sdfg
        written = written_arrays(state, kernel)
        candidates = []
        for edge in state.in_edges(block):
            if edge.src is not kernel or edge.data.is_empty() or edge.data.data in written:
                continue
            found = self.region(sdfg, edge.data)
            if found is not None and found[1] <= budget[0]:
                budget[0] -= found[1]
                candidates.append((edge.data.data, found[0]))
        buffers = []
        copies = []
        for name, _ in candidates:
            before = {e.dst for e in state.out_edges(kernel)}
            InLocalStorage.apply_to(
                sdfg, options={"array": name, "prefix": SHARED_PREFIX}, verify=False, node_a=kernel, node_b=block
            )
            buffer = next(e.dst for e in state.out_edges(kernel) if e.dst not in before)
            self.shared_buffer(sdfg, buffer.data)
            copies.append(self.copy_in(state, kernel, buffer))
            buffers.append(buffer)
        if not buffers:
            return []
        # The copies run one after the other with no barrier between them; one barrier closes the stage.
        for buffer, nxt in zip(buffers, copies[1:]):
            state.add_edge(buffer, None, nxt, None, Memlet())
        barrier = add_barrier(state, "stage_loads_barrier")
        for buffer in buffers:
            state.add_edge(buffer, None, barrier, None, Memlet())
        state.add_edge(barrier, None, block, None, Memlet())
        return [name for name, _ in candidates]

    def copy_in(self, state: SDFGState, kernel: nodes.MapEntry, buffer: nodes.AccessNode) -> CopyLibraryNode:
        """Replace the edge ``kernel -> buffer`` by a block-collective copy of the region, clipped to the array."""
        sdfg = state.sdfg
        (edge,) = [e for e in state.out_edges(kernel) if e.dst is buffer]
        source = sdfg.arrays[edge.data.data]
        region = clipped(edge.data.subset, list(source.shape))
        local = copy.deepcopy(region)
        local.offset(subsets.Range.from_indices([begin for begin, _, _ in region.ranges]), True)
        node = CopyLibraryNode(f"stage_{edge.data.data}", sync=False, aligned_words=self.aligned_words)
        node.implementation = "SharedMemoryCollective"
        state.add_node(node)
        state.add_edge(
            kernel,
            edge.src_conn,
            node,
            CopyLibraryNode.INPUT_CONNECTOR_NAME,
            Memlet(data=edge.data.data, subset=region),
        )
        state.add_edge(
            node, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, buffer, None, Memlet(data=buffer.data, subset=local)
        )
        state.remove_edge(edge)
        return node

    def stage_writes(self, state: SDFGState, kernel: nodes.MapEntry, block: nodes.MapEntry, budget: list[int]) -> list:
        """Stage the regions ``block`` writes; returns the arrays staged."""
        sdfg = state.sdfg
        block_exit, kernel_exit = state.exit_node(block), state.exit_node(kernel)
        read = {edge.data.data for edge in state.out_edges(kernel) if not edge.data.is_empty()}
        candidates = []
        for edge in state.out_edges(block_exit):
            if edge.dst is not kernel_exit or edge.data.data in read:
                continue
            # Every thread must write its cell: the copy back covers the whole region.
            if any(e.data.dynamic or e.data.wcr is not None for e in state.memlet_tree(edge)):
                continue
            found = self.region(sdfg, edge.data)
            if found is not None and found[1] <= budget[0]:
                budget[0] -= found[1]
                candidates.append((edge.data.data, found[0]))
        if not candidates:
            return []
        barrier = add_barrier(state, "stage_stores_barrier")
        state.add_edge(block_exit, None, barrier, None, Memlet())
        for name, _ in candidates:
            before = {e.src for e in state.in_edges(kernel_exit)}
            OutLocalStorage.apply_to(
                sdfg,
                options={"array": name, "prefix": SHARED_PREFIX},
                verify=False,
                node_a=block_exit,
                node_b=kernel_exit,
            )
            buffer = next(e.src for e in state.in_edges(kernel_exit) if e.src not in before)
            self.shared_buffer(sdfg, buffer.data)
            node = self.copy_out(state, kernel_exit, buffer)
            state.add_edge(barrier, None, node, None, Memlet())
        return [name for name, _ in candidates]

    def copy_out(self, state: SDFGState, kernel_exit: nodes.MapExit, buffer: nodes.AccessNode) -> CopyLibraryNode:
        """Replace the edge ``buffer -> kernel_exit`` by a block-collective copy of the region, clipped to the
        array."""
        sdfg = state.sdfg
        (edge,) = [e for e in state.out_edges(buffer) if e.dst is kernel_exit]
        target = sdfg.arrays[edge.data.data]
        region = clipped(edge.data.subset, list(target.shape))
        local = copy.deepcopy(region)
        local.offset(subsets.Range.from_indices([begin for begin, _, _ in region.ranges]), True)
        node = CopyLibraryNode(f"unstage_{edge.data.data}", sync=False, aligned_words=self.aligned_words)
        node.implementation = "SharedMemoryCollective"
        state.add_node(node)
        state.add_edge(buffer, None, node, CopyLibraryNode.INPUT_CONNECTOR_NAME, Memlet(data=buffer.data, subset=local))
        state.add_edge(
            node,
            CopyLibraryNode.OUTPUT_CONNECTOR_NAME,
            kernel_exit,
            edge.dst_conn,
            Memlet(data=edge.data.data, subset=region),
        )
        state.remove_edge(edge)
        return node
