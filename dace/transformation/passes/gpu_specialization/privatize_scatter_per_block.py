# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Privatize a kernel's scatter reduction into an array too large for shared memory per thread block, in global memory.

A kernel accumulating ``A[f(i, B), ...] (op)= v`` updates ``A`` with one global atomic per write. The GPU mirror of the
CPU privatization (:mod:`~dace.transformation.passes.cpu_specialization.range_privatize_scatter_reductions`) gives each
thread block a private copy of ``A``: its lanes fold into the copy, and after a barrier the block commits each element
with one atomic. The code generator keeps a copy that fits in shared memory on its own
(:func:`~dace.codegen.targets.cpu.block_private_capacities`); this pass gives the larger ones a ``[blocks, elements]``
buffer in global memory, one slice per block, and names it on the thread-block map in ``Map.gpu_block_private``.

A slice pays only when a block covers many more writes than the copy has elements, so the kernel is grid-strided: a
grid of at most the blocks the device holds resident
(:meth:`~dace.transformation.passes.gpu_specialization.resident_blocks.DeviceLimits.resident_blocks`), each walking
many iterations. The target must be an array of at
most ``compiler.cuda.block_private_global_elements`` elements whose size does not grow with the iteration space, which
the kernel only reduces into with one associative, commutative operator with an identity and never reads
(:func:`~dace.codegen.targets.cpu.block_private_target`). Anything else keeps the atomics.
"""

from typing import Any

from dace import SDFG, Memlet, dtypes, properties
from dace.config import Config
from dace.frontend import operations
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.gpu_specialization.coarsen_scatter_kernels import COVERAGE
from dace.transformation.passes.gpu_specialization.grid_stride_kernels import (
    as_int,
    block_threads,
    strip_mined_pair,
    thread_block_child,
)
from dace.transformation.passes.gpu_specialization.resident_blocks import DeviceLimits, device_limits


@properties.make_properties
class PrivatizeScatterPerBlock(ppl.Pass):
    """Give every large scatter target of a kernel a per-block slice of a global buffer (module docstring)."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors | ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """:returns: The number of targets privatized, or ``None`` if none."""
        elements = Config.get("compiler", "cuda", "block_private_global_elements")
        if elements <= 0:
            return None
        kernels = [
            (n, s)
            for n, s in sdfg.all_nodes_recursive()
            if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_Device
        ]
        limits: list[DeviceLimits] = []  # asked for at the first kernel that gets that far, once
        count = sum(self.privatize(state, entry, elements, limits) for entry, state in kernels)
        return count or None

    def privatize(self, state: SDFGState, entry: nodes.MapEntry, elements: int, limits: list[DeviceLimits]) -> int:
        """Privatize the large scatter targets of the kernel ``entry``. :returns: How many."""
        from dace.codegen.targets.cpu import block_private_target, indirect_write

        # Cheapest first: a reduction leaves the kernel
        wcr = [e for e in state.in_edges(state.exit_node(entry)) if e.data.wcr is not None and e.data.data]
        if not wcr:
            return 0
        lanes = strip_mined_pair(state, entry)
        strided = lanes is None
        if strided:
            # Already grid-strided: a device-sized grid of blocks, each walking the iteration space
            lanes = thread_block_child(state, entry)
            threads = None if lanes is None else as_int(lanes.map.range.num_elements())
        else:
            threads = block_threads(entry, lanes)
        if lanes is None or threads is None:
            return 0
        limits[:] = limits or [device_limits()]
        blocks = limits[0].resident_blocks(threads)
        targets = []
        # The writes as the lanes' map sees them, where the code generator decides what a block privatizes
        params = set(entry.map.params) | set(lanes.map.params)
        for e in state.in_edges(state.exit_node(lanes)):
            if e.data.wcr is None or not e.data.data:
                continue
            name, desc = e.data.data, state.sdfg.arrays[e.data.data]
            if name in targets or not (
                e.data.dynamic or {str(s) for s in e.data.subset.free_symbols} & params or indirect_write(e.data)
            ):
                continue  # one block-invariant element: a tree reduction, not a scatter
            size = as_int(desc.total_size)
            if (
                size is not None
                and not Config.get("compiler", "cuda", "block_private_shared_elements") < size <= elements
            ):
                continue  # fits shared memory, or no slice holds it
            if isinstance(desc.dtype, dtypes.vector):
                continue
            redtype = operations.detect_reduction_type(e.data.wcr)
            if redtype == dtypes.ReductionType.Custom or dtypes.reduction_identity(desc.dtype, redtype) is None:
                continue
            # A block must cover many more iterations than its slice has elements
            grid = None if strided else as_int(entry.map.range.num_elements())
            if grid is not None and size is not None and grid * threads < COVERAGE * size * min(blocks, grid):
                continue
            if block_private_target(state.sdfg, state, lanes, name, desc):
                targets.append(name)
        if not targets:
            return 0
        if not strided:
            from dace.transformation.dataflow.gpu_grid_stride_tiling import GPUGridStridedTiling

            where = {"outer_map_entry": entry, "inner_map_entry": lanes}
            options = {"max_grid_dim": blocks, "block_dim": threads}
            if not GPUGridStridedTiling.can_be_applied_to(state.sdfg, options=options, **where):
                return 0
            GPUGridStridedTiling.apply_to(state.sdfg, options=options, save=False, **where)
            # ``entry`` is now the sequential walk under the new device map's thread-block map
            lanes = state.entry_node(entry)
        for name in targets:
            desc = state.sdfg.arrays[name]
            size = as_int(desc.total_size)
            buffer, _ = state.sdfg.add_array(
                f"{name}_block_private",
                [blocks, elements if size is None else size],
                desc.dtype,
                storage=dtypes.StorageType.GPU_Global,
                transient=True,
                lifetime=dtypes.AllocationLifetime.Persistent,
                find_new_name=True,
            )
            # The buffer is an argument of the kernel; the code generator addresses it
            access = state.add_access(buffer)
            state.add_nedge(lanes, access, Memlet())
            state.add_nedge(access, state.exit_node(lanes), Memlet())
            lanes.map.gpu_block_private = {**lanes.map.gpu_block_private, name: buffer}
        return len(targets)


__all__ = ["PrivatizeScatterPerBlock"]
