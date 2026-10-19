# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Domain-matched thread-block size selection for GPU device maps.

A ``GPU_Device`` map with no explicit ``gpu_block_size`` and no inner
``GPU_ThreadBlock`` map falls back at codegen to the single configured default
(``compiler.cuda.default_block_size``). That default is a 1-D block
(``128,1,1``), which is right for a 1-D kernel but wastes the second grid
dimension of a 2-D kernel: a 2-D iteration domain then launches one thread per
row of the block only, so consecutive threads stride across the outer domain
dimension instead of the contiguous one.

This pass picks a thread-block **shaped like the iteration domain**: the nicest cube, rectangle or
line of a fixed thread count that the domain can fill. ``threadIdx.x`` maps to the last
(contiguous) map dimension -- codegen reverses the map range into grid order -- so
``gpu_block_size`` is stored in CUDA ``(x, y, z)`` order, i.e. reversed map-parameter order.

    * 1-D domain              -> a line of 512          (``[512, 1, 1]``)
    * 2-D domain              -> a rectangle of 256     (``N x N`` -> ``[32, 8, 1]``)
    * 3-D domain              -> a box of 512           (``N x N x N`` -> ``[32, 4, 4]``)

The coalesced axis -- the one indexing the most accesses at unit stride, ``x`` on a tie -- starts
at ``compiler.cuda.block_min_lanes`` threads (32: half a CDNA wavefront, a whole NVIDIA warp), so a
wavefront's loads cover at least one contiguous 32-element run; a square or cube would split it
into 16- or 8-element pieces.

The block is grown one doubling at a time, always on the axis with the most domain left to cover
(``extent / block``), so a skewed domain gets a matching rectangle (``8 x 1024`` ->
``[32, 8, 1]``) and an axis never exceeds its extent rounded up to a power of two. Symbolic
extents are taken as large and equal, which keeps the cube. Where axes tie, the one that indexes
the most accesses at unit stride takes the extra threads first (``[16, 8, 4]`` or ``[32, 4, 4]``
rather than a cube that cannot be filled evenly), then ``x``.

The pass never overrides a user-set ``gpu_block_size`` and skips any device map
that already carries an inner ``GPU_ThreadBlock`` / ``GPU_ThreadBlock_Dynamic``
map, or a map still tagged ``is_warp_tile`` and waiting to become one -- those
derive the block size from the thread-block map, and a preset ``gpu_block_size``
would conflict at codegen.
"""

import math
from typing import Any

from dace import SDFG, dtypes
from dace.codegen.common import cuda_emits_tree_reductions
from dace.config import Config
from dace.optionals import required
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: Schedules whose maps derive the block size themselves; a device map wrapping
#: one of these must not also carry a preset ``gpu_block_size``.
THREADBLOCK_SCHEDULES = (dtypes.ScheduleType.GPU_ThreadBlock, dtypes.ScheduleType.GPU_ThreadBlock_Dynamic)

#: Threads per block by domain rank: a line of 512, a square of 256, a cube of 512.
BLOCK_THREADS_BY_RANK = {1: 512, 2: 256, 3: 512}


def warp_width() -> int:
    """Lanes in one hardware warp: 64 on a CDNA wavefront, 32 on NVIDIA.

    Asked of the backend rather than configured, the way
    :mod:`dace.libraries.standard.nodes.reduce` asks it, so one build serves both and neither
    inherits the other's shape.
    """
    from dace.codegen import common

    return 64 if common.get_gpu_backend() == "hip" else 32


#: A device map whose reduction is lowered as a block tree-reduce (a WCR map output, where the active
#: CUDA codegen tree-reduces) wants a DEEP block: more lanes per block-reduce means more of the
#: reduction is folded inside the block (one shared-memory tree) and fewer partial results
#: race through the cross-block atomic. 512 (vs the 128/256 a plain elementwise map takes)
#: keeps well under the 1024 thread/block limit while roughly halving the atomic traffic.
TREE_REDUCTION_BLOCK_SIZE = [512, 1, 1]


def constant_extent(extent) -> int | None:
    """Return ``extent`` as a positive Python ``int`` if it is a known constant, else ``None``.

    A symbolic (non-constant) extent -- the common case for size symbols -- returns
    ``None`` so callers treat the dimension as "large and unknown".
    """
    try:
        value = int(extent)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def next_power_of_two(value: int) -> int:
    """The smallest power of two at least ``value``."""
    return 1 << max(value - 1, 0).bit_length()


def domain_matched_block(extents: list, unit_stride: list[int] | None = None) -> list[int]:
    """A thread block (CUDA ``x, y, z`` order) shaped like ``extents`` (the same order).

    Starts from one thread per axis and doubles, :data:`BLOCK_THREADS_BY_RANK` threads in all, the axis
    with the most domain left per thread -- ``extent / block``, a symbolic extent counting as larger
    than any constant -- never past the axis' extent rounded up to a power of two. Ties go to the axis
    with the most unit-stride accesses (``unit_stride``, same order), then to the lower axis (``x``).
    The axis with the most unit-stride accesses (``x`` on a tie) starts at ``compiler.cuda.block_min_lanes``.
    A domain smaller than the block keeps only the threads it can use.
    """
    rank = len(extents)
    threads = BLOCK_THREADS_BY_RANK[rank]
    unit_stride = unit_stride or [0] * rank
    constants = [constant_extent(extent) for extent in extents]
    caps = [next_power_of_two(c) if c is not None else threads for c in constants]
    if rank == 3:
        # The z axis holds fewer threads than x and y; past the limit the kernel cannot launch.
        caps[2] = min(caps[2], Config.get("compiler", "cuda", "block_size_lastdim_limit"))
    block = [1] * rank
    if rank > 1:
        lanes = max(range(rank), key=lambda axis: (unit_stride[axis], -axis))
        block[lanes] = min(next_power_of_two(Config.get("compiler", "cuda", "block_min_lanes")), caps[lanes], threads)

    def priority(axis: int) -> tuple:
        # Most domain left per thread first (a symbolic extent above every constant, and equal to the others), then
        # the narrower axis, then the one with the most unit-stride accesses, then ``x``.
        left = float("inf") if constants[axis] is None else constants[axis] / block[axis]
        return (left, -block[axis], unit_stride[axis], -axis)

    while math.prod(block) < threads:
        growable = [axis for axis in range(rank) if block[axis] * 2 <= caps[axis]]
        if not growable:
            break
        block[max(growable, key=priority)] *= 2
    return block + [1] * (3 - rank)


def unit_stride_counts(sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry) -> list[int]:
    """Per map parameter, in CUDA ``x, y, z`` order, how many accesses in the map's scope it indexes at unit
    stride: alone, with coefficient one, on the array's stride-1 dimension."""
    from dace import symbolic

    params = list(map_entry.map.params)[::-1]
    counts = [0] * len(params)
    for edge in state.scope_subgraph(map_entry, include_entry=False, include_exit=False).edges():
        if edge.data.data is None or edge.data.subset is None or edge.data.data not in sdfg.arrays:
            continue
        strides = sdfg.arrays[edge.data.data].strides
        for dim, stride in enumerate(strides):
            if symbolic.simplify(stride) != 1 or dim >= len(edge.data.subset.ranges):
                continue
            begin = symbolic.pystr_to_symbolic(edge.data.subset.ranges[dim][0])
            for axis, param in enumerate(params):
                if symbolic.simplify(begin.diff(symbolic.symbol(param))) == 1:
                    counts[axis] += 1
    return counts


def pick_gpu_block_size(gpu_map: nodes.Map, unit_stride: list[int] | None = None) -> list[int] | None:
    """Domain-matched ``gpu_block_size`` for a ``GPU_Device`` ``map``, in CUDA ``(x, y, z)`` order
    (:func:`domain_matched_block` of its reversed range sizes, so the last, contiguous map dimension is
    ``threadIdx.x``). ``None`` past three dimensions, which keeps the configured default."""
    if len(gpu_map.params) > 3:
        return None
    # ``gpu_block_size`` is read in CUDA (x, y, z) order, which is reversed map order: codegen builds
    # ``grid_size = map.range.size()[::-1]`` and zips it with the block.
    return domain_matched_block(gpu_map.range.size()[::-1], unit_stride)


def is_block_reduce(node) -> bool:
    """True iff ``node`` is a ``Reduce`` library node whose reduction folds ACROSS the thread
    block (``gpucub::BlockReduce``, sized by the block) rather than sequentially per thread. A
    ``Sequential`` Reduce is a per-thread loop -- its block size does not deepen any tree."""
    from dace.libraries.standard.nodes.reduce import Reduce

    return isinstance(node, Reduce) and node.schedule != dtypes.ScheduleType.Sequential


def scope_contains_block_reduce(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """True iff ``map_entry``'s scope (incl. nested SDFGs) holds a block-level ``Reduce`` (see
    :func:`is_block_reduce`), which the CUDA-block expansion lowers to a ``gpucub::BlockReduce<T,
    N>`` whose ``N`` is this map's flattened block size (via :func:`devicelevel_block_size`) --
    so a bigger block deepens the block-reduce directly."""
    for node in state.all_nodes_between(map_entry, required(state.exit_node(map_entry))):
        if is_block_reduce(node):
            return True
        if isinstance(node, nodes.NestedSDFG):
            for inner, _ in node.sdfg.all_nodes_recursive():
                if is_block_reduce(inner):
                    return True
    return False


def map_is_tree_reduction(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """True iff this device map's reduction is lowered as an in-block tree-reduce whose depth
    is this map's thread-block size -- either a WCR accumulator write out of the map exit (where the
    active CUDA codegen tree-reduces, it emits the inline warp/block reduce) or a ``Reduce``
    library node in the map's scope (``gpucub::BlockReduce`` sized by the block, see
    :func:`scope_contains_block_reduce`). Such a map benefits from a larger thread block (see
    :data:`TREE_REDUCTION_BLOCK_SIZE`); a plain elementwise / scatter map does not."""
    if cuda_emits_tree_reductions():
        map_exit = state.exit_node(map_entry)
        for edge in state.out_edges(map_exit):
            if edge.data is not None and edge.data.wcr is not None:
                return True
    return scope_contains_block_reduce(state, map_entry)


def owns_the_thread_block(node) -> bool:
    """True iff ``node`` is a map that gives its enclosing kernel a thread-block level of its own.

    Two spellings of one thing. A map already on a thread-block schedule is one. A map tagged
    ``is_warp_tile`` is the other: the tag is a request the device offload could not honour, and
    ``PromoteWarpTiles`` redeems it as ``GPU_ThreadBlock`` during codegen preprocessing. The tag has
    to count here, because the promotion is still ahead of any pipeline that has not run that pass
    itself -- a block size chosen now would meet a thread-block map that did not exist when it was
    chosen, and codegen refuses that pair as a conflict rather than picking one.
    """
    return isinstance(node, nodes.MapEntry) and (node.map.schedule in THREADBLOCK_SCHEDULES or node.map.is_warp_tile)


def scope_contains_threadblock_map(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """True iff ``map_entry``'s scope (including nested SDFGs) contains a map that owns the
    thread-block level (see :func:`owns_the_thread_block`)."""
    for node in state.all_nodes_between(map_entry, required(state.exit_node(map_entry))):
        if owns_the_thread_block(node):
            return True
        if isinstance(node, nodes.NestedSDFG):
            if any(owns_the_thread_block(inner) for inner, _ in node.sdfg.all_nodes_recursive()):
                return True
    return False


@transformation.explicit_cf_compatible
class SelectGPUDeviceBlockSize(ppl.Pass):
    """Assign a domain-matched ``gpu_block_size`` to every eligible ``GPU_Device`` map.

    Eligible = ``GPU_Device`` schedule, no user-set ``gpu_block_size``, and no inner map that owns
    the thread-block level. See :func:`pick_gpu_block_size` for the selection logic.
    """

    CATEGORY: str = "Optimization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # New/rescheduled scopes may introduce device maps that still need a block size.
        return bool(modified & ppl.Modifies.Scopes)

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> dict[str, list[int]] | None:
        """Set ``gpu_block_size`` on eligible device maps. Returns ``{map label: block size}``
        for the maps it assigned, or ``None`` if it assigned none."""
        assigned: dict[str, list[int]] = {}
        for node, state in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.MapEntry):
                continue
            gpu_map = node.map
            if gpu_map.schedule != dtypes.ScheduleType.GPU_Device:
                continue
            if gpu_map.gpu_block_size is not None:
                continue
            if scope_contains_threadblock_map(state, node):
                continue
            # A tree-reduction map wants a deep block regardless of its domain rank -- the
            # block-reduce folds along the flattened block, so more threads = more reduction
            # per block. A plain map falls back to the domain-matched shape.
            if map_is_tree_reduction(state, node):
                block_size = list(TREE_REDUCTION_BLOCK_SIZE)
            else:
                block_size = pick_gpu_block_size(gpu_map, unit_stride_counts(state.sdfg, state, node))
                if block_size is None:
                    continue
            gpu_map.gpu_block_size = block_size
            assigned[gpu_map.label] = block_size
        return assigned or None


def select_gpu_device_block_size(sdfg: SDFG) -> dict[str, list[int]]:
    """Functional entry point: assign domain-matched block sizes in place.

    Returns the ``{map label: block size}`` map of assignments (empty if none)."""
    return SelectGPUDeviceBlockSize().apply_pass(sdfg, {}) or {}
