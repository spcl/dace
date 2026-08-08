# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Domain-matched thread-block size selection for GPU device maps.

A ``GPU_Device`` map with no explicit ``gpu_block_size`` and no inner
``GPU_ThreadBlock`` map falls back at codegen to the single configured default
(``compiler.cuda.default_block_size``). That default is a 1-D block
(``128,1,1``), which is right for a 1-D kernel but wastes the second grid
dimension of a 2-D kernel: a 2-D iteration domain then launches one thread per
row of the block only, so consecutive threads stride across the outer domain
dimension instead of the contiguous one.

This pass picks a **2-D thread-block whose shape matches the iteration domain**
for 2-D device maps, and leaves 1-D maps on the 1-D default. ``threadIdx.x``
maps to the last (contiguous) map dimension -- codegen reverses the map range
into grid order -- so ``gpu_block_size`` is stored in CUDA ``(x, y, z)`` order,
i.e. reversed map-parameter order. Sizes stay in the 256-512 thread band that
keeps occupancy high without exceeding the per-block thread limit:

    * 1-D domain                 -> ``[128, 1, 1]``
    * ~square 2-D domain         -> ``[16, 16, 1]``  (256 threads)
    * moderately skewed 2-D      -> ``[32, 16, 1]`` / ``[16, 32, 1]``  (512)

The heuristic assumes every symbolic extent is large (>> 64) and roughly equal,
so a symbolic 2-D domain is treated as square. Skew is only acted on when both
extents are known integer constants and one is at least twice the other; the
wider block dimension (32) is then assigned to the wider domain dimension.

The pass never overrides a user-set ``gpu_block_size`` and skips any device map
that already carries an inner ``GPU_ThreadBlock`` / ``GPU_ThreadBlock_Dynamic``
map -- those derive the block size from the thread-block map, and a preset
``gpu_block_size`` would conflict at codegen.
"""
from typing import Any, Dict, List, Optional

from dace import SDFG, dtypes
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation

#: Schedules whose maps derive the block size themselves; a device map wrapping
#: one of these must not also carry a preset ``gpu_block_size``.
THREADBLOCK_SCHEDULES = (dtypes.ScheduleType.GPU_ThreadBlock, dtypes.ScheduleType.GPU_ThreadBlock_Dynamic)

#: Default 1-D thread-block (matches ``compiler.cuda.default_block_size``).
DEFAULT_1D_BLOCK_SIZE = [128, 1, 1]

#: A device map whose reduction is lowered as a block tree-reduce (``compiler.emit_tree_reductions``
#: on + a WCR map output) wants a DEEP block: more lanes per block-reduce means more of the
#: reduction is folded inside the block (one shared-memory tree) and fewer partial results
#: race through the cross-block atomic. 512 (vs the 128/256 a plain elementwise map takes)
#: keeps well under the 1024 thread/block limit while roughly halving the atomic traffic.
TREE_REDUCTION_BLOCK_SIZE = [512, 1, 1]

#: Per-dimension block extents for a 2-D device map (CUDA ``x, y`` order).
SQUARE_2D_BLOCK_EXTENT = 16
WIDE_2D_BLOCK_EXTENT = 32

#: A domain dimension at least this many times larger than the other is "skewed".
SKEW_RATIO = 2


def constant_extent(extent) -> Optional[int]:
    """Return ``extent`` as a positive Python ``int`` if it is a known constant, else ``None``.

    A symbolic (non-constant) extent -- the common case for size symbols -- returns
    ``None`` so callers treat the dimension as "large and unknown".
    """
    try:
        value = int(extent)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def domain_matched_2d_block(ext_x, ext_y) -> List[int]:
    """Choose a 2-D thread-block (CUDA ``x, y`` order) for extents ``ext_x`` (contiguous,
    last map dimension) and ``ext_y`` (outer map dimension).

    Square by default (``16x16``); when both extents are known constants and one is at
    least :data:`SKEW_RATIO` times the other, the wider block dimension (``32``) is placed
    on the wider domain dimension (``32x16`` / ``16x32``). Total stays in 256-512 threads.
    """
    cx = constant_extent(ext_x)
    cy = constant_extent(ext_y)
    if cx is not None and cy is not None:
        if cx >= SKEW_RATIO * cy:
            return [WIDE_2D_BLOCK_EXTENT, SQUARE_2D_BLOCK_EXTENT, 1]
        if cy >= SKEW_RATIO * cx:
            return [SQUARE_2D_BLOCK_EXTENT, WIDE_2D_BLOCK_EXTENT, 1]
    return [SQUARE_2D_BLOCK_EXTENT, SQUARE_2D_BLOCK_EXTENT, 1]


def pick_gpu_block_size(gpu_map: nodes.Map) -> Optional[List[int]]:
    """Domain-matched ``gpu_block_size`` for a ``GPU_Device`` ``map``, in CUDA ``(x, y, z)`` order.

    * 1-D map -> :data:`DEFAULT_1D_BLOCK_SIZE`.
    * 2-D map -> :func:`domain_matched_2d_block` of its (reversed) range sizes; the reversal
      puts the last (contiguous) map dimension on ``threadIdx.x`` to match codegen's grid order.
    * 3-D and higher -> ``None`` (keep the configured 1-D default on ``x``).
    """
    ndim = len(gpu_map.params)
    if ndim <= 1:
        return list(DEFAULT_1D_BLOCK_SIZE)
    if ndim == 2:
        # ``gpu_block_size`` is read in CUDA (x, y) order, which is reversed map order:
        # codegen builds ``grid_size = map.range.size()[::-1]`` and zips it with the block.
        ext_x, ext_y = gpu_map.range.size()[::-1]
        return domain_matched_2d_block(ext_x, ext_y)
    return None


def is_block_reduce(node) -> bool:
    """True iff ``node`` is a ``Reduce`` library node whose reduction folds ACROSS the thread
    block (``cub::BlockReduce``, sized by the block) rather than sequentially per thread. A
    ``Sequential`` Reduce is a per-thread loop -- its block size does not deepen any tree."""
    from dace.libraries.standard.nodes.reduce import Reduce
    return isinstance(node, Reduce) and node.schedule != dtypes.ScheduleType.Sequential


def scope_contains_block_reduce(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """True iff ``map_entry``'s scope (incl. nested SDFGs) holds a block-level ``Reduce`` (see
    :func:`is_block_reduce`), which the CUDA-block expansion lowers to a ``cub::BlockReduce<T,
    N>`` whose ``N`` is this map's flattened block size (via :func:`devicelevel_block_size`) --
    so a bigger block deepens the block-reduce directly."""
    for node in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if is_block_reduce(node):
            return True
        if isinstance(node, nodes.NestedSDFG):
            for inner, _ in node.sdfg.all_nodes_recursive():
                if is_block_reduce(inner):
                    return True
    return False


def map_is_tree_reduction(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """True iff this device map's reduction is lowered as an in-block tree-reduce whose depth
    is this map's thread-block size -- either a WCR accumulator write out of the map exit (with
    ``compiler.emit_tree_reductions`` on, codegen emits the inline warp/block reduce) or a ``Reduce``
    library node in the map's scope (``cub::BlockReduce`` sized by the block, see
    :func:`scope_contains_block_reduce`). Such a map benefits from a larger thread block (see
    :data:`TREE_REDUCTION_BLOCK_SIZE`); a plain elementwise / scatter map does not."""
    if Config.get_bool('compiler', 'emit_tree_reductions'):
        map_exit = state.exit_node(map_entry)
        for edge in state.out_edges(map_exit):
            if edge.data is not None and edge.data.wcr is not None:
                return True
    return scope_contains_block_reduce(state, map_entry)


def scope_contains_threadblock_map(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """True iff ``map_entry``'s scope (including nested SDFGs) contains a thread-block map."""
    for node in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if isinstance(node, nodes.MapEntry) and node.map.schedule in THREADBLOCK_SCHEDULES:
            return True
        if isinstance(node, nodes.NestedSDFG):
            for inner, _ in node.sdfg.all_nodes_recursive():
                if isinstance(inner, nodes.MapEntry) and inner.map.schedule in THREADBLOCK_SCHEDULES:
                    return True
    return False


@transformation.explicit_cf_compatible
class SelectGPUDeviceBlockSize(ppl.Pass):
    """Assign a domain-matched ``gpu_block_size`` to every eligible ``GPU_Device`` map.

    Eligible = ``GPU_Device`` schedule, no user-set ``gpu_block_size``, and no inner
    thread-block map. See :func:`pick_gpu_block_size` for the selection logic.
    """

    CATEGORY: str = 'Optimization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # New/rescheduled scopes may introduce device maps that still need a block size.
        return bool(modified & ppl.Modifies.Scopes)

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[Dict[str, List[int]]]:
        """Set ``gpu_block_size`` on eligible device maps. Returns ``{map label: block size}``
        for the maps it assigned, or ``None`` if it assigned none."""
        assigned: Dict[str, List[int]] = {}
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
                block_size = pick_gpu_block_size(gpu_map)
                if block_size is None:
                    continue
            gpu_map.gpu_block_size = block_size
            assigned[gpu_map.label] = block_size
        return assigned or None


def select_gpu_device_block_size(sdfg: SDFG) -> Dict[str, List[int]]:
    """Functional entry point: assign domain-matched block sizes in place.

    Returns the ``{map label: block size}`` map of assignments (empty if none)."""
    return SelectGPUDeviceBlockSize().apply_pass(sdfg, {}) or {}
