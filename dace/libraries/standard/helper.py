# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared helpers for CopyLibraryNode and FillLibraryNode expansions."""
from typing import Callable, List, Tuple

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.sdfg.scope import is_in_scope

# Both legacy and experimental codegens consume this exact name for stream wiring.
CURRENT_STREAM_NAME = "__dace_current_stream"

# Register is intentionally in neither set: resolves by scope (GPU register vs. host stack slot).
GPU_RESIDENT_STORAGES = frozenset({
    dtypes.StorageType.GPU_Global,
    dtypes.StorageType.GPU_Shared,
})
CPU_RESIDENT_STORAGES = frozenset({
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.CPU_Pinned,
    dtypes.StorageType.CPU_ThreadLocal,
})


def host_accessible_info_storage(storage: dtypes.StorageType) -> dtypes.StorageType:
    """
    Return the storage a cuSOLVER/LAPACK status scalar (``devInfo``) should use so that it stays
    host-checkable. When the matrix operand lives in GPU-resident memory, the status is placed in
    pinned host memory (DMA-reachable from the device, so cuSOLVER can write it via the unified
    address space while the host can still read the result). CPU-resident inputs keep their storage.

    Lives here rather than in ``dace.dtypes`` because it keys off ``GPU_RESIDENT_STORAGES``
    (``{GPU_Global, GPU_Shared}``), which is defined above -- note ``dtypes.GPU_STORAGES`` is a
    *different*, narrower set (``{GPU_Shared}``) and is not a substitute.
    """
    if storage in GPU_RESIDENT_STORAGES:
        return dtypes.StorageType.CPU_Pinned
    return storage


def collapse_shape_and_strides(
        subset: dace.subsets.Range,
        strides: List[dace.symbolic.SymExpr]) -> Tuple[List[dace.symbolic.SymExpr], List[dace.symbolic.SymExpr]]:
    """Drop length-1 dims from a (subset, strides) pair; surviving strides scale by the subset step.

    A tiled dimension (``b:e:step:tile``) addresses ``tile`` contiguous elements per step, which no
    single (length, stride) pair expresses -- it expands into two dims: the step count at
    ``stride * step``, then the tile at ``stride``.

    :param subset: The access range, one ``(begin, end, step)`` per dimension.
    :param strides: The parent array strides, aligned with ``subset``.
    :returns: ``(collapsed_shape, collapsed_strides)`` with singletons removed.
    """
    collapsed_shape = []
    collapsed_strides = []
    # ``Range.size()`` already folds the tile in (``tile * ceiling((e + 1 - b) / step)``); dividing
    # it back out is exact and avoids re-deriving a per-dim count formula that could drift from it.
    for (_, _, s), stride, tile, dim_size in zip(subset, strides, subset.tile_sizes, subset.size()):
        length = dim_size / tile
        if length != 1:
            collapsed_shape.append(length)
            collapsed_strides.append(stride * s)
        if tile != 1:
            collapsed_shape.append(tile)
            collapsed_strides.append(stride)
    return collapsed_shape, collapsed_strides


def is_parallel_cpu_transfer_size(num_elements: dace.symbolic.SymbolicType) -> bool:
    """False only when ``num_elements`` is a compile-time constant below
    ``compiler.cpu.parallel_transfer_min_elements``; a symbolic (unknown-at-compile-time) size
    is assumed large and takes the parallel path too.

    :param num_elements: total contiguous element count (constant or symbolic).
    :returns: ``True`` to route to the mapped expansion, ``False`` to keep the single libc call.
    """
    threshold = int(dace.Config.get('compiler', 'cpu', 'parallel_transfer_min_elements'))
    try:
        return int(dace.symbolic.simplify(num_elements)) >= threshold
    except (TypeError, ValueError):
        return True


def is_in_parallel_scope(node: nodes.LibraryNode, parent_state: dace.SDFGState) -> bool:
    """True when a multi-threaded map encloses this transfer, so the mapped form would open one
    OpenMP region per entry instead of one for the whole transfer.

    ``Default`` counts: an unresolved enclosing map becomes ``CPU_Multicore`` at the top level.

    :param node: the transfer library node.
    :param parent_state: state containing ``node``.
    :returns: ``True`` if a parallel map scope encloses the node, at any nesting depth.
    """
    return is_in_scope(parent_state.sdfg, parent_state, node,
                       [dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.Default])


def auto_dispatch(node: nodes.LibraryNode, parent_state: dace.SDFGState,
                  select_fn: Callable[[nodes.LibraryNode, dace.SDFGState], str], library_cls: type):
    """Dispatch a library node's ``'Auto'`` implementation to the one ``select_fn`` picks, setting
    ``node.implementation`` so introspection reflects what was chosen.

    :param node: the library node being expanded.
    :param parent_state: state containing ``node`` (owning SDFG is ``parent_state.sdfg``).
    :param select_fn: callable returning a concrete implementation name (not ``'Auto'``).
    :param library_cls: the library node class with the ``implementations`` dict.
    :returns: whatever the resolved expansion returns.
    """
    impl_name = select_fn(node, parent_state)
    assert impl_name != 'Auto', f"{select_fn.__name__} must not return 'Auto'."
    node.implementation = impl_name
    return library_cls.implementations[impl_name].expansion(node, parent_state, parent_state.sdfg)


# --------------------------------------------------------------------------------------------

#: An enclosing loop of provably fewer than this many trips pays the fork/join of a library node
#: inside it few enough times to ignore, so it does not count as re-entry.
REENTRY_SHORT_LOOP_TRIPS = 8


def is_short_loop(loop) -> bool:
    """Whether ``loop`` provably runs fewer than :data:`REENTRY_SHORT_LOOP_TRIPS` ascending trips.

    :param loop: the :class:`~dace.sdfg.state.LoopRegion` to measure.
    :returns: ``True`` only when the trip count is provably short; an unanalyzable, descending or
              symbolic-length loop answers ``False``.
    """
    from dace.transformation.passes.analysis import loop_analysis
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None:
        return False
    if dace.symbolic.ask('positive', dace.symbolic.simplify(stride)) is not True:
        return False
    trips = dace.symbolic.int_floor(end - start, stride) + 1
    return dace.symbolic.ask('negative', dace.symbolic.simplify(trips - REENTRY_SHORT_LOOP_TRIPS)) is True


def is_reentered_cpu_transfer(node: nodes.LibraryNode, state: dace.SDFGState) -> bool:
    """Whether an enclosing parallel map or long loop re-enters ``node``, so its own OpenMP region
    would be re-opened on every entry.

    Same scope walk (:func:`~dace.transformation.helpers.get_parent_map_and_loop_scopes`) as
    :func:`~dace.transformation.auto.auto_optimize.libnode_is_sequential`, but TRIP-COUNT aware: a
    parallel enclosing map is always a hazard (nested parallelism), while an enclosing loop counts
    only when it is not provably short -- pinning every loop-nested transfer sequential throws
    away real parallelism around a handful of trips.

    :param node: the library node to classify.
    :param state: the state containing ``node``.
    :returns: ``True`` if an enclosing parallel map or a not-provably-short loop re-enters ``node``.
    """
    from dace.sdfg.state import LoopRegion
    from dace.transformation.helpers import get_parent_map_and_loop_scopes
    for scope in get_parent_map_and_loop_scopes(state.sdfg, node, state):
        if isinstance(scope, nodes.MapEntry):
            if scope.map.schedule != dtypes.ScheduleType.Sequential:
                return True
        elif isinstance(scope, LoopRegion) and not is_short_loop(scope):
            return True
    return False


def cpu_transfer_parallelizes(node: nodes.LibraryNode, state: dace.SDFGState,
                              num_elements: dace.symbolic.SymbolicType) -> bool:
    """Whether a CPU transfer of ``num_elements`` at ``node`` keeps its own OpenMP region.

    Both reasons to take it away: provably too small to amortize a fork/join, or re-entered by an
    enclosing parallel map / long loop that pays that fork/join again on every entry. Cheap size
    check first, so the scope walk is skipped for a transfer that is small either way.

    :param node: the library node to classify.
    :param state: the state containing ``node``.
    :param num_elements: total element count of the transfer (constant or symbolic).
    :returns: ``True`` to keep the parallel element map, ``False`` to sequentialize it.
    """
    return is_parallel_cpu_transfer_size(num_elements) and not is_reentered_cpu_transfer(node, state)
