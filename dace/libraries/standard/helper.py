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
GPU_RESIDENT_STORAGES = frozenset(
    {
        dtypes.StorageType.GPU_Global,
        dtypes.StorageType.GPU_Shared,
    }
)
CPU_RESIDENT_STORAGES = frozenset(
    {
        dtypes.StorageType.CPU_Heap,
        dtypes.StorageType.CPU_Pinned,
        dtypes.StorageType.CPU_ThreadLocal,
    }
)


def collapse_shape_and_strides(
    subset: dace.subsets.Range, strides: List[dace.symbolic.SymExpr]
) -> Tuple[List[dace.symbolic.SymExpr], List[dace.symbolic.SymExpr]]:
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
    return is_in_scope(
        parent_state.sdfg, parent_state, node, [dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.Default]
    )


def auto_dispatch(
    node: nodes.LibraryNode,
    parent_state: dace.SDFGState,
    select_fn: Callable[[nodes.LibraryNode, dace.SDFGState], str],
    library_cls: type,
):
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
