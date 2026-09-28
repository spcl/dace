# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared helpers for the standard library node expansions."""
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


def broadcast_axes(result: Sequence, rank: int, axis: Optional[int]) -> List:
    """The entries of ``result``, one per result axis, that an operand of rank ``rank`` lines up with.

    :param axis: The result axis the operand lacks (Fortran ``SPREAD``), or ``None`` to right-align the
                 operand's axes against the result's (NumPy). An operand that fits neither way gets a
                 list of a length other than ``rank``.
    """
    if axis is None:
        return list(result[len(result) - rank:]) if rank <= len(result) else []
    return list(result[:axis]) + list(result[axis + 1:])


def broadcast_map_expansion(label: str, parent_sdfg: dace.SDFG, inputs: Dict[str, Tuple[dace.Memlet, Optional[int]]],
                            output: Tuple[str, dace.Memlet], code: str) -> dace.SDFG:
    """Expand an element-wise library node into one map over its output.

    Every operand keeps its own layout and is broadcast by the NumPy rule: an axis of extent 1 is read at
    index 0, any other axis at the result iterator it lines up with (see :func:`broadcast_axes`). The
    tasklet connector of a node connector ``c`` is ``c_v``.

    :param inputs: Node input connector -> (its memlet, the ``axis`` of :func:`broadcast_axes`).
    :param output: The node output connector and its memlet.
    :param code: The tasklet code.
    :returns: The nested SDFG.
    """
    out_conn, out_memlet = output
    params = [f'__i{d}' for d in range(out_memlet.subset.dims())]
    sdfg = dace.SDFG(f'{label}_sdfg')

    def operand(conn: str, memlet: dace.Memlet, iterators: List[str]) -> dace.Memlet:
        desc = parent_sdfg.arrays[memlet.data]
        shape = memlet.subset.size()
        if len(shape) != len(iterators):
            raise ValueError(f'{label}: {conn} has rank {len(shape)} and cannot broadcast to rank {len(params)}')
        strides = [stride * step for stride, (_, _, step) in zip(desc.strides, memlet.subset)]
        sdfg.add_array(conn, shape, desc.dtype, desc.storage, strides=strides)
        return dace.Memlet(f"{conn}[{', '.join('0' if n == 1 else i for n, i in zip(shape, iterators))}]")

    tasklet_inputs = {
        f'{conn}_v': operand(conn, memlet, broadcast_axes(params, memlet.subset.dims(), axis))
        for conn, (memlet, axis) in inputs.items()
    }
    sdfg.add_state().add_mapped_tasklet(f'{label}_tasklet', {
        p: f'0:{n}'
        for p, n in zip(params, out_memlet.subset.size())
    },
                                        tasklet_inputs,
                                        code, {f'{out_conn}_v': operand(out_conn, out_memlet, params)},
                                        external_edges=True)
    return sdfg
