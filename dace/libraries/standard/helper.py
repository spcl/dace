# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared helpers for library node expansions: CopyLibraryNode, FillLibraryNode and the ``'Auto'`` dispatch."""
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
    # ``Range.size_exact()`` already folds the tile in (``tile * ceiling((e + 1 - b) / step)``); dividing
    # it back out is exact and avoids re-deriving a per-dim count formula that could drift from it.
    # Not ``Range.size()``: it uses the over-approximation of bounds such as the end of a partial tile, and the
    # copy would then overrun the subset.
    for (_, _, s), stride, tile, dim_size in zip(subset, strides, subset.tile_sizes, subset.size_exact()):
        length = dim_size / tile
        if length != 1:
            collapsed_shape.append(length)
            collapsed_strides.append(stride * s)
        if tile != 1:
            collapsed_shape.append(tile)
            collapsed_strides.append(stride)
    return collapsed_shape, collapsed_strides


def collapse_to_elements(
        subset: dace.subsets.Range,
        strides: List[dace.symbolic.SymExpr]) -> Tuple[List[dace.symbolic.SymExpr], List[dace.symbolic.SymExpr]]:
    """:func:`collapse_shape_and_strides` for an expansion that writes element by element. A single-element
    subset collapses to zero dimensions; this helper returns one length-1 dimension for it, so the expansion
    has an index to write through."""
    shape, strides = collapse_shape_and_strides(subset, strides)
    return (shape, strides) if shape else ([1], [1])


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


def select_implementation_by_schedule(node: nodes.LibraryNode, parent_state: dace.SDFGState) -> str:
    """The lowering an ``'Auto'`` node takes from its schedule: ``CUDA`` on a GPU schedule, ``pure`` when
    ``Sequential`` or not yet inferred (``Default``: a caller expanding a graph it still transforms), ``CPU``
    otherwise. A node without the picked lowering takes ``pure``, else ``CPU``.

    :param node: the library node being expanded.
    :param parent_state: state containing ``node``.
    :returns: a key of the node's ``implementations``.
    """
    if node.schedule in dtypes.ALL_GPU_SCHEDULES:
        name = 'CUDA'
    elif node.schedule in (dtypes.ScheduleType.Sequential, dtypes.ScheduleType.Default):
        name = 'pure'
    else:
        name = 'CPU'
    implementations = type(node).implementations
    if name in implementations:
        return name
    return 'pure' if 'pure' in implementations else 'CPU'


def schedule_dispatch(auto_cls: type, node: nodes.LibraryNode, parent_state: dace.SDFGState):
    """Expand ``node`` through :func:`select_implementation_by_schedule`, carrying the picked lowering's
    environments onto ``auto_cls`` (the ``'Auto'`` expansion) so a CUDA pick still links its library.

    :param auto_cls: the calling ``'Auto'`` expansion class.
    :param node: the library node being expanded.
    :param parent_state: state containing ``node``.
    :returns: whatever the picked expansion returns.
    """
    picked = type(node).implementations[select_implementation_by_schedule(node, parent_state)]
    auto_cls.environments = list(picked.environments)
    return auto_dispatch(node, parent_state, select_implementation_by_schedule, type(node))


#: An enclosing loop of provably fewer than this many trips pays the fork/join of a library node
#: inside it few enough times to ignore, so it does not count as re-entry.
REENTRY_SHORT_LOOP_TRIPS = 8


def is_short_loop(loop, cache: Optional[Dict[int, bool]] = None) -> bool:
    """Whether ``loop`` provably runs fewer than :data:`REENTRY_SHORT_LOOP_TRIPS` ascending trips.

    :param loop: the :class:`~dace.sdfg.state.LoopRegion` to measure.
    :param cache: an optional ``id(loop) -> verdict`` map a caller reuses across many nodes sharing
                 the same enclosing loops (a CPU-specialization pass visits one transfer at a time,
                 but the loop nest above it repeats). Valid only for as long as the pass that owns
                 it runs without mutating any ``LoopRegion``'s bounds -- neither
                 :class:`~dace.transformation.passes.cpu_specialization.specialize_cpu_transfers.SpecializeCpuTransfers`
                 nor
                 :class:`~dace.transformation.passes.cpu_specialization.sequentialize_unprofitable_parallel_scopes.SequentializeUnprofitableParallelScopes`
                 does; both only set ``schedule``/``implementation`` on library and map nodes.
    :returns: ``True`` only when the trip count is provably short; an unanalyzable, descending or
              symbolic-length loop answers ``False``.
    """
    if cache is not None:
        cached = cache.get(id(loop))
        if cached is not None:
            return cached
    from dace.transformation.passes.analysis import loop_analysis
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None:
        verdict = False
    elif dace.symbolic.ask('positive', dace.symbolic.simplify(stride)) is not True:
        verdict = False
    else:
        trips = dace.symbolic.int_floor(end - start, stride) + 1
        verdict = dace.symbolic.ask('negative', dace.symbolic.simplify(trips - REENTRY_SHORT_LOOP_TRIPS)) is True
    if cache is not None:
        cache[id(loop)] = verdict
    return verdict


def is_reentered_cpu_transfer(node: nodes.LibraryNode,
                              state: dace.SDFGState,
                              loop_cache: Optional[Dict[int, bool]] = None) -> bool:
    """Whether an enclosing parallel map or long loop re-enters ``node``, so its own OpenMP region
    would be re-opened on every entry.

    Same scope walk (:func:`~dace.transformation.helpers.get_parent_map_and_loop_scopes`) as
    :func:`~dace.transformation.auto.auto_optimize.libnode_is_sequential`, but TRIP-COUNT aware: a
    parallel enclosing map is always a hazard (nested parallelism), while an enclosing loop counts
    only when it is not provably short -- pinning every loop-nested transfer sequential throws
    away real parallelism around a handful of trips.

    :param node: the library node to classify.
    :param state: the state containing ``node``.
    :param loop_cache: forwarded to :func:`is_short_loop`; see its docstring for the invalidation
                       contract.
    :returns: ``True`` if an enclosing parallel map or a not-provably-short loop re-enters ``node``.
    """
    from dace.sdfg.state import LoopRegion
    from dace.transformation.helpers import get_parent_map_and_loop_scopes
    for scope in get_parent_map_and_loop_scopes(state.sdfg, node, state):
        if isinstance(scope, nodes.MapEntry):
            if scope.map.schedule != dtypes.ScheduleType.Sequential:
                return True
        elif isinstance(scope, LoopRegion) and not is_short_loop(scope, cache=loop_cache):
            return True
    return False


def cpu_transfer_parallelizes(node: nodes.LibraryNode,
                              state: dace.SDFGState,
                              num_elements: dace.symbolic.SymbolicType,
                              loop_cache: Optional[Dict[int, bool]] = None) -> bool:
    """Whether a CPU transfer of ``num_elements`` at ``node`` keeps its own OpenMP region.

    Both reasons to take it away: provably too small to amortize a fork/join, or re-entered by an
    enclosing parallel map / long loop that pays that fork/join again on every entry. Cheap size
    check first, so the scope walk is skipped for a transfer that is small either way.

    :param node: the library node to classify.
    :param state: the state containing ``node``.
    :param num_elements: total element count of the transfer (constant or symbolic).
    :param loop_cache: forwarded to :func:`is_reentered_cpu_transfer`.
    :returns: ``True`` to keep the parallel element map, ``False`` to sequentialize it.
    """
    return is_parallel_cpu_transfer_size(num_elements) and not is_reentered_cpu_transfer(
        node, state, loop_cache=loop_cache)


def broadcast_indices(shape: Sequence, result: Sequence, axis: Optional[int] = None) -> List[str]:
    """The subscripts, one per axis of an operand of ``shape``, that read it for the result iterators
    ``__i0, __i1, ...`` by the NumPy broadcasting rule: an axis of extent 1 is read at ``0``.

    :param axis: The result axis the operand lacks (Fortran ``SPREAD``), or ``None`` to right-align the
                 operand's axes against the result's (NumPy). An operand of one element is a Fortran scalar
                 whatever its rank, and broadcasts to every shape.
    :raises ValueError: if the operand cannot broadcast to ``result``. Extents that are not provably unequal
                        are taken to match.
    """
    from dace.frontend.python.replacements.utils import broadcast_together  # Avoid import loop

    shape = list(shape)
    if all(extent == 1 for extent in shape):
        return ['0'] * len(shape)
    if axis is not None:
        shape.insert(axis, 1)
        if len(shape) != len(result):
            raise ValueError(f'a spread adds one axis, so rank {len(shape) - 1} cannot become rank {len(result)}')
    try:
        indices = broadcast_together(result, shape, unidirectional=True)[4]
    except IndexError as ex:
        raise ValueError(f'cannot broadcast shape {tuple(shape)} to {tuple(result)}') from ex
    indices = indices.split(', ') if indices else []
    return indices if axis is None else indices[:axis] + indices[axis + 1:]


def broadcast_map_expansion(label: str, parent_sdfg: dace.SDFG, inputs: Dict[str, Tuple[dace.Memlet, Optional[int]]],
                            output: Tuple[str, dace.Memlet], code: str) -> dace.SDFG:
    """Expand an element-wise library node into one map over its output.

    Every operand keeps its own layout and is read by :func:`broadcast_indices`. The tasklet connector of a
    node connector ``c`` is ``c_v``.

    :param inputs: Node input connector -> (its memlet, the ``axis`` of :func:`broadcast_indices`).
    :param output: The node output connector and its memlet.
    :param code: The tasklet code.
    :returns: The nested SDFG.
    """
    out_conn, out_memlet = output
    result = out_memlet.subset.size()
    params = [f'__i{d}' for d in range(len(result))]
    sdfg = dace.SDFG(f'{label}_sdfg')

    def operand(conn: str, memlet: dace.Memlet, indices: List[str]) -> dace.Memlet:
        desc = parent_sdfg.arrays[memlet.data]
        strides = [stride * step for stride, (_, _, step) in zip(desc.strides, memlet.subset)]
        sdfg.add_array(conn, memlet.subset.size(), desc.dtype, desc.storage, strides=strides)
        return dace.Memlet(f"{conn}[{', '.join(indices)}]")

    tasklet_inputs = {
        f'{conn}_v': operand(conn, memlet, broadcast_indices(memlet.subset.size(), result, axis))
        for conn, (memlet, axis) in inputs.items()
    }
    sdfg.add_state().add_mapped_tasklet(f'{label}_tasklet', {
        p: f'0:{n}'
        for p, n in zip(params, result)
    },
                                        tasklet_inputs,
                                        code, {f'{out_conn}_v': operand(out_conn, out_memlet, params)},
                                        external_edges=True)
    return sdfg
