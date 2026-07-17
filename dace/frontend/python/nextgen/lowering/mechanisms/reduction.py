# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Reduction mechanism: lowers full-array and per-axis reductions
(``numpy.sum``, ``min``, ``max``, ``prod`` and the equivalent array methods)
into frontend-legal nodes: an initialization step followed by a map whose
output memlet carries a write-conflict resolution (WCR) function from the
shared ufunc table.
"""
import ast
from typing import Optional, Tuple

from dace import subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import DataAccess, indexed_subset, nondegenerate_shape
from dace.frontend.python.nextgen.lowering.registry import LoweringState

#: Reductions with a usable identity element initialize the target with it.
#: min/max have no identity; they initialize with the first element (or the
#: first slice along the reduced axis), which is correct because their WCR is
#: idempotent.
_IDENTITY_UFUNCS = frozenset({'add', 'multiply'})


def emit_reduction(target: DataAccess,
                   ufunc_name: str,
                   source: DataAccess,
                   statement: ast.stmt,
                   state: LoweringState,
                   axis: Optional[int] = None) -> None:
    """
    Emit a reduction of ``source`` into ``target`` using the given ufunc's
    WCR function: a full reduction into a scalar (``axis=None``), or a
    per-axis reduction into an array of one rank less.

    :raises UnsupportedFeatureError: If the reduction form is unsupported
                                     (rank mismatch, unknown ufunc).
    """
    from dace.frontend.python.replacements.ufunc import ufuncs  # Deferred to avoid an import cycle
    specification = ufuncs.get(ufunc_name)
    if specification is None or not specification.get('reduce'):
        raise UnsupportedFeatureError(f'No reduction form for ufunc "{ufunc_name}"', state.context.filename, statement)

    if axis is not None:
        rank = len(source.subset.ranges)
        normalized_axis = axis + rank if axis < 0 else axis
        if not 0 <= normalized_axis < rank:
            raise UnsupportedFeatureError(f'Reduction axis {axis} is out of range for rank {rank}',
                                          state.context.filename, statement)
        # A 1-D source reduced over its only axis is the full reduction
        if rank > 1:
            _emit_axis_reduction(target, specification, ufunc_name, source, normalized_axis, statement, state)
            return

    if not target.is_scalar_access:
        raise UnsupportedFeatureError('Full reductions require a scalar target', state.context.filename, statement)

    line = getattr(statement, 'lineno', 0)
    source_shape = nondegenerate_shape(source.subset)
    if not source_shape:
        # Scalar source: the reduction is the identity copy
        _emit_scalar_tasklet(f'reduce_init_{line}', '__in0', source, target, state)
        return

    # Initialization: identity element, or the first source element for
    # identity-free reductions (min/max), whose WCR is idempotent.
    if ufunc_name in _IDENTITY_UFUNCS:
        initial = specification['initial']
        init_tasklet = nodes.Tasklet(f'reduce_init_{line}', set(), {'__out'}, f'__out = {initial}')
        state.emitter.emit(
            tn.TaskletNode(node=init_tasklet,
                           in_memlets={},
                           out_memlets={'__out': Memlet(data=target.container, subset=target.subset)}))
    else:
        first_element = subsets.Range([(start, start, 1) for start, _, _ in source.subset.ranges])
        first_access = DataAccess(source.container, first_element, source.descriptor)
        _emit_scalar_tasklet(f'reduce_init_{line}', '__in0', first_access, target, state)

    # Reduction map: every element folds into the target through the WCR
    params = [f'__i{i}' for i in range(len(source_shape))]
    map_range = subsets.Range([(0, size - 1, 1) for size in source_shape])
    map_node = nodes.MapEntry(nodes.Map(f'reduce_{line}', params, map_range))
    tasklet = nodes.Tasklet(f'reduce_{line}', {'__in0'}, {'__out'}, '__out = __in0')
    in_memlets = {'__in0': Memlet(data=source.container, subset=indexed_subset(source, params, source_shape))}
    out_memlets = {'__out': Memlet(data=target.container, subset=target.subset, wcr=specification['reduce'])}
    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))


def _emit_axis_reduction(target: DataAccess, specification: dict, ufunc_name: str, source: DataAccess, axis: int,
                         statement: ast.stmt, state: LoweringState) -> None:
    """
    Per-axis reduction: initialize the (rank-1) target over the kept
    dimensions, then fold every source element into it through a full-rank
    WCR map whose output drops the reduced dimension.
    """
    source_sizes = source.subset.size()
    rank = len(source_sizes)
    kept_dims = [dim for dim in range(rank) if dim != axis]
    if len(target.subset.ranges) != len(kept_dims):
        raise UnsupportedFeatureError('Per-axis reduction target rank does not match the reduced source',
                                      state.context.filename, statement)

    line = getattr(statement, 'lineno', 0)

    def _source_index(dim: int, param: Optional[str]) -> Tuple:
        start, _, step = source.subset.ranges[dim]
        if param is None:
            return (start, start, 1)
        index = start + symbolic.pystr_to_symbolic(param) * step
        return (index, index, 1)

    def _target_subset() -> subsets.Range:
        ranges = []
        for target_dim, source_dim in enumerate(kept_dims):
            start, _, step = target.subset.ranges[target_dim]
            index = start + symbolic.pystr_to_symbolic(f'__i{source_dim}') * step
            ranges.append((index, index, 1))
        return subsets.Range(ranges)

    # Initialization map over the kept dimensions
    kept_params = [f'__i{dim}' for dim in kept_dims]
    init_range = subsets.Range([(0, source_sizes[dim] - 1, 1) for dim in kept_dims])
    init_map = nodes.MapEntry(nodes.Map(f'reduce_init_{line}', kept_params, init_range))
    if ufunc_name in _IDENTITY_UFUNCS:
        init_tasklet = nodes.Tasklet(f'reduce_init_{line}', set(), {'__out'}, f'__out = {specification["initial"]}')
        init_inputs = {}
    else:
        # First slice along the reduced axis (idempotent WCR)
        init_tasklet = nodes.Tasklet(f'reduce_init_{line}', {'__in0'}, {'__out'}, '__out = __in0')
        first_slice = subsets.Range([_source_index(dim, f'__i{dim}' if dim != axis else None) for dim in range(rank)])
        init_inputs = {'__in0': Memlet(data=source.container, subset=first_slice)}
    with state.emitter.scope(tn.MapScope(node=init_map, children=[])):
        state.emitter.emit(
            tn.TaskletNode(node=init_tasklet,
                           in_memlets=init_inputs,
                           out_memlets={'__out': Memlet(data=target.container, subset=_target_subset())}))

    # Full-rank WCR map; the output subset drops the reduced dimension
    all_params = [f'__i{dim}' for dim in range(rank)]
    main_range = subsets.Range([(0, size - 1, 1) for size in source_sizes])
    main_map = nodes.MapEntry(nodes.Map(f'reduce_{line}', all_params, main_range))
    main_tasklet = nodes.Tasklet(f'reduce_{line}', {'__in0'}, {'__out'}, '__out = __in0')
    main_inputs = {
        '__in0':
        Memlet(data=source.container, subset=subsets.Range([_source_index(dim, f'__i{dim}') for dim in range(rank)]))
    }
    main_outputs = {'__out': Memlet(data=target.container, subset=_target_subset(), wcr=specification['reduce'])}
    with state.emitter.scope(tn.MapScope(node=main_map, children=[])):
        state.emitter.emit(tn.TaskletNode(node=main_tasklet, in_memlets=main_inputs, out_memlets=main_outputs))


def _emit_scalar_tasklet(label: str, connector: str, source: DataAccess, target: DataAccess,
                         state: LoweringState) -> None:
    tasklet = nodes.Tasklet(label, {connector}, {'__out'}, f'__out = {connector}')
    state.emitter.emit(
        tn.TaskletNode(node=tasklet,
                       in_memlets={connector: Memlet(data=source.container, subset=source.subset)},
                       out_memlets={'__out': Memlet(data=target.container, subset=target.subset)}))
