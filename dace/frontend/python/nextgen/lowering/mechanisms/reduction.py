# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Reduction mechanism: lowers full-array and per-axis reductions
(``numpy.sum``, ``min``, ``max``, ``prod`` and the equivalent array methods)
into frontend-legal nodes: an initialization step followed by a map whose
output memlet carries a write-conflict resolution (WCR) function from the
shared ufunc table.
"""
import ast
import copy
from typing import Dict, List, Optional, Tuple

from dace import data, subsets, symbolic
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
        raise UnsupportedFeatureError(f'No reduction form for ufunc "{ufunc_name}"',
                                      state.context.filename,
                                      statement,
                                      category='reduction')

    if target.container == source.container:
        _emit_aliased_reduction(target, ufunc_name, source, statement, state, axis)
        return

    if axis is not None:
        rank = len(source.subset.ranges)
        normalized_axis = axis + rank if axis < 0 else axis
        if not 0 <= normalized_axis < rank:
            raise UnsupportedFeatureError(f'Reduction axis {axis} is out of range for rank {rank}',
                                          state.context.filename,
                                          statement,
                                          category='reduction')
        # A 1-D source reduced over its only axis is the full reduction
        if rank > 1:
            _emit_axis_reduction(target, specification, ufunc_name, source, normalized_axis, statement, state)
            return

    if not target.is_scalar_access:
        # ``out[i:i + 5] = numpy.sum(A)``: a full reduction produces ONE value,
        # and assigning it to an array subset broadcasts it (NumPy's own rule).
        # Reduce into a scalar temporary and fill the target from it.
        _emit_broadcast_reduction(target, ufunc_name, source, statement, state)
        return

    line = getattr(statement, 'lineno', 0)
    # The operand's NumPy shape, not its squeezed one: :func:`indexed_subset`
    # right-aligns the map parameters against exactly that, so squeezing here
    # dropped a LEADING dimension instead of the size-1 one it meant to
    # (``numpy.sum(a)`` over a ``(20, 1)`` array aligned ``__i0`` with the
    # size-1 axis and rejected the rank it was left with).
    source_shape = list(source.numpy_shape)
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
                           out_memlets={'__out': Memlet(data=target.container, subset=copy.deepcopy(target.subset))}))
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
    out_memlets = {
        '__out': Memlet(data=target.container, subset=copy.deepcopy(target.subset), wcr=specification['reduce'])
    }
    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))


def _emit_aliased_reduction(target: DataAccess, ufunc_name: str, source: DataAccess, statement: ast.stmt,
                            state: LoweringState, axis: Optional[int]) -> None:
    """
    Reduce into a temporary of the target's own shape, then copy it into the
    target, for a reduction that writes into the container it reads
    (``tmp[i] = numpy.sum(tmp)``).

    The direct form below cannot express this. It initializes the target with
    the identity and then folds every source element into it through a WCR --
    so the target's own previous value is destroyed before it is read, and the
    map's reads race with its accumulation. NumPy evaluates the whole reduction
    against the operand as it stood, and staging the result restores that: the
    temporary is written only after every read of the source has happened.
    """
    line = getattr(statement, 'lineno', 0)
    descriptor = (data.Scalar(target.descriptor.dtype) if target.is_scalar_access else data.Array(
        target.descriptor.dtype, list(nondegenerate_shape(target.subset))))
    container = state.context.add_container(f'__reduce_alias_{line}', descriptor)
    staged = DataAccess(container, subsets.Range.from_array(descriptor), descriptor)
    emit_reduction(staged, ufunc_name, source, statement, state, axis=axis)
    state.emitter.emit(
        tn.CopyNode(target=target.container,
                    memlet=Memlet(data=container,
                                  subset=subsets.Range.from_array(descriptor),
                                  other_subset=copy.deepcopy(target.subset))))


def _emit_broadcast_reduction(target: DataAccess, ufunc_name: str, source: DataAccess, statement: ast.stmt,
                              state: LoweringState) -> None:
    """Reduce into a scalar temporary, then broadcast it over ``target``'s
    subset (see the call site for why this is what NumPy does)."""
    line = getattr(statement, 'lineno', 0)
    scalar_descriptor = data.Scalar(target.descriptor.dtype)
    container = state.context.add_container(f'__reduce_{line}', scalar_descriptor)
    scalar_target = DataAccess(container, subsets.Range.from_array(scalar_descriptor), scalar_descriptor)
    emit_reduction(scalar_target, ufunc_name, source, statement, state)

    target_shape = nondegenerate_shape(target.subset)
    if not target_shape:
        _emit_scalar_tasklet(f'reduce_broadcast_{line}', '__in0', scalar_target, target, state)
        return
    params = [f'__i{i}' for i in range(len(target_shape))]
    map_node = nodes.MapEntry(
        nodes.Map(f'reduce_broadcast_{line}', params, subsets.Range([(0, size - 1, 1) for size in target_shape])))
    tasklet = nodes.Tasklet(f'reduce_broadcast_{line}', {'__in0'}, {'__out'}, '__out = __in0')
    in_memlets = {'__in0': Memlet(data=container, subset=subsets.Range.from_array(scalar_descriptor))}
    out_memlets = {'__out': Memlet(data=target.container, subset=indexed_subset(target, params, target_shape))}
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
    placement = _target_placement(target, source_sizes, kept_dims)
    if placement is None:
        raise UnsupportedFeatureError('Per-axis reduction target rank does not match the reduced source',
                                      state.context.filename,
                                      statement,
                                      category='reduction')

    line = getattr(statement, 'lineno', 0)

    def _source_index(dim: int, param: Optional[str]) -> Tuple:
        start, _, step = source.subset.ranges[dim]
        if param is None:
            return (start, start, 1)
        index = start + symbolic.pystr_to_symbolic(param) * step
        return (index, index, 1)

    def _target_subset() -> subsets.Range:
        ranges = []
        for target_dim, (start, _, step) in enumerate(target.subset.ranges):
            source_dim = placement.get(target_dim)
            if source_dim is None:
                # A size-1 target dimension the reduced source has no axis for
                # (a keepdims target): it holds exactly one element, pinned.
                ranges.append((start, start, 1))
                continue
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


def _target_placement(target: DataAccess, source_sizes: List, kept_dims: List[int]) -> Optional[Dict[int, int]]:
    """
    Map each target dimension of a per-axis reduction onto the source
    dimension that indexes it, or None when the two cannot be lined up.

    The target need not have the reduced source's rank. NumPy assignment
    squeezes size-1 dimensions on both sides, and the ``keepdims`` form of a
    reduction relies on exactly that: ``reduced[:] = numpy.sum(data, axis=1)``
    with ``data`` of shape ``(M, N)`` and ``reduced`` of shape ``(M, 1)`` is
    how ONNX's ``ReduceSum`` expansion writes its result
    (``dace/libraries/onnx/op_implementations/reduction_ops.py``), and the
    classic frontend accepts it. So the alignment is between the
    NON-degenerate dimensions of each side, in order; every other target
    dimension holds a single element and is pinned by the caller.

    :return: A dict of {target dimension: source dimension}, or None if the
             sides carry a different number of non-degenerate dimensions
             (a genuine broadcast, which this mechanism cannot express --
             the extents would have to iterate the target rather than the
             source).
    """
    target_sizes = target.subset.size()
    if len(target_sizes) == len(kept_dims):
        # Same rank: the kept dimensions correspond one-to-one, in order.
        return dict(enumerate(kept_dims))
    kept_extents = [dim for dim in kept_dims if source_sizes[dim] != 1]
    target_extents = [dim for dim, size in enumerate(target_sizes) if size != 1]
    if len(kept_extents) != len(target_extents):
        return None
    return dict(zip(target_extents, kept_extents))


def _emit_scalar_tasklet(label: str, connector: str, source: DataAccess, target: DataAccess,
                         state: LoweringState) -> None:
    tasklet = nodes.Tasklet(label, {connector}, {'__out'}, f'__out = {connector}')
    state.emitter.emit(
        tn.TaskletNode(node=tasklet,
                       in_memlets={connector: Memlet(data=source.container, subset=source.subset)},
                       out_memlets={'__out': Memlet(data=target.container, subset=copy.deepcopy(target.subset))}))
