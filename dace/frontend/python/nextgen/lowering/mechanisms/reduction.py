# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Reduction mechanism: lowers full-array reductions (``numpy.sum``, ``min``,
``max``, ``prod`` and the equivalent array methods) into frontend-legal nodes:
an initialization tasklet followed by a map whose output memlet carries a
write-conflict resolution (WCR) function from the shared ufunc table.

Per-axis reductions (``axis=k``) are not lowered yet and fall back to the
callback path through the dispatch seam.
"""
import ast

from dace import subsets
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import DataAccess, indexed_subset, nondegenerate_shape
from dace.frontend.python.nextgen.lowering.registry import LoweringState

#: Reductions with a usable identity element initialize the target with it.
#: min/max have no identity; they initialize with the first element instead,
#: which is correct because their WCR is idempotent.
_IDENTITY_UFUNCS = frozenset({'add', 'multiply'})


def emit_reduction(target: DataAccess, ufunc_name: str, source: DataAccess, statement: ast.stmt,
                   state: LoweringState) -> None:
    """
    Emit a full reduction of ``source`` into the scalar ``target`` using the
    given ufunc's WCR function.

    :raises UnsupportedFeatureError: If the reduction form is unsupported
                                     (non-scalar target, unknown ufunc).
    """
    from dace.frontend.python.replacements.ufunc import ufuncs  # Deferred to avoid an import cycle
    specification = ufuncs.get(ufunc_name)
    if specification is None or not specification.get('reduce'):
        raise UnsupportedFeatureError(f'No reduction form for ufunc "{ufunc_name}"', state.context.filename, statement)
    if not target.is_scalar_access:
        raise UnsupportedFeatureError('Per-axis reductions are not supported yet', state.context.filename, statement)

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


def _emit_scalar_tasklet(label: str, connector: str, source: DataAccess, target: DataAccess,
                         state: LoweringState) -> None:
    tasklet = nodes.Tasklet(label, {connector}, {'__out'}, f'__out = {connector}')
    state.emitter.emit(
        tn.TaskletNode(node=tasklet,
                       in_memlets={connector: Memlet(data=source.container, subset=source.subset)},
                       out_memlets={'__out': Memlet(data=target.container, subset=target.subset)}))
