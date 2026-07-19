# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rules for canonical ``return`` statements.

Return values are materialized into the conventional non-transient
``__return`` containers (``__return_<index>`` for tuples), followed by an
explicit :class:`ReturnNode` naming them.

Inside an inlined nested ``@dace.program``
(:attr:`ProgramContext.return_prefix` is non-empty), return containers are
prefixed transients instead and their names are recorded in
:attr:`ProgramContext.return_names` for the call rule to bind. The
:class:`ReturnNode` is emitted uniformly; inside a
:class:`FunctionCallScope` it means "exit the inlined callee", and the call
rule strips a trailing one (a tail return falls off the scope end).
"""
import ast
from typing import List

from dace import data, subsets
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import resolve_access
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule


@rule(ast.Return)
def lower_return(statement: ast.Return, state: LoweringState) -> None:
    prefix = state.context.return_prefix
    if statement.value is None:
        state.emitter.emit(tn.ReturnNode())
        return

    values = statement.value.elts if isinstance(statement.value, ast.Tuple) else [statement.value]
    names: List[str] = []
    for index, value in enumerate(values):
        base_name = '__return' if len(values) == 1 else f'__return_{index}'
        names.append(_materialize_return_value(f'{prefix}{base_name}', value, statement, state))
    state.context.return_names.extend(names)
    state.emitter.emit(tn.ReturnNode(values=names))


def _materialize_return_value(return_name: str, value: ast.expr, statement: ast.Return, state: LoweringState) -> str:
    # Compile-time Python sequences materialize as constant containers first
    if isinstance(value, ast.Name):
        sequence = state.context.static_value_of(value.id)
        if sequence is not None:
            from dace.frontend.python.nextgen.lowering.mechanisms import static_values
            access = static_values.materialize(sequence, state)
            value = ast.copy_location(ast.Name(id=access.container, ctx=ast.Load()), value)

    # Top-level returns are non-transient program outputs; inlined-callee
    # returns are internal temporaries.
    transient = bool(state.context.return_prefix)

    access = resolve_access(value, state) if isinstance(value, (ast.Name, ast.Subscript)) else None
    if access is not None:
        shape = [s for s in access.subset.size() if s != 1] or [1]
        if return_name not in state.context.containers:
            descriptor = data.Array(access.descriptor.dtype, shape)
            return_name = state.context.add_container(return_name, descriptor, transient=transient)
        descriptor = state.context.containers[return_name]
        state.emitter.emit(
            tn.CopyNode(target=return_name,
                        memlet=Memlet(data=access.container,
                                      subset=access.subset,
                                      other_subset=subsets.Range.from_array(descriptor))))
        return return_name

    inferred = state.inference.infer(value)
    dtype = inferred.dtype
    if dtype is None:
        raise UnsupportedFeatureError(f'Cannot determine return value type: {astutils.unparse(value)}',
                                      state.context.filename,
                                      statement,
                                      category='type-inference')
    if return_name not in state.context.containers:
        return_name = state.context.add_container(return_name, data.Array(dtype, [1]), transient=transient)
    tasklet = nodes.Tasklet(f'return_{statement.lineno}', set(), {'__out'}, f'__out = {astutils.unparse(value)}')
    state.emitter.emit(
        tn.TaskletNode(node=tasklet,
                       in_memlets={},
                       out_memlets={'__out': Memlet(data=return_name, subset=subsets.Range([(0, 0, 1)]))}))
    return return_name
