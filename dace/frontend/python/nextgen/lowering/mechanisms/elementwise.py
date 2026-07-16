# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Elementwise computation mechanism: lowers a canonical flat expression into a
:class:`TaskletNode` (scalar result) or a :class:`MapScope` with a tasklet
(array result), with NumPy-style broadcasting of operands.

This mechanism serves *any* frontend construct that reduces to an elementwise
operation over data operands — Python operators, NumPy ufuncs, and future
registry entries all converge here.
"""
import ast
import re
from typing import List, Tuple

from dace import subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import (DataAccess, indexed_subset, nondegenerate_shape,
                                                          resolve_access, substitute_data_operands)
from dace.frontend.python.nextgen.lowering.registry import LoweringState
from dace.frontend.python.nextgen.semantics.inference import broadcast_shapes


def iteration_shape(target: DataAccess, operands: List[Tuple[str, DataAccess]], statement: ast.stmt,
                    state: LoweringState) -> List:
    """
    Compute the elementwise iteration space for a computation into a target:
    the target's non-degenerate shape, validated against the NumPy-style
    broadcast of all operand shapes.

    The target subset stays authoritative for the map extent (writes never
    exceed the target); the operand broadcast only detects incompatibilities.
    Symbolically unequal dimensions are assumed equal, deferring the mismatch
    to runtime like the stable frontend.

    :raises UnsupportedFeatureError: If the broadcast operand rank exceeds the
                                     target rank (the result cannot fit the
                                     write subset elementwise).
    """
    target_shape = nondegenerate_shape(target.subset)
    operand_shape: Tuple = ()
    for _, access in operands:
        operand_shape = broadcast_shapes(operand_shape, tuple(nondegenerate_shape(access.subset)))
    if len(operand_shape) > len(target_shape):
        raise UnsupportedFeatureError(
            f'Broadcast operand shape {operand_shape} has higher rank than the write target '
            f'shape {tuple(target_shape)}', state.context.filename, statement)
    return target_shape


def emit_computation(target: DataAccess, value: ast.expr, statement: ast.stmt, state: LoweringState) -> None:
    """
    Emit a tasklet (scalar result) or map-with-tasklet (array result) that
    computes a canonical flat expression into the target access.
    """
    code, operands = substitute_data_operands(value, state)
    emit_elementwise(target, code, operands, statement, state)


def emit_ufunc(target: DataAccess, ufunc_name: str, arguments: List[ast.expr], statement: ast.stmt,
               state: LoweringState) -> None:
    """
    Emit an elementwise computation for a NumPy universal function call,
    taking the scalar tasklet code from the shared ufunc table in
    :mod:`dace.frontend.python.replacements.ufunc`.

    :raises UnsupportedFeatureError: If the ufunc is unknown, its tasklet code
                                     is not a single-expression form, or an
                                     argument is neither data nor a
                                     constant/symbolic value.
    """
    from dace.frontend.python.replacements.ufunc import ufuncs  # Deferred to avoid an import cycle
    specification = ufuncs.get(ufunc_name)
    if specification is None:
        raise UnsupportedFeatureError(f'Unknown NumPy ufunc "{ufunc_name}"', state.context.filename, statement)
    if len(specification['outputs']) != 1 or len(arguments) != len(specification['inputs']):
        raise UnsupportedFeatureError(f'Unsupported call form for NumPy ufunc "{ufunc_name}"', state.context.filename,
                                      statement)
    code = specification['code']
    prefix = f'{specification["outputs"][0]} ='
    if '\n' in code or not code.startswith(prefix):
        raise UnsupportedFeatureError(f'NumPy ufunc "{ufunc_name}" has no single-expression tasklet form',
                                      state.context.filename, statement)
    expression = code[len(prefix):].strip()

    operands: List[Tuple[str, DataAccess]] = []
    for connector, argument in zip(specification['inputs'], arguments):
        access = resolve_access(argument, state)
        if access is not None:
            operands.append((connector, access))
            continue
        inferred = state.inference.infer(argument)
        if inferred.kind not in ('constant', 'symbolic'):
            raise UnsupportedFeatureError(f'Unsupported ufunc argument type for "{ufunc_name}"', state.context.filename,
                                          statement)
        expression = re.sub(rf'\b{connector}\b', f'({inferred.value})', expression)
    emit_elementwise(target, expression, operands, statement, state)


def emit_elementwise(target: DataAccess, expression: str, operands: List[Tuple[str, DataAccess]], statement: ast.stmt,
                     state: LoweringState) -> None:
    """
    Emit a tasklet (scalar result) or map-with-tasklet (array result) that
    computes ``expression`` — scalar code over the given (connector, access)
    operands — into the target access. Map parameters ``__i0..__iN`` are in
    scope inside the expression for array results.
    """
    code = expression
    line = getattr(statement, 'lineno', 0)
    result_shape = iteration_shape(target, operands, statement, state)

    if not result_shape:
        # Scalar result: single tasklet
        tasklet = nodes.Tasklet(f'assign_{line}', {connector
                                                   for connector, _ in operands}, {'__out'}, f'__out = {code}')
        in_memlets = {connector: Memlet(data=access.container, subset=access.subset) for connector, access in operands}
        out_memlets = {'__out': Memlet(data=target.container, subset=target.subset)}
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))
        return

    # Array result: elementwise map
    params = [f'__i{i}' for i in range(len(result_shape))]
    map_range = subsets.Range([(0, size - 1, 1) for size in result_shape])
    map_node = nodes.MapEntry(nodes.Map(f'map_{line}', params, map_range))
    tasklet = nodes.Tasklet(f'assign_{line}', {connector for connector, _ in operands}, {'__out'}, f'__out = {code}')

    in_memlets = {}
    for connector, access in operands:
        if access.is_scalar_access:
            in_memlets[connector] = Memlet(data=access.container, subset=access.subset)
        else:
            in_memlets[connector] = Memlet(data=access.container, subset=indexed_subset(access, params, result_shape))
    out_memlets = {'__out': Memlet(data=target.container, subset=target_indexed_subset(target.subset, params))}

    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))


def target_indexed_subset(subset: subsets.Range, params: List[str]) -> subsets.Range:
    """
    Index the non-degenerate dimensions of a write subset with map parameters,
    keeping degenerate dimensions pinned to their start.
    """
    ranges = []
    param_iterator = iter(params)
    for size, (start, _, step) in zip(subset.size(), subset.ranges):
        if size == 1:
            ranges.append((start, start, 1))
        else:
            param = symbolic.pystr_to_symbolic(next(param_iterator))
            index = start + param * step
            ranges.append((index, index, 1))
    return subsets.Range(ranges)
