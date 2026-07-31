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
from typing import List, Optional, Tuple

from dace import dtypes, subsets
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import (DataAccess, indexed_subset, resolve_access,
                                                          substitute_data_operands)
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
    target_shape = target.numpy_shape
    operand_shape: Tuple = ()
    for _, access in operands:
        if access.indirect:
            continue  # Full-array pointer connectors do not participate in broadcasting
        operand_shape = broadcast_shapes(operand_shape, tuple(access.numpy_shape))
    excess = len(operand_shape) - len(target_shape)
    if excess > 0 and any(size != 1 for size in operand_shape[:excess]):
        # Leading SINGLETON dimensions beyond the target rank are fine -- NumPy
        # assigns a (1, 3) value into a (3,) target -- but a real extent has
        # nowhere to go. A dace ``Scalar``-valued result carried as an
        # ``Array(dtype, [1])`` (a callee's scalar return) is the common case
        # of the former.
        raise UnsupportedFeatureError(
            f'Broadcast operand shape {operand_shape} has higher rank than the write '
            f'target shape {tuple(target_shape)}',
            state.context.filename,
            statement,
            category='broadcast')
    return target_shape


def emit_computation(target: DataAccess,
                     value: ast.expr,
                     statement: ast.stmt,
                     state: LoweringState,
                     wcr: Optional[str] = None) -> None:
    """
    Emit a tasklet (scalar result) or map-with-tasklet (array result) that
    computes a canonical flat expression into the target access.

    :param wcr: Conflict-resolution lambda applied to the write, for
                accumulations inside a dataflow scope (see
                ``mechanisms/conflict.py::accumulation_wcr``).
    """
    code, operands = substitute_data_operands(value, state)
    emit_elementwise(target, code, operands, statement, state, wcr=wcr)


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
        raise UnsupportedFeatureError(f'Unknown NumPy ufunc "{ufunc_name}"',
                                      state.context.filename,
                                      statement,
                                      category='ufunc')
    if len(specification['outputs']) != 1 or len(arguments) != len(specification['inputs']):
        raise UnsupportedFeatureError(f'Unsupported call form for NumPy ufunc "{ufunc_name}"',
                                      state.context.filename,
                                      statement,
                                      category='ufunc')
    code = specification['code']
    prefix = f'{specification["outputs"][0]} ='
    if '\n' in code or not code.startswith(prefix):
        raise UnsupportedFeatureError(f'NumPy ufunc "{ufunc_name}" has no single-expression tasklet form',
                                      state.context.filename,
                                      statement,
                                      category='ufunc')
    expression = code[len(prefix):].strip()

    operands: List[Tuple[str, DataAccess]] = []
    for connector, argument in zip(specification['inputs'], arguments):
        access = resolve_access(argument, state)
        if access is not None:
            operands.append((connector, access))
            continue
        inferred = state.inference.infer(argument)
        if inferred.kind not in ('constant', 'symbolic'):
            raise UnsupportedFeatureError(f'Unsupported ufunc argument type for "{ufunc_name}"',
                                          state.context.filename,
                                          statement,
                                          category='ufunc')
        expression = re.sub(rf'\b{connector}\b', f'({inferred.value})', expression)
    emit_elementwise(target, expression, operands, statement, state)


def emit_cast(target: DataAccess, dtype: dtypes.typeclass, argument: ast.expr, statement: ast.stmt,
              state: LoweringState) -> None:
    """
    Emit a datatype conversion (``dace.int64(x)``) as a single-operand
    elementwise computation.

    This mirrors the tasklet the registry converter builds
    (``replacements/array_manipulation.py::_datatype_converter``) — including
    its exception for booleans, whose typeclass string is already the callable
    name — but without the surrounding state machinery, so a cast can also be
    emitted inside a dataflow scope, where deferred replacement expansion
    cannot run.

    :raises UnsupportedFeatureError: If the argument is neither data nor a
                                     compile-time constant/symbolic value.
    """
    name = dtype.to_string()
    function = name if dtype in (dtypes.bool, dtypes.bool_) else f'dace.{name}'
    access = resolve_access(argument, state)
    if access is not None:
        emit_elementwise(target, f'{function}(__inp)', [('__inp', access)], statement, state)
        return
    inferred = state.inference.infer(argument)
    if inferred.kind not in ('constant', 'symbolic'):
        raise UnsupportedFeatureError(f'Unsupported operand for the "{name}" conversion',
                                      state.context.filename,
                                      statement,
                                      category='type-inference')
    emit_elementwise(target, f'{function}({inferred.value})', [], statement, state)


def emit_elementwise(target: DataAccess,
                     expression: str,
                     operands: List[Tuple[str, DataAccess]],
                     statement: ast.stmt,
                     state: LoweringState,
                     wcr: Optional[str] = None) -> None:
    """
    Emit a tasklet (scalar result) or map-with-tasklet (array result) that
    computes ``expression`` — scalar code over the given (connector, access)
    operands — into the target access. Map parameters ``__i0..__iN`` are in
    scope inside the expression for array results.

    :param wcr: Conflict-resolution lambda applied to the write memlet, for
                accumulations inside a dataflow scope.
    """
    code = expression
    line = getattr(statement, 'lineno', 0)
    result_shape = iteration_shape(target, operands, statement, state)

    # A size-1 result dimension is part of the SHAPE but not of the iteration
    # space: it is pinned in every subset, so giving it a map parameter emits a
    # one-iteration map dimension that no write depends on -- which then reads
    # as a race to the write-conflict pass. ``None`` marks such a dimension for
    # ``indexed_subset``, which pins it.
    params: List[Optional[str]] = [f'__i{i}' if size != 1 else None for i, size in enumerate(result_shape)]
    iterating = [(param, size) for param, size in zip(params, result_shape) if param is not None]

    if not iterating:
        # Scalar result: single tasklet
        tasklet = nodes.Tasklet(f'assign_{line}', {connector
                                                   for connector, _ in operands}, {'__out'}, f'__out = {code}')
        in_memlets = {connector: Memlet(data=access.container, subset=access.subset) for connector, access in operands}
        out_memlets = {'__out': Memlet(data=target.container, subset=target.subset, wcr=wcr)}
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))
        return

    # Array result: elementwise map over the dimensions that actually iterate
    map_range = subsets.Range([(0, size - 1, 1) for _, size in iterating])
    map_node = nodes.MapEntry(nodes.Map(f'map_{line}', [param for param, _ in iterating], map_range))
    tasklet = nodes.Tasklet(f'assign_{line}', {connector for connector, _ in operands}, {'__out'}, f'__out = {code}')

    in_memlets = {}
    for connector, access in operands:
        if access.indirect or access.is_scalar_access:
            in_memlets[connector] = Memlet(data=access.container, subset=access.subset)
        else:
            in_memlets[connector] = Memlet(data=access.container, subset=indexed_subset(access, params, result_shape))
    out_memlets = {'__out': Memlet(data=target.container, subset=indexed_subset(target, params, result_shape), wcr=wcr)}

    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))
