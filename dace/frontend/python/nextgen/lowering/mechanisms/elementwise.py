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

from dace import data, dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.common import InvalidProgram
from dace.frontend.python.replacements.utils import sym_type
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
    code = _cast_operands(code, value, operands, state)
    emit_elementwise(target, code, operands, statement, state, wcr=wcr)


def operand_casts(operand_values: List, operator: Optional[str]) -> Optional[List[Optional[str]]]:
    """
    The per-operand casts ``replacements.operators.result_type`` prescribes for
    an operation, or None when it has no rule for this operator.

    A tasklet computes in the types of its CONNECTORS: ``__in0 / __in1`` over
    two ``uint32`` connectors is integer division in the generated C++ however
    the result container is typed, so ``3 / 5`` stores 0 and not 0.6. These
    casts -- the same ones the classic frontend emits -- are what make the
    tasklet compute what NumPy would.

    Per operand rather than "cast everything to the result type": a comparison
    yields ``bool`` yet must still compare its operands in their promoted type.

    :raises InvalidProgram: If the operand types are invalid for the operator
                            (``float & float``).
    """
    from dace.frontend.python.replacements.operators import result_type

    if not operator:
        return None
    try:
        _, casting = result_type(operand_values, operator)
    except InvalidProgram:
        raise
    except Exception:
        return None
    if not isinstance(casting, (list, tuple)) or len(casting) != len(operand_values):
        return None
    return list(casting)


def _cast_operands(code: str, value: ast.expr, operands: List[Tuple[str, DataAccess]], state: LoweringState) -> str:
    """
    Apply :func:`operand_casts` to the tasklet code of a binary operator.

    An operand spelled inline -- a literal, or a symbol the generated code
    declares itself -- is left alone when C++ promotes it on its own, which it
    does exactly when some other operand already carries the operation's type
    and that type is not complex. Both halves matter: ``10 ** -2`` has no such
    operand, so without the casts the generated code raises an int to an int
    and stores 0 rather than 0.01, and ``std::complex`` defines its arithmetic
    only between its own type, so ``complex128 + 32`` does not compile however
    the other operand is typed.
    """
    if not isinstance(value, ast.BinOp):
        return code
    operand_values = state.inference.binary_operand_values(value)
    if operand_values is None:
        return code
    casts = operand_casts(operand_values, type(value.op).__name__)
    if casts is None:
        return code
    rewritten = ast.parse(code, mode='eval').body
    if not isinstance(rewritten, ast.BinOp):
        return code
    operation_dtype = state.inference.binary_operation_dtype(value)
    promotes = (operation_dtype not in (dtypes.complex64, dtypes.complex128)
                and any(_carries_dtype(operand, operation_dtype) for operand in operand_values))
    # Each side as a WHOLE, rather than by connector name: an indirect read
    # substitutes as ``__in0[__in1]``, and casting the names inside it would
    # cast the index rather than the value it selects.
    for operand, side, cast in zip(operand_values, ('left', 'right'), casts):
        if cast is None or (promotes and not isinstance(operand, data.Data)):
            continue
        function = ast.parse(cast, mode='eval').body
        setattr(rewritten, side, ast.Call(func=function, args=[getattr(rewritten, side)], keywords=[]))
    return astutils.unparse(ast.fix_missing_locations(rewritten))


def _carries_dtype(operand, dtype: Optional[dtypes.typeclass]) -> bool:
    """Whether an operand reaches the generated code already typed ``dtype`` --
    a container through its connector, a symbol through its own declaration. A
    literal does not: its C++ type comes from how it is spelled."""
    if dtype is None:
        return False
    if isinstance(operand, data.Data):
        return operand.dtype == dtype
    if symbolic.issymbolic(operand):
        return sym_type(operand) == dtype
    return False


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
    operand_values: List = []
    for connector, argument in zip(specification['inputs'], arguments):
        access = resolve_access(argument, state)
        if access is not None:
            operands.append((connector, access))
            operand_values.append(access.descriptor)
            continue
        inferred = state.inference.infer(argument)
        if inferred.kind not in ('constant', 'symbolic'):
            raise UnsupportedFeatureError(f'Unsupported ufunc argument type for "{ufunc_name}"',
                                          state.context.filename,
                                          statement,
                                          category='ufunc')
        expression = re.sub(rf'\b{connector}\b', f'({inferred.value})', expression)
        operand_values.append(inferred.value)

    # The same hazard :func:`_cast_operands` covers for Python operators:
    # ``numpy.divide`` of two ``uint32`` arrays produces ``float64``, but its
    # table code is ``__in1 / __in2`` and two uint32 connectors divide as
    # integers. Table code names its operands as bare connectors, so a name
    # substitution reaches exactly the operand.
    casts = operand_casts(operand_values, specification.get('operator'))
    if casts is not None:
        data_connectors = dict(operands)
        for connector, cast in zip(specification['inputs'], casts):
            if cast is not None and connector in data_connectors:
                expression = re.sub(rf'\b{connector}\b', f'{cast}({connector})', expression)
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


def _is_thread_local(target: DataAccess, state: LoweringState) -> bool:
    """Whether a write target lives in per-thread storage."""
    descriptor = state.context.containers.get(target.container)
    return getattr(descriptor, 'storage', None) == dtypes.StorageType.CPU_ThreadLocal


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
    map_label = state.context.fresh_map_label(line)
    map_node = nodes.MapEntry(nodes.Map(map_label, [param for param, _ in iterating], map_range))
    if _is_thread_local(target, state):
        # A thread-local destination is a DIFFERENT array in every thread, so a
        # parallel map leaves each one holding only the slice of the result its
        # thread happened to iterate. Whoever reads the array afterwards --
        # ordinarily the master thread, outside any parallel region -- then sees
        # a partly uninitialized array. Writing it in one thread is what makes
        # the whole array well defined.
        map_node.map.schedule = dtypes.ScheduleType.Sequential
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
