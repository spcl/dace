# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
The type-directed dispatch seam between syntax rules and lowering mechanisms.

Syntax rules (one per canonical statement form) delegate computations here;
this module inspects *operand types* and routes to the appropriate mechanism:

- any pyobject operand: the computation must run in the interpreter
  (callback mechanism),
- static Python sequences mixed with data: materialize, then elementwise,
- data/symbolic/constant operands: elementwise map or tasklet.

Semantic feature gaps (:class:`UnsupportedFeatureError`) also fall back to the
callback mechanism, preserving totality: no user program fails to lower merely
because a construct has no dedicated mechanism yet. Future registry entries
(NumPy functions, library nodes, user dunders) plug in here rather than as
separate per-library rule sets.
"""
import ast
from typing import List, Optional, Tuple

from dace import dtypes
from dace.frontend.python import astutils
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.canonical.cpa import OpaqueStmt, statement_io_sets
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import DataAccess, resolve_access
from dace.frontend.python.nextgen.lowering.registry import LoweringState
from dace.frontend.python.nextgen.lowering.mechanisms import creation, elementwise, reduction, static_values

#: Full-reduction calls by registry-qualified name, mapped to their WCR ufunc.
_REDUCTION_CALLS = {
    'numpy.sum': 'add',
    'numpy.prod': 'multiply',
    'numpy.max': 'maximum',
    'numpy.amax': 'maximum',
    'numpy.min': 'minimum',
    'numpy.amin': 'minimum',
}

#: Array-method reductions (``a.sum()``), mapped to their WCR ufunc.
_REDUCTION_METHODS = {
    'sum': 'add',
    'prod': 'multiply',
    'max': 'maximum',
    'min': 'minimum',
}


def lower_computation(target: DataAccess, value: ast.expr, statement: ast.stmt, state: LoweringState) -> None:
    """
    Lower a canonical flat expression into a target access, dispatching on
    operand types.
    """
    try:
        value = static_values.fold_static_subscripts(value, state)
        if _consumes_pyobject(value, state):
            fallback_to_callback(statement, state, 'operates on an opaque Python object')
            return
        rewritten = static_values.materialize_operands(value, state)
        elementwise.emit_computation(target, rewritten, statement, state)
    except UnsupportedFeatureError as reason:
        fallback_to_callback(statement, state, str(reason))


def lower_call(target: Optional[ast.expr], call: ast.Call, statement: ast.stmt, state: LoweringState) -> None:
    """
    Lower a canonical call, routing by callee identity and operand types:
    nested ``@dace.program``/SDFG-convertible callees go to the inlining rule,
    registry-known NumPy calls to their mechanisms (elementwise ufuncs, array
    creation, WCR reductions), and everything else to the callback fallback.

    :param target: The assignment target expression, or None for a bare call.
    """
    from dace.frontend.python.nextgen.lowering.rules import calls  # Deferred: rules import this module
    qualname, callee = state.inference.resolve_callee(call.func)
    if calls.is_sdfg_convertible(callee):
        calls.lower_nested_call(target, call, callee, statement, state)
        return
    try:
        if _lower_registry_call(target, call, qualname, callee, statement, state):
            return
    except UnsupportedFeatureError as reason:
        fallback_to_callback(statement, state, str(reason))
        return
    fallback_to_callback(statement, state, f'no lowering for call "{qualname}"')


def _lower_registry_call(target: Optional[ast.expr], call: ast.Call, qualname: str, callee: object, statement: ast.stmt,
                         state: LoweringState) -> bool:
    """
    Try to lower a call through the descriptor-inference registry and the
    NumPy mechanisms. Returns False when no mechanism applies (the caller
    falls back to a callback).
    """
    import numpy  # Deferred; keep module import light

    inferred = state.inference.infer_call(call)
    if inferred is None or not inferred.is_data or target is None:
        # No registry entry, a multi-output result, or an unused result:
        # the interpreter fallback preserves semantics in all three cases.
        return False

    # NumPy universal functions: elementwise map from the shared ufunc table
    if isinstance(callee, numpy.ufunc):
        if call.keywords:
            return False  # out=/where=/dtype= change semantics; run in the interpreter
        target_access = _call_target_access(target, inferred, statement, state)
        elementwise.emit_ufunc(target_access, callee.__name__, call.args, statement, state)
        return True

    # Array/stream creation. Resolution may yield the callee's real module
    # path (dace.define_stream lives in dace.frontend.python.wrappers), so
    # fall back to the source-level name: the qualname preprocessing attaches
    # to embedded callee constants, or the textual call name.
    creation_name = qualname
    if creation_name not in creation.CREATION_CALLS:
        creation_name = getattr(call.func, 'qualname', None) or astutils.rname(call.func)
    if creation_name in creation.CREATION_CALLS:
        if any(keyword.arg not in ('dtype', 'fill_value', 'shape') for keyword in call.keywords):
            return False
        target_access = _call_target_access(target, inferred, statement, state)
        creation.lower_creation(creation_name, target_access, call, statement, state)
        return True

    # Reductions (full or per-axis with a compile-time axis)
    matched = _match_reduction(call, qualname)
    if matched is not None:
        reduction_ufunc, source_expr, axis_expr = matched
        axis: Optional[int] = None
        if axis_expr is not None and not (isinstance(axis_expr, ast.Constant) and axis_expr.value is None):
            axis = state.inference.constant_int(axis_expr)
            if axis is None:
                return False  # Non-constant axis: run in the interpreter
        source = resolve_access(source_expr, state)
        if source is None:
            return False
        target_access = _call_target_access(target, inferred, statement, state)
        reduction.emit_reduction(target_access, reduction_ufunc, source, statement, state, axis=axis)
        return True

    return False


def _match_reduction(call: ast.Call, qualname: str) -> Optional[Tuple[str, ast.expr, Optional[ast.expr]]]:
    """
    Match a reduction call form (``np.sum(A[, axis])`` or ``A.sum([axis])``,
    ``axis=`` keyword allowed).

    :return: (WCR ufunc name, source expression, axis expression or None), or
             None if the call is not a lowerable reduction form (extra
             arguments like ``out=``/``initial=`` change semantics).
    """
    if qualname in _REDUCTION_CALLS:
        if not call.args or len(call.args) > 2:
            return None
        ufunc_name = _REDUCTION_CALLS[qualname]
        source_expr = call.args[0]
        axis_expr = call.args[1] if len(call.args) == 2 else None
    elif isinstance(call.func, ast.Attribute) and call.func.attr in _REDUCTION_METHODS:
        if len(call.args) > 1:
            return None
        ufunc_name = _REDUCTION_METHODS[call.func.attr]
        source_expr = call.func.value
        axis_expr = call.args[0] if call.args else None
    else:
        return None
    for keyword in call.keywords:
        if keyword.arg != 'axis' or axis_expr is not None:
            return None
        axis_expr = keyword.value
    return ufunc_name, source_expr, axis_expr


def _call_target_access(target: ast.expr, inferred, statement: ast.stmt, state: LoweringState) -> DataAccess:
    """Prepare the write target of a call result (allocating a container for
    fresh names from the registry-inferred descriptor)."""
    # Deferred import: rules.assign imports this module at load time
    from dace.frontend.python.nextgen.lowering.rules.assign import prepare_name_target
    if isinstance(target, ast.Name):
        return prepare_name_target(target, inferred, state, statement)
    access = resolve_access(target, state)
    if access is None:
        raise UnsupportedFeatureError('Unsupported call assignment target', state.context.filename, statement)
    return access


def fallback_to_callback(statement: ast.stmt, state: LoweringState, reason: str) -> None:
    """Wrap a statement in a fully specified Python callback."""
    from dace.frontend.python.nextgen.lowering.rules.callbacks import lower_opaque
    if isinstance(statement, ast.Return):
        _fallback_return(statement, state, reason)
        return
    reads, writes = statement_io_sets(statement)
    lower_opaque(OpaqueStmt(statement, reason, reads, writes), state)


def _fallback_return(statement: ast.Return, state: LoweringState, reason: str) -> None:
    """
    Fall back a ``return`` whose value cannot be lowered: a ``return`` cannot
    execute inside a Python callback, so the value computation runs in the
    interpreter as an assignment to the conventional return container(s),
    followed by a regular :class:`ReturnNode` naming them.
    """
    from dace.frontend.python.nextgen.lowering.rules.callbacks import lower_opaque
    if statement.value is None:
        state.emitter.emit(tn.ReturnNode())
        return

    prefix = state.context.return_prefix
    values = statement.value.elts if isinstance(statement.value, ast.Tuple) else [statement.value]
    names: List[str] = []
    for index, value in enumerate(values):
        base_name = '__return' if len(values) == 1 else f'__return_{index}'
        target_name = f'{prefix}{base_name}'
        assign = ast.copy_location(ast.Assign(targets=[ast.Name(id=target_name, ctx=ast.Store())], value=value),
                                   statement)
        ast.fix_missing_locations(assign)
        reads, writes = statement_io_sets(assign)
        lower_opaque(OpaqueStmt(assign, reason, reads, writes), state)
        names.append(state.context.resolve(target_name).container)
    state.context.return_names.extend(names)
    state.emitter.emit(tn.ReturnNode(values=names))


def _consumes_pyobject(value: ast.expr, state: LoweringState) -> bool:
    """Check whether any operand of a flat expression is an opaque Python object."""
    for node in ast.walk(value):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            binding = state.context.resolve(node.id)
            if binding is not None and binding.kind == 'container':
                descriptor = state.context.containers[binding.container]
                if isinstance(descriptor.dtype, dtypes.pyobject):
                    return True
    return False
