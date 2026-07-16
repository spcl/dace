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
from typing import Optional

from dace import dtypes
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

    # Array creation
    if qualname in creation.CREATION_CALLS:
        if any(keyword.arg not in ('dtype', 'fill_value', 'shape') for keyword in call.keywords):
            return False
        target_access = _call_target_access(target, inferred, statement, state)
        creation.lower_creation(qualname, target_access, call, statement, state)
        return True

    # Full-array reductions (axis=None)
    reduction_ufunc: Optional[str] = None
    source_expr: Optional[ast.expr] = None
    if qualname in _REDUCTION_CALLS:
        # axis=/initial=/out= and positional axis change semantics: interpreter
        if not call.keywords and len(call.args) == 1:
            reduction_ufunc = _REDUCTION_CALLS[qualname]
            source_expr = call.args[0]
    elif isinstance(call.func, ast.Attribute) and call.func.attr in _REDUCTION_METHODS:
        if not call.keywords and not call.args:
            reduction_ufunc = _REDUCTION_METHODS[call.func.attr]
            source_expr = call.func.value
    if reduction_ufunc is not None and source_expr is not None:
        source = resolve_access(source_expr, state)
        if source is None:
            return False
        target_access = _call_target_access(target, inferred, statement, state)
        reduction.emit_reduction(target_access, reduction_ufunc, source, statement, state)
        return True

    return False


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
    reads, writes = statement_io_sets(statement)
    lower_opaque(OpaqueStmt(statement, reason, reads, writes), state)


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
