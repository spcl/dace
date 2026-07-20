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

Callback provenance: every fallback reason carries a stable kebab-case
``[category]`` prefix (set at the raise site via
``UnsupportedFeatureError(category=...)`` or at the fallback call site), so
discrepancy checks and gap reports can aggregate interpreter fallbacks by
cause. The taxonomy in use:

- ``detected-callback`` — the callee was already wrapped as a Python callback
  by preprocessing (intended interpreter work, mirrored in the classic
  frontend's ``callback_mapping``),
- ``unknown-call:<qualname>`` — a call with no lowering (the
  missing-replacements worklist),
- ``opaque-syntax:<stmt-type>`` — statement outside the CPA subset (marked
  during canonicalization),
- ``pyobject-propagation`` — a consumed operand/callee is an opaque Python
  object produced by an earlier callback,
- ``inline-fallback:<subreason>`` — a nested ``@dace.program`` call that
  could not be inlined,
- ``memlet-parse`` / ``indirect-memlet`` — explicit-tasklet memlet gaps,
- ``data-dependent-subscript``, ``dynamic-bound``, ``broadcast``,
  ``explicit-map``, ``explicit-consume``, ``join-merge``, ``loop-stability``,
  ``type-inference``, ``undefined-name``, ``static-sequence``, ``ufunc``,
  ``array-creation``, ``reduction``, ``reference-set``, ``structure-member``,
  ``assign-target`` — per-feature semantic gaps,
- ``safety-net`` — an uncategorized error reaching the totality net in
  ``registry.lower_statement`` (highest bug suspicion),
- ``uncategorized`` — a fallback site with no assigned category yet.
"""
import ast
import copy
from typing import List, Optional, Tuple, Union

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


def _replacement_registered(name: str) -> bool:
    """
    Whether a call is eligible for deferred replacement expansion
    (``tree_to_sdfg.visit_ReplacementCallNode``): it must have BOTH a
    registered replacement (the expansion body) and a descriptor inference
    entry (the frontend must type the result to allocate the target
    container). Replacements needing ``ProgramVisitor`` machinery beyond the
    expansion shim's surface fail loudly at expansion time.
    """
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    if oprepo.Replacements.get(name) is None:
        return False
    return oprepo.Replacements.get_descriptor_inference(name) is not None


def lower_computation(target: DataAccess, value: ast.expr, statement: ast.stmt, state: LoweringState) -> None:
    """
    Lower a canonical flat expression into a target access, dispatching on
    operand types.
    """
    try:
        value = static_values.fold_static_subscripts(value, state)
        if _consumes_pyobject(value, state):
            fallback_to_callback(statement,
                                 state,
                                 'operates on an opaque Python object',
                                 category='pyobject-propagation')
            return
        rewritten = static_values.materialize_operands(value, state)
        elementwise.emit_computation(target, rewritten, statement, state)
    except UnsupportedFeatureError as reason:
        fallback_to_callback(statement, state, reason)


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
        fallback_to_callback(statement, state, reason)
        return
    category = _call_gap_category(call, qualname, state)
    if category == 'detected-callback':
        message = f'call to Python callback "{qualname}"'
    else:
        message = f'no lowering for call "{qualname}"'
    fallback_to_callback(statement, state, message, category=category)


def _call_gap_category(call: ast.Call, qualname: str, state: LoweringState) -> str:
    """
    The provenance category of a call with no lowering: calls whose callee
    preprocessing already wrapped as a Python callback are *intended*
    interpreter work (``detected-callback``, mirrored in the classic
    frontend's ``callback_mapping``); callees that are themselves opaque
    Python objects are downstream of an earlier gap
    (``pyobject-propagation``); everything else is a genuine missing-lowering
    gap, recorded with its qualified name (``unknown-call:<qualname>``).
    """
    callback_mapping = state.emitter.root.callback_mapping
    # Preprocessing rewrites detected-callable call sites to the callback
    # name and records it on the Call node's qualname (or leaves the name as
    # the callee for coroutine/decorated callables).
    detected_names = {astutils.rname(call.func), getattr(call.func, 'qualname', None), getattr(call, 'qualname', None)}
    if detected_names & set(callback_mapping):
        return 'detected-callback'
    if isinstance(call.func, ast.Name):
        binding = state.context.resolve(call.func.id)
        if binding is not None and binding.kind == 'container':
            descriptor = state.context.containers[binding.container]
            if isinstance(descriptor.dtype, dtypes.pyobject):
                return 'pyobject-propagation'
    return f'unknown-call:{qualname}'


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
        if any(keyword.arg not in ('dtype', 'fill_value', 'shape', 'storage', 'lifetime', 'buffer_size')
               for keyword in call.keywords):
            return False
        target_access = _call_target_access(target, inferred, statement, state)
        creation.lower_creation(creation_name, target_access, call, statement, state)
        return True

    # Reductions (full or per-axis with a compile-time scalar axis). Forms the
    # WCR-map mechanism does not cover (e.g. tuple axes) fall through to the
    # deferred replacement expansion below.
    matched = _match_reduction(call, qualname)
    if matched is not None:
        reduction_ufunc, source_expr, axis_expr = matched
        axis: Optional[int] = None
        supported = True
        if axis_expr is not None and not (isinstance(axis_expr, ast.Constant) and axis_expr.value is None):
            axis = state.inference.constant_int(axis_expr)
            supported = axis is not None
        source = resolve_access(source_expr, state) if supported else None
        if supported and source is not None:
            target_access = _call_target_access(target, inferred, statement, state)
            reduction.emit_reduction(target_access, reduction_ufunc, source, statement, state, axis=axis)
            return True

    # Deferred replacement expansion: the descriptor inference typed the call,
    # so vetted registry functions emit a ReplacementCallNode that
    # tree_to_sdfg expands through the classic replacement implementation.
    if _lower_replacement_call(target, call, qualname, inferred, statement, state):
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


def _lower_replacement_call(target: ast.expr, call: ast.Call, qualname: str, inferred, statement: ast.stmt,
                            state: LoweringState) -> bool:
    """
    Emit a deferred :class:`~dace.sdfg.analysis.schedule_tree.treenodes.ReplacementCallNode`
    for a vetted registry replacement. Returns False when the call is not
    vetted or an argument cannot be resolved (the caller falls back).
    """
    name = qualname
    if not _replacement_registered(name):
        name = getattr(call.func, 'qualname', None) or astutils.rname(call.func)
        if not _replacement_registered(name):
            return False
    if state.emitter.in_dataflow_scope:
        return False  # Expansion adds state machinery; no CFG inside dataflow scopes

    converted = _replacement_arguments(call, state)
    if converted is None:
        return False
    arguments, keywords, data_arguments = converted
    if not _expansion_viable(name, arguments, keywords, data_arguments, state):
        return False
    target_access = _call_target_access(target, inferred, statement, state)
    state.emitter.emit(
        tn.ReplacementCallNode(qualname=name,
                               target=target_access.container,
                               arguments=arguments,
                               keyword_arguments=keywords,
                               data_arguments=data_arguments))
    return True


def _expansion_viable(name: str, arguments: List, keywords: dict, data_arguments: set, state: LoweringState) -> bool:
    """
    Trial-run a replacement on a scratch SDFG to decide, at tree-build time
    (where the graceful fallback is a callback), whether deferred expansion
    will succeed at tree-to-SDFG time (where failure would be a hard error).
    Runs the exact code path of ``tree_to_sdfg.visit_ReplacementCallNode``:
    non-viable outcomes are exceptions, recorded view bindings, and
    unsupported return forms.
    """
    from dace.frontend.common import op_repository as oprepo
    from dace.sdfg.analysis.schedule_tree.tree_to_sdfg import ReplacementVisitorShim
    from dace.sdfg.sdfg import SDFG

    function = oprepo.Replacements.get(name)
    scratch = SDFG('__replacement_viability')
    for data_name in data_arguments:
        descriptor = copy.deepcopy(state.context.containers[data_name])
        descriptor.transient = False
        scratch.add_datadesc(data_name, descriptor)
    scratch_state = scratch.add_state()
    shim = ReplacementVisitorShim(scratch, scratch_state, '__viability_target')
    try:
        result = function(shim, scratch, scratch_state, *copy.deepcopy(arguments), **copy.deepcopy(keywords))
    except Exception:
        return False
    if isinstance(result, tuple) and len(result) == 2 and type(result[0]).__name__ == 'NestedCall':
        result = result[1]
    if shim.views:
        return False  # View bindings are frontend state; expansion cannot defer them
    if isinstance(result, str):
        return result in scratch.arrays
    return result is None or result == []


def _replacement_arguments(call: ast.Call, state: LoweringState) -> Optional[Tuple[List, dict, set]]:
    """
    Resolve call arguments to the replacement invocation convention: data
    operands pass as repository container names, everything else as
    compile-time Python values.

    :return: (positional arguments, keyword arguments, data argument names),
             or None if any argument cannot be represented.
    """
    data_arguments = set()

    def convert(expression: ast.expr):
        if isinstance(expression, ast.Name):
            binding = state.context.resolve(expression.id)
            if binding is not None and binding.kind == 'container':
                # Compile-time-valued temps (symbolic aliases) pass by value
                symbolic_value = state.context.symbolic_scalar_values.get(binding.container)
                if symbolic_value is not None:
                    return True, symbolic_value
                descriptor = state.context.containers[binding.container]
                if isinstance(descriptor.dtype, dtypes.pyobject):
                    return False, None
                data_arguments.add(binding.container)
                return True, binding.container
        try:
            value = state.inference.infer(expression)
        except UnsupportedFeatureError:
            return False, None
        if value.kind in ('constant', 'symbolic'):
            return True, value.value
        if value.kind == 'static':
            try:
                elements = state.inference.sequence_constants(value.value)
            except UnsupportedFeatureError:
                return False, None
            return True, tuple(elements) if value.value.kind == 'tuple' else elements
        return False, None  # Data-valued compound expressions need a name

    arguments = []
    for argument in call.args:
        ok, value = convert(argument)
        if not ok:
            return None
        arguments.append(value)
    keywords = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        ok, value = convert(keyword.value)
        if not ok:
            return None
        keywords[keyword.arg] = value
    return arguments, keywords, data_arguments


def _call_target_access(target: ast.expr, inferred, statement: ast.stmt, state: LoweringState) -> DataAccess:
    """Prepare the write target of a call result (allocating a container for
    fresh names from the registry-inferred descriptor)."""
    # Deferred import: rules.assign imports this module at load time
    from dace.frontend.python.nextgen.lowering.rules.assign import prepare_name_target
    if isinstance(target, ast.Name):
        return prepare_name_target(target, inferred, state, statement)
    access = resolve_access(target, state)
    if access is None:
        raise UnsupportedFeatureError('Unsupported call assignment target',
                                      state.context.filename,
                                      statement,
                                      category='assign-target')
    return access


def fallback_to_callback(statement: ast.stmt,
                         state: LoweringState,
                         reason: Union[str, Exception],
                         category: Optional[str] = None) -> None:
    """
    Wrap a statement in a fully specified Python callback.

    :param reason: Why the statement runs in the interpreter — either a plain
        string or the :class:`UnsupportedFeatureError` that triggered the
        fallback.
    :param category: Stable kebab-case gap category for callback provenance,
        rendered as a ``[category]`` prefix on the callback reason. A category
        carried by the ``reason`` exception (set at the raise site) takes
        precedence; without either, the reason is ``[uncategorized]``.
    """
    from dace.frontend.python.nextgen.lowering.rules.callbacks import lower_opaque
    resolved = getattr(reason, 'category', None) or category or 'uncategorized'
    reason_text = f'[{resolved}] {reason}'
    if isinstance(statement, ast.Return):
        _fallback_return(statement, state, reason_text)
        return
    reads, writes = statement_io_sets(statement)
    lower_opaque(OpaqueStmt(statement, reason_text, reads, writes), state)


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
