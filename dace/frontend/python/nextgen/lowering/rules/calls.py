# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rules for canonical calls (assignment position and bare statements).

Call routing lives in the type-directed dispatch seam
(:func:`~dace.frontend.python.nextgen.lowering.dispatch.lower_call`): nested
``@dace.program``/SDFG-convertible callees are inlined here, registry-known
NumPy calls go to the mechanism modules, and everything else falls back to the
callback path with full I/O specifications — the same totality guarantee as
any other opaque statement.
"""
import ast
import copy
import inspect
import types
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import data, dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.sdfg import InterstateEdge
from dace.utils import prod
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.common import closure_constant_descriptor, interpreter_callable
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule
from dace.frontend.python.nextgen.semantics.values import StaticSequence


def lower_call_assign(statement: ast.Assign, state: LoweringState) -> None:
    """Lower ``target = f(args...)`` through the call dispatch seam."""
    from dace.frontend.python.nextgen.lowering import dispatch
    dispatch.lower_call(statement.targets[0], statement.value, statement, state)


@rule(ast.Expr)
def lower_expr(statement: ast.Expr, state: LoweringState) -> None:
    """
    Lower a bare call statement ``f(args...)`` (the only canonical bare
    expression form) through the call dispatch seam, with no target.
    """
    from dace.frontend.python.nextgen.lowering import dispatch
    if isinstance(statement.value, ast.Call):
        dispatch.lower_call(None, statement.value, statement, state)
        return
    dispatch.fallback_to_callback(statement, state, 'bare expression statement', category='opaque-syntax:Expr')


def is_sdfg_convertible(callee: Any) -> bool:
    """Whether a resolved callee is a nested dace program or an SDFG-valued
    object. Convertibility is duck-typed on ``__sdfg__``, matching the classic
    frontend (convertible objects need not inherit ``SDFGConvertible``)."""
    if callee is None:
        return False
    from dace.sdfg import SDFG  # Deferred to avoid an import cycle
    return isinstance(callee, SDFG) or hasattr(callee, '__sdfg__')


def lower_nested_call(target: Optional[ast.expr], call: ast.Call, callee: Any, statement: ast.stmt,
                      state: LoweringState) -> None:
    """
    Lower a call to a nested ``@dace.program`` (inlined into a
    :class:`FunctionCallScope` sharing the caller's repository) or an
    SDFG-valued callee (an explicit :class:`SDFGCallNode`).

    The callee is preprocessed and canonicalized recursively; its body is
    lowered under :meth:`ProgramContext.inline_scope`, so parameter names bind
    to the caller's repository containers directly and all callee-allocated
    containers go through the shared uniquifying repository — no post-hoc
    renaming pass. Anything that cannot be inlined soundly (recursion, early
    returns, unsupported arguments, unparseable callees) falls back to the
    callback path before any node is emitted.
    """
    from dace.frontend.python.nextgen.lowering.dispatch import fallback_to_callback
    from dace.frontend.python.parser import DaceProgram  # Deferred to avoid an import cycle
    from dace.sdfg import SDFG

    # A callee declared ``@dace.program(inline=False)`` keeps its own SDFG, the
    # same way any other SDFG-convertible callee does. Its parse goes through
    # ``__sdfg__``, so the callee's declared argument shapes are checked at the
    # boundary rather than being absorbed into the caller's containers.
    if not isinstance(callee, DaceProgram) or not getattr(callee, 'inline', True):
        sdfg = callee if isinstance(callee, SDFG) else _convertible_to_sdfg(callee, call, state)
        if sdfg is None:
            fallback_to_callback(statement,
                                 state,
                                 'SDFG-convertible callee could not produce an SDFG',
                                 category='inline-fallback:no-sdfg')
            return
        _lower_sdfg_call(target, call, sdfg, statement, state, callee)
        return

    if callee.f in state.context.inline_stack:
        fallback_to_callback(statement, state, 'recursive @dace.program call', category='inline-fallback:recursion')
        return

    # Steps 1-3 (argument mapping, preprocessing, canonicalization) emit
    # nothing; any failure here falls back to the interpreter, preserving
    # totality. Failures during emission (step 4) are frontend bugs and raise.
    try:
        (callee_body, parameter_bindings, callee_globals, argument_labels, pending_views,
         pending_symbols) = _prepare_callee(call, callee, state)
    except Exception as reason:  # Unparseable callee, unsupported argument, ...
        fallback_to_callback(statement,
                             state,
                             f'cannot inline call to "{callee.name}": {reason}',
                             category=getattr(reason, 'category', None) or 'inline-fallback:parse-failure')
        return
    unsupported = _unsupported_return_shape(callee_body)
    if unsupported is not None:
        fallback_to_callback(statement,
                             state,
                             f'{unsupported} in nested dace program "{callee.name}"',
                             category='inline-fallback:return-shape')
        return

    # Restructure early returns into tail positions (statements following a
    # returning branch move into the other branch), so exiting the callee
    # coincides with falling off the scope end everywhere.
    callee_body = _normalize_early_returns(callee_body)
    if _has_non_tail_return(callee_body):
        fallback_to_callback(statement,
                             state,
                             f'early return that cannot be restructured in nested dace program "{callee.name}"',
                             category='inline-fallback:early-return')
        return

    # Inlining is now committed: materialize any argument reinterpretation
    # views (see ``_reshape_view_descriptor``) before entering the callee
    # scope, matching the classic frontend's NView placement immediately
    # before the nested call.
    # Symbols specialized from runtime scalars are defined here, before the
    # call scope, so the callee body (and the shapes of anything it allocates)
    # can use them (see ``_map_symbol_keywords``).
    for symbol_name, source_container in pending_symbols:
        # A Scalar generates as a value rather than a pointer, so subscripting
        # it does not compile -- the same array-versus-scalar distinction
        # ``access.scalar_read_expression`` draws, and that the classic frontend
        # draws in ``newast.ProgramVisitor._parse_sdfg_call``.
        source_descriptor = state.context.containers.get(source_container)
        if isinstance(source_descriptor, data.Scalar):
            assignment = source_container
        else:
            assignment = f'{source_container}[0]'
        state.emitter.emit(
            tn.AssignNode(name=symbol_name,
                          value=CodeBlock(assignment),
                          edge=InterstateEdge(assignments={symbol_name: assignment})))
    for view_container, source_container, source_descriptor, view_descriptor in pending_views:
        state.emitter.emit(
            tn.ViewNode(target=view_container,
                        source=source_container,
                        memlet=Memlet(data=source_container, subset=subsets.Range.from_array(source_descriptor)),
                        src_desc=source_descriptor,
                        view_desc=view_descriptor))

    return_prefix = state.context.fresh_name(f'__{callee.name}_ret')
    scope = tn.FunctionCallScope(call=tn.FrontendFunctionCall(callee_name=callee.name, arguments=argument_labels),
                                 children=[])
    with state.context.inline_scope(callee.f, parameter_bindings, callee_globals, return_prefix,
                                    _annotated_return_dtype(callee), callee_body) as return_names:
        with state.emitter.scope(scope):
            state.lower_body(callee_body)
        _strip_tail_returns(scope)  # Tail returns fall off the scope end
        returned = list(dict.fromkeys(return_names))
    _bind_call_results(target, returned, statement, state)


def _annotated_return_dtype(callee: Any) -> Optional[dtypes.typeclass]:
    """
    Element type declared by an inlined callee's ``-> dtype`` return
    annotation, or None when it has none.

    A top-level program's annotation reaches the frontend as a pre-registered
    ``__return`` container (see ``DaceProgram._get_type_annotations``), but an
    inlined callee never has ``argtypes`` built for it -- its parameters bind
    to caller containers at the call site -- so the annotation has to be read
    off the signature here. Without it, ``return 5`` in a callee declared
    ``-> dace.int32`` hands back the literal's own inferred type.

    Tuple annotations describe several return values with potentially
    differing types; those are left to inference.
    """
    signature = getattr(callee, 'signature', None)
    if signature is None or getattr(callee, 'ignore_type_hints', False):
        return None
    annotation = signature.return_annotation
    if annotation is inspect.Signature.empty or annotation is None or isinstance(annotation, tuple):
        return None
    try:
        return data.create_datadescriptor(annotation).dtype
    except (TypeError, ValueError):
        return None


def _prepare_callee(
    call: ast.Call, callee: Any, state: LoweringState
) -> Tuple[List[ast.stmt], Dict[str, str], Dict[str, Any], Dict[str, str], List[Tuple[str, str, data.Data, data.Data]]]:
    """
    Map call arguments to callee parameters, then fetch (or produce) the
    callee's preprocessed and canonicalized parse through the per-program
    parse cache: a callee invoked from multiple call sites with the same
    specialization parses once.

    Data arguments bind parameters to the caller's repository containers (by
    reference); constant, symbolic, and compile-time-sequence arguments
    specialize the callee through its globals. An argument whose descriptor
    shape differs from the callee's own declared parameter annotation (but
    matches in total element count and dtype) binds through a reinterpreting
    view instead (see ``_reshape_view_descriptor``); its container is
    registered but not yet emitted (see ``pending_views``) — emission is
    deferred to the caller, which commits to inlining only after the
    return-shape checks pass.

    :return: A 6-tuple of (canonical callee body — a fresh deep copy, since
             lowering mutates it, parameter-to-container bindings, resolved
             callee globals, argument label mapping, a list of pending
             argument-reinterpretation views as (view container, source
             container, source descriptor, view descriptor) tuples, and a list
             of pending symbol definitions as (symbol name, source container)
             pairs — see ``_map_symbol_keywords``).
    """
    (argtypes, callee_globals, parameter_bindings, argument_labels, injected_defaults, spec_key, pending_views,
     pending_symbols) = _map_arguments(call, callee, state)

    # Cache key: callee identity (function AND bound object — two instances
    # share __call__ source but specialize separately through their attribute
    # values, which enter the parse via preprocessing, invisible to the
    # specialization key) plus the argument specialization.
    key = (id(callee), callee.resolve_functions, id(callee.methodobj), spec_key)
    parse = state.context.parse_cache.get_or_parse(
        key, lambda: _parse_callee(callee, argtypes, callee_globals, injected_defaults))

    # Merge closure metadata into the shared repository/root. Idempotent
    # (setdefault), and the closure object is deliberately shared across call
    # sites: closure-array descriptor identity drives qualified-name
    # deduplication.
    closure = parse.closure
    # Two closures can name their callbacks identically -- a helper called ``b``
    # defined inside each of two factory functions is two different functions
    # under one name. Registering with ``setdefault`` alone kept whichever
    # arrived first and pointed BOTH call sites at it, so one of the callbacks
    # silently ran twice and the other never. Give the newcomer a fresh name and
    # rewrite the callee body that refers to it.
    callback_renames: Dict[str, str] = {}
    for callback_name, (original, function, _) in closure.callbacks.items():
        callable_object = interpreter_callable(function)
        existing = state.context.callback_callables.get(callback_name)
        if existing is not None and existing is not callable_object:
            renamed = data.find_new_name(callback_name, state.context.callback_callables)
            callback_renames[callback_name] = renamed
            callback_name = renamed
        state.emitter.root.callback_mapping.setdefault(callback_name, original)
        state.context.callback_callables.setdefault(callback_name, callable_object)
    for constant_name, value in closure.closure_constants.items():
        descriptor = closure_constant_descriptor(value)
        if descriptor is not None:
            state.context.constants.setdefault(constant_name, (descriptor, value))
            state.context.folded_constants.add(constant_name)
    # External arrays the callee references bind inside its inline scope,
    # deduplicated by qualified name across the whole program
    for reference_name, (qualified_name, descriptor, _, _) in closure.closure_arrays.items():
        container = state.context.register_closure_array(reference_name, qualified_name, descriptor)
        parameter_bindings[reference_name] = container

    # Lowering mutates the body (early-return restructuring, annotation
    # hints), so every call site works on its own copy — a reduced one, which
    # keeps the IDENTITY of the objects preprocessing embedded (resolved dace
    # programs, SDFGs, arbitrary constants): resolution, convertibility checks
    # and the parse-cache key itself are identity-based on them.
    body = astutils.copy_tree(parse.canonical_body)
    if callback_renames:
        body = [_rename_callback_references(statement, callback_renames) for statement in body]
    return body, parameter_bindings, parse.program_globals, argument_labels, pending_views, pending_symbols


def _rename_callback_references(statement: ast.stmt, renames: Dict[str, str]) -> ast.stmt:
    """Point a callee body's callback references at their deduplicated names."""

    class _Renamer(ast.NodeTransformer):

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if isinstance(node.ctx, ast.Load) and node.id in renames:
                node.id = renames[node.id]
            return node

    return ast.fix_missing_locations(_Renamer().visit(statement))


def _declared_parameter_descriptor(callee: Any, parameter: str) -> Optional[data.Data]:
    """
    The callee's own explicit parameter-annotation descriptor, if any. Python
    evaluates type annotations at function-definition time, so
    ``a: dace.float64[4, 5, 10]`` resolves directly to an ``Array`` descriptor
    on ``callee.signature`` — independent of whatever descriptor the call
    site's argument happens to carry.
    """
    parameter_info = callee.signature.parameters.get(parameter)
    if parameter_info is None:
        return None
    annotation = parameter_info.annotation
    return annotation if isinstance(annotation, data.Data) else None


def _reshape_view_descriptor(caller_descriptor: data.Data, declared: Optional[data.Data]) -> Optional[data.Data]:
    """
    A fresh view descriptor reinterpreting ``caller_descriptor`` with the
    callee's own declared parameter shape, when it differs but preserves the
    logical element count (the product of ``shape``, not ``total_size`` —
    the latter is the allocated footprint including stride padding, which a
    strided source view like a nested slice legitimately has larger than its
    element count) and dtype.

    Mirrors the classic frontend's ``NView`` (n-dimensional view) semantics
    for a shape-mismatched nested-call connector — the classic frontend
    builds a genuine nested SDFG, where a shape mismatch on the connector is
    resolved by the SDFG-to-schedule-tree conversion; nextgen inlines calls
    directly (sharing the repository, no nested SDFG), so the equivalent
    reinterpretation is a plain view materialized at the call site.

    :return: None when no reinterpretation is needed (shapes already match)
             or possible (the mismatch is not proven size/dtype-compatible;
             the caller then binds the parameter directly and any genuine
             shape conflict surfaces as a feature gap inside the callee).
    """
    if (declared is None or not isinstance(declared, data.Array) or not isinstance(caller_descriptor, data.Array)
            or tuple(caller_descriptor.shape) == tuple(declared.shape)):
        return None
    if caller_descriptor.dtype != declared.dtype:
        return None
    try:
        if bool(prod(caller_descriptor.shape) != prod(declared.shape)):
            return None
    except Exception:
        return None  # Symbolic sizes that cannot prove equality: no reinterpretation
    return data.ArrayView(declared.dtype, list(declared.shape))


def _map_arguments(
    call: ast.Call, callee: Any, state: LoweringState
) -> Tuple[Dict[str, data.Data], Dict[str, Any], Dict[str, str], Dict[str, str], set, Tuple, List[Tuple[
        str, str, data.Data, data.Data]]]:
    """
    Map call arguments to callee parameters against the caller's context.
    Cheap and caller-state-dependent (runs on every call site, unlike the
    cached parse).

    :return: A 7-tuple of (argument descriptors by parameter, callee globals
             with specialized values, parameter-to-container bindings,
             argument label mapping, injected default-argument names, the
             hashable specialization key for the parse cache, and a list of
             pending argument-reinterpretation views — see
             ``_reshape_view_descriptor`` — as (view container, source
             container, source descriptor, view descriptor) tuples; their
             container is registered here but emission is deferred to the
             caller, which commits to inlining only after later checks pass).
    """
    parameter_names = list(callee.argnames)
    if len(call.args) > len(parameter_names):
        raise UnsupportedFeatureError(f'Too many arguments in call to "{callee.name}"',
                                      state.context.filename,
                                      call,
                                      category='inline-fallback:arguments')
    provided: Dict[str, ast.expr] = dict(zip(parameter_names, call.args))
    symbol_keywords: Dict[str, ast.expr] = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in provided:
            raise UnsupportedFeatureError(f'Unsupported keyword argument in call to "{callee.name}"',
                                          state.context.filename,
                                          call,
                                          category='inline-fallback:arguments')
        if keyword.arg not in parameter_names:
            # A keyword naming one of the callee's free SYMBOLS specializes it
            # for this call (``inner(A=A, symsym=tmp)``,
            # ``pressure_poisson(p, dx, dy, b, nit=nit)``) — the classic
            # frontend's symbol mapping on the nested SDFG. Anything else is a
            # keyword the callee has no use for.
            if not _names_callee_symbol(callee, keyword.arg):
                raise UnsupportedFeatureError(f'Unsupported keyword argument in call to "{callee.name}"',
                                              state.context.filename,
                                              call,
                                              category='inline-fallback:arguments')
            symbol_keywords[keyword.arg] = keyword.value
            continue
        provided[keyword.arg] = keyword.value

    callee_globals = dict(callee.global_vars)
    # Bound methods: "self" is resolved through the closure, not an argument
    # (mirrors parse_program's bound-method handling in nextgen/__init__.py)
    if callee.methodobj is not None and callee.objname is not None:
        callee_globals[callee.objname] = callee.methodobj
    injected_defaults = set()
    argtypes: Dict[str, data.Data] = {}
    parameter_bindings: Dict[str, str] = {}
    argument_labels: Dict[str, str] = {}
    pending_views: List[Tuple[str, str, data.Data, data.Data]] = []
    specialization = []
    shape_symbols: Dict[str, Any] = {}
    for parameter in parameter_names:
        if parameter not in provided:
            if parameter not in callee.default_args:
                raise UnsupportedFeatureError(f'Missing argument "{parameter}" in call to "{callee.name}"',
                                              state.context.filename,
                                              call,
                                              category='inline-fallback:arguments')
            default_value = callee.default_args[parameter]
            callee_globals[parameter] = default_value
            injected_defaults.add(parameter)
            specialization.append((parameter, 'default', repr(default_value)))
            continue
        argument = provided[parameter]
        argument_labels[parameter] = astutils.unparse(argument)
        inferred = state.inference.infer(argument)
        if inferred.is_pyobject:
            raise UnsupportedFeatureError(f'Opaque Python object passed to "{callee.name}"',
                                          state.context.filename,
                                          argument,
                                          category='pyobject-propagation')
        if inferred.is_data:
            if not isinstance(argument, ast.Name):
                raise UnsupportedFeatureError(f'Unsupported data argument form in call to "{callee.name}"',
                                              state.context.filename,
                                              argument,
                                              category='inline-fallback:arguments')
            container = state.context.container_of(argument.id, argument)
            caller_descriptor = state.context.containers[container]  # By reference: shared repository
            view_descriptor = _reshape_view_descriptor(caller_descriptor,
                                                       _declared_parameter_descriptor(callee, parameter))
            if view_descriptor is not None:
                view_container = state.context.add_container(f'{container}_view', view_descriptor)
                pending_views.append((view_container, container, caller_descriptor, view_descriptor))
                container = view_container
                argtypes[parameter] = view_descriptor
            else:
                argtypes[parameter] = caller_descriptor
            parameter_bindings[parameter] = container
            specialization.append((parameter, 'descriptor', repr(argtypes[parameter])))
            _collect_shape_symbols(_declared_parameter_descriptor(callee, parameter), argtypes[parameter], callee,
                                   shape_symbols)
        elif inferred.kind in ('constant', 'symbolic'):
            callee_globals[parameter] = inferred.value
            specialization.append((parameter, type(inferred.value).__name__, repr(inferred.value)))
        elif inferred.kind == 'static':
            constants = state.inference.sequence_constants(inferred.value)
            callee_globals[parameter] = tuple(constants) if inferred.value.kind == 'tuple' else constants
            specialization.append((parameter, 'static', repr(callee_globals[parameter])))
        else:
            raise UnsupportedFeatureError(f'Unsupported argument kind in call to "{callee.name}"',
                                          state.context.filename,
                                          argument,
                                          category='inline-fallback:arguments')
    # Applied before the explicit ``SYMBOL=value`` keywords below, which name
    # the same symbols deliberately and must win.
    for name, value in shape_symbols.items():
        if value is _CONFLICTING:
            continue
        callee_globals[name] = value
        specialization.append((name, 'symbol', repr(value)))
    pending_symbols = _map_symbol_keywords(symbol_keywords, callee, callee_globals, specialization, call, state)
    return (argtypes, callee_globals, parameter_bindings, argument_labels, injected_defaults, tuple(specialization),
            pending_views, pending_symbols)


#: Marker for a callee symbol two arguments pin to different sizes; it stays
#: free rather than taking either.
_CONFLICTING = object()


def _collect_shape_symbols(declared: Optional[data.Data], actual: data.Data, callee: Any, collected: Dict[str,
                                                                                                          Any]) -> None:
    """
    Collect the callee free symbols an argument's actual shape pins down: a
    parameter declared ``A: dace.float64[M]`` bound to a length-``k`` array
    means the callee's ``M`` IS the caller's ``k`` — everywhere in its body,
    including the shapes it allocates from it (``B = numpy.ndarray((M,))``).

    This is the inlining counterpart of the symbol mapping the classic
    frontend puts on a nested SDFG. Without it the callee's allocations keep
    the callee's own symbols, and a caller-side consumer of the result sees a
    shape unrelated to what it passed in (``numpy.dot`` of an ``(M,)`` result
    with a ``(k,)`` operand types as a mismatch, and the whole statement —
    then the whole enclosing loop — degrades to a callback).

    Only a dimension written as a BARE symbol is unified; a compound one
    (``M + 1``) would need real equation solving. Symbols are keyed by the
    name they have in the CALLEE's namespace, which is what its parse
    resolves; a symbol two parameters pin to different sizes is marked
    :data:`_CONFLICTING` and left free.
    """
    if declared is None or not isinstance(declared, data.Array) or not isinstance(actual, data.Array):
        return
    if len(declared.shape) != len(actual.shape):
        return
    callee_names: Dict[str, List[str]] = {}
    for name, value in getattr(callee, 'global_vars', {}).items():
        if isinstance(value, symbolic.symbol):
            callee_names.setdefault(value.name, []).append(name)
    for declared_size, actual_size in zip(declared.shape, actual.shape):
        if not isinstance(declared_size, symbolic.symbol):
            continue
        for name in callee_names.get(declared_size.name, ()):
            existing = collected.get(name)
            if existing is not None and existing != actual_size:
                collected[name] = _CONFLICTING
            elif existing is None:
                collected[name] = actual_size


def _names_callee_symbol(callee: Any, name: str) -> bool:
    """
    Whether a keyword names one of the callee's free symbols, under either
    spelling: the SYMBOL's name (``N=...`` for ``dace.symbol('N')``) or the
    name the callee's namespace binds it to, which differs for an anonymous
    symbol (``W = dace.symbol()`` is really ``sym_0``).
    """
    if name in getattr(callee, 'symbols', ()):
        return True
    return isinstance(getattr(callee, 'global_vars', {}).get(name), symbolic.symbol)


def _map_symbol_keywords(symbol_keywords: Dict[str, ast.expr], callee: Any, callee_globals: Dict[str, Any],
                         specialization: List, call: ast.Call, state: LoweringState) -> List[Tuple[str, str]]:
    """
    Specialize the callee's free symbols from ``SYMBOL=value`` keywords.

    A compile-time value (a constant or an expression over the caller's own
    symbols) is substituted directly into the callee's namespace, so its parse
    sees the caller's expression wherever the symbol appears — including in the
    shapes of the containers it allocates.

    A RUNTIME value (a scalar the caller computed) cannot be substituted, so it
    is promoted to a symbol of the caller: a fresh symbol is defined from the
    scalar by an interstate assignment emitted immediately before the call
    (returned as pending work, since inlining is not committed yet), and the
    callee is specialized to that symbol. This is the schedule-tree equivalent
    of the classic frontend's symbol mapping on a nested SDFG. The symbol is
    named per CALL SITE: two sites passing different values must not share one,
    since containers allocated by the first are still sized by it.

    :return: (symbol name, source container) pairs whose interstate assignment
             the caller emits once it commits to inlining.
    """
    pending_symbols: List[Tuple[str, str]] = []
    for name, argument in symbol_keywords.items():
        inferred = state.inference.infer(argument)
        if inferred.kind in ('constant', 'symbolic'):
            callee_globals[name] = inferred.value
            specialization.append((name, 'symbol', repr(inferred.value)))
            continue
        if not inferred.is_data or not isinstance(argument, ast.Name):
            raise UnsupportedFeatureError(f'Unsupported symbol argument "{name}" in call to "{callee.name}"',
                                          state.context.filename,
                                          argument,
                                          category='inline-fallback:arguments')
        container = state.context.container_of(argument.id, argument)
        descriptor = state.context.containers[container]
        if data._prod(descriptor.shape) != 1:
            raise UnsupportedFeatureError(f'Symbol argument "{name}" in call to "{callee.name}" is not a scalar',
                                          state.context.filename,
                                          argument,
                                          category='inline-fallback:arguments')
        symbol_name = f'__sym_{name}_{call.lineno}_{call.col_offset}'
        state.context.symbols.setdefault(symbol_name, symbolic.symbol(symbol_name, descriptor.dtype))
        callee_globals[name] = state.context.symbols[symbol_name]
        specialization.append((name, 'symbol', symbol_name))
        pending_symbols.append((symbol_name, container))
    return pending_symbols


def _parse_callee(callee: Any, argtypes: Dict[str, data.Data], callee_globals: Dict[str, Any],
                  injected_defaults: set) -> 'parse_cache.CalleeParse':
    """
    Preprocess and canonicalize a callee. Pure with respect to the caller's
    lowering state — it touches no context, emitter, or inference objects —
    so results are cacheable (and, later, parallelizable).
    """
    from dace.frontend.python import preprocessing  # Deferred to keep rule import light
    from dace.frontend.python.nextgen.canonical.passes import default_passes
    from dace.frontend.python.nextgen.lowering import parse_cache
    from dace.frontend.python.nextgen.pipeline import CanonicalizationPipeline, PipelineContext

    modules = {key: value.__name__ for key, value in callee_globals.items() if isinstance(value, types.ModuleType)}
    modules['builtins'] = ''
    parsed_ast, closure = preprocessing.preprocess_dace_program(callee.f,
                                                                argtypes,
                                                                callee_globals,
                                                                modules,
                                                                resolve_functions=callee.resolve_functions,
                                                                default_args=injected_defaults)

    program_ast = parsed_ast.preprocessed_ast
    program_node = program_ast.body[0] if isinstance(program_ast, ast.Module) else program_ast
    if not isinstance(program_node, ast.FunctionDef):
        raise UnsupportedFeatureError(f'Preprocessing "{callee.name}" did not produce a function', parsed_ast.filename)
    pipeline_context = PipelineContext(callee.name, parsed_ast.filename, parsed_ast.program_globals, argtypes)
    program = CanonicalizationPipeline(default_passes()).run(program_node, pipeline_context)
    return parse_cache.CalleeParse(canonical_body=program.body,
                                   program_globals=parsed_ast.program_globals,
                                   closure=closure,
                                   filename=parsed_ast.filename)


def _unsupported_return_shape(body: List[ast.stmt]) -> Optional[str]:
    """
    Check the return statements of a canonical callee body for shapes that
    cannot be inlined soundly:

    - returns swallowed by opaque statements (a ``return`` cannot execute
      inside a Python callback),
    - returns of inconsistent arity (they would materialize into different
      container sets),
    - value-returning functions where control may fall through the end (the
      Python result would be ``None`` on that path, which dataflow cannot
      represent).

    Early (non-tail) returns of a single consistent arity are supported: they
    lower to :class:`ReturnNode`, which exits the enclosing
    :class:`FunctionCallScope`.

    :return: A human-readable reason, or None if the shape is supported.
    """
    from dace.frontend.python.nextgen.canonical.cpa import OpaqueStmt

    arities: set = set()

    def _scan(statements: List[ast.stmt]) -> Optional[str]:
        for node in statements:
            if isinstance(node, ast.Return):
                if node.value is None:
                    arities.add(0)
                elif isinstance(node.value, ast.Tuple):
                    arities.add(len(node.value.elts))
                else:
                    arities.add(1)
                continue
            if isinstance(node, OpaqueStmt):
                for original in node.originals:
                    if any(isinstance(inner, ast.Return) for inner in ast.walk(original)):
                        return 'return inside an interpreter-fallback region'
                continue
            for field in ('body', 'orelse'):
                child = getattr(node, field, None)
                if child:
                    reason = _scan(child)
                    if reason is not None:
                        return reason
        return None

    reason = _scan(body)
    if reason is not None:
        return reason
    if len(arities) > 1:
        return 'inconsistent return arities'
    if arities and 0 not in arities and not _always_returns(body):
        return 'control may fall through without returning'
    return None


def _normalize_early_returns(body: List[ast.stmt]) -> List[ast.stmt]:
    """
    Restructure a canonical callee body so that every ``return`` sits in tail
    position of its control path: statements following an if-statement in
    which one branch always returns are hoisted into the other branch, and
    statements following an unconditional return are dropped (dead code).
    Returns inside loops cannot be restructured this way and are left in
    place for :func:`_has_non_tail_return` to reject.
    """
    body = list(body)
    for index, node in enumerate(body):
        if isinstance(node, ast.Return):
            return body[:index + 1]  # Anything after an unconditional return is dead
        if isinstance(node, ast.If):
            node.body = _normalize_early_returns(node.body)
            node.orelse = _normalize_early_returns(node.orelse)
            rest = body[index + 1:]
            body_returns = _always_returns(node.body)
            orelse_returns = bool(node.orelse) and _always_returns(node.orelse)
            if body_returns and orelse_returns:
                return body[:index + 1]  # Both branches return; the rest is dead
            if rest and (body_returns or orelse_returns):
                if body_returns:
                    node.orelse = _normalize_early_returns(list(node.orelse) + rest)
                else:
                    node.body = _normalize_early_returns(list(node.body) + rest)
                return body[:index + 1]
    return body


def _has_non_tail_return(body: List[ast.stmt]) -> bool:
    """Whether any ``return`` remains outside tail position (e.g., inside a
    loop, or in a branch that only sometimes returns with statements
    following) after :func:`_normalize_early_returns`."""
    for index, node in enumerate(body):
        in_tail = index == len(body) - 1
        if isinstance(node, ast.Return):
            if not in_tail:
                return True
        elif isinstance(node, ast.If):
            if in_tail:
                if _has_non_tail_return(node.body) or _has_non_tail_return(node.orelse):
                    return True
            elif any(isinstance(inner, ast.Return) for child in node.body + node.orelse for inner in ast.walk(child)):
                return True
        elif any(isinstance(inner, ast.Return) for inner in ast.walk(node)):
            return True  # Returns inside loops/other compounds cannot be restructured
    return False


def _strip_tail_returns(scope: tn.ScheduleTreeScope) -> None:
    """
    Remove :class:`ReturnNode`\\ s in tail position anywhere in an inlined
    callee scope. After early-return normalization every return sits at the
    end of its control path, where exiting the callee coincides with falling
    off the scope end — the nodes carry no remaining semantics.
    """
    children = scope.children
    while children and isinstance(children[-1], tn.ReturnNode):
        children.pop()
    index = len(children) - 1
    while index >= 0 and isinstance(children[index], (tn.ElifScope, tn.ElseScope)):
        _strip_tail_returns(children[index])
        index -= 1
    if index >= 0 and isinstance(children[index], tn.IfScope):
        _strip_tail_returns(children[index])


def _always_returns(body: List[ast.stmt]) -> bool:
    """Whether every control-flow path through a canonical statement list ends
    in a return (conservative: loops are assumed to possibly not execute)."""
    for node in body:
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, ast.If) and node.orelse and _always_returns(node.body) and _always_returns(node.orelse):
            return True
    return False


def _bind_call_results(target: Optional[ast.expr], returned: List[str], statement: ast.stmt,
                       state: LoweringState) -> None:
    """
    Bind an inlined callee's materialized return containers to the caller's
    assignment target. Runs after :meth:`inline_scope` exits, in the caller's
    binding scope; the repository container names remain valid.
    """
    if target is None or not returned:
        return  # Bare call or no return value: results are discarded

    # The return containers become caller-visible under their repository names
    for name in returned:
        state.context.bind(name, name)

    if isinstance(target, ast.Name):
        if len(returned) == 1:
            state.context.bind(target.id, returned[0])
        else:
            # Tuple results stay in the value domain as a static sequence of
            # container references; element reads fold to direct accesses.
            elements = [ast.copy_location(ast.Name(id=name, ctx=ast.Load()), statement) for name in returned]
            state.context.bind_static(target.id, StaticSequence(elements=elements, kind='tuple'))
        return

    if isinstance(target, ast.Subscript) and len(returned) == 1:
        from dace.frontend.python.nextgen.lowering.access import resolve_access
        target_access = resolve_access(target, state)
        source_descriptor = state.context.containers[returned[0]]
        if target_access is not None:
            state.emitter.emit(
                tn.CopyNode(target=target_access.container,
                            memlet=Memlet(data=returned[0],
                                          subset=subsets.Range.from_array(source_descriptor),
                                          other_subset=target_access.subset)))
            return

    raise UnsupportedFeatureError('Unsupported assignment target for a nested program call',
                                  state.context.filename,
                                  statement,
                                  category='inline-fallback:result-binding')


def _convertible_to_sdfg(callee: Any, call: ast.Call, state: LoweringState) -> Optional[Any]:
    """
    Produce an SDFG from a non-DaceProgram SDFG-convertible, if possible.
    ``__sdfg__`` receives the inferred argument descriptors (data) and values
    (constants/symbols), mirroring the classic frontend's convention.
    """
    from dace.sdfg import SDFG
    if not hasattr(callee, '__sdfg__'):
        return None

    # Failing to resolve the arguments means this call site cannot describe the
    # callee, not that the callee is broken: stay silent and let the caller fall
    # back to a callback.
    try:
        arguments = [_sdfg_convertible_argument(argument, state) for argument in call.args]
        keywords = {
            keyword.arg: _sdfg_convertible_argument(keyword.value, state)
            for keyword in call.keywords if keyword.arg is not None
        }
    except Exception:
        return None

    # An error raised by ``__sdfg__`` itself is the user's, and propagates the
    # way the classic frontend propagates it (see ``newast.py``'s handling of
    # ``raise_nested_parsing_errors``). Swallowing it here would silently
    # downgrade the call to a Python callback and hide the real failure.
    from dace.frontend.python.parser import DaceProgram  # Deferred to avoid an import cycle
    if isinstance(callee, DaceProgram):
        # ``__sdfg__`` parses with ``simplify=None``, i.e. the config default,
        # which would simplify the callee on its own and inline away any nesting
        # INSIDE it before the caller ever sees it. Parse it unsimplified and
        # let the caller's own simplification decide, the way the classic
        # frontend propagates its ``simplify`` to nested programs
        # (``newast.py:4141``).
        sdfg = callee.to_sdfg(*arguments, simplify=False, save=False, **keywords)
    else:
        sdfg = callee.__sdfg__(*arguments, **keywords)
    return sdfg if isinstance(sdfg, SDFG) else None


def _sdfg_convertible_argument(argument: ast.expr, state: LoweringState) -> Any:
    """The data descriptor (data arguments) or compile-time value
    (constant/symbolic arguments) an SDFG-convertible's ``__sdfg__`` sees."""
    inferred = state.inference.infer(argument)
    if inferred.is_data:
        return inferred.descriptor
    return inferred.value


def _sdfg_call_argument(argument: ast.expr, state: LoweringState) -> str:
    """
    The text an SDFG call passes for one argument: the repository container a
    name is bound to, or the argument as written when it denotes no container.

    A name and the container behind it part ways as soon as the name is
    rebound -- ``new_sym = anotherarray`` binds ``new_sym`` to the closure's
    ``__g_anotherarray`` -- and the consumer of this call
    (:meth:`~dace.sdfg.analysis.schedule_tree.tree_to_sdfg.
    ScheduleTreeToSDFG.visit_SDFGCallNode`) matches the text against the
    caller's containers to tell a data argument from a symbolic one. Handing it
    the source name makes a container argument look symbolic, and the callee's
    parameter is then added as a symbol of an SDFG that already has a data
    descriptor under that name.
    """
    if isinstance(argument, ast.Name):
        binding = state.context.resolve(argument.id)
        if binding is not None and binding.kind == 'container' and binding.container in state.context.containers:
            return binding.container
    return astutils.unparse(argument)


def _sdfg_positional_parameters(callee: Any, sdfg: Any) -> Tuple[List[Optional[str]], Set[str]]:
    """
    The parameter names an SDFG callee's positional arguments bind to, in call
    order, together with the names that are compile-time constants.

    An SDFG-convertible may declare constant parameters through
    ``__sdfg_signature__``. Those are baked into the SDFG it returns and so do
    not appear in ``arg_names``, yet they still occupy a position in the call.
    Zipping ``arg_names`` against the call's arguments directly would shift
    every later argument left by one -- binding a data argument to the name of
    a constant one, which then reaches
    :meth:`~dace.sdfg.analysis.schedule_tree.tree_to_sdfg.ScheduleTreeToSDFG.
    visit_SDFGCallNode` as a symbolic argument and is added as a symbol of an
    SDFG that already holds a data descriptor under that name.

    Constant positions are returned as ``None`` so that alignment is preserved.
    """
    constants: Set[str] = set()
    signature = getattr(callee, '__sdfg_signature__', None)
    if signature is not None:
        try:
            names, constant_names = signature()
        except Exception:
            names, constant_names = None, None
        if names is not None:
            constants = set(constant_names or ())
            return [None if name in constants else name for name in names], constants
    return list(getattr(sdfg, 'arg_names', None) or []), constants


def _lower_sdfg_call(target: Optional[ast.expr],
                     call: ast.Call,
                     sdfg: Any,
                     statement: ast.stmt,
                     state: LoweringState,
                     callee: Any = None) -> None:
    """
    Emit an explicit :class:`SDFGCallNode` for an SDFG-valued callee. The SDFG
    stays a black box; only its return containers are registered (as copies)
    so the caller can consume the results.
    """
    from dace.frontend.python.nextgen.lowering.dispatch import fallback_to_callback

    arguments: Dict[str, str] = {}
    argument_names, constant_names = _sdfg_positional_parameters(callee, sdfg)
    for name, argument in zip(argument_names, call.args):
        if name is None:  # A compile-time argument, already baked into the SDFG
            continue
        arguments[name] = _sdfg_call_argument(argument, state)
    for keyword in call.keywords:
        if keyword.arg is not None and keyword.arg not in constant_names:
            arguments[keyword.arg] = _sdfg_call_argument(keyword.value, state)

    return_targets: List[str] = []
    result_copy: Optional[tn.CopyNode] = None
    if target is not None:
        from dace.frontend.python.nextgen.lowering.access import resolve_access

        return_descriptors = {
            name: descriptor
            for name, descriptor in sdfg.arrays.items() if name.startswith('__return')
        }
        if len(return_descriptors) != 1:
            fallback_to_callback(statement,
                                 state,
                                 'unsupported result binding for an SDFG call',
                                 category='inline-fallback:result-binding')
            return
        descriptor = copy.deepcopy(next(iter(return_descriptors.values())))
        if isinstance(target, ast.Name):
            container = state.context.add_container(target.id, descriptor)
            state.context.bind(target.id, container)
        elif isinstance(target, ast.Subscript):
            # ``B[:] = sdfg(...)``: the SDFG still writes its own return
            # container, which is then copied into the target subset.
            target_access = resolve_access(target, state)
            if target_access is None:
                fallback_to_callback(statement,
                                     state,
                                     'unsupported result binding for an SDFG call',
                                     category='inline-fallback:result-binding')
                return
            container = state.context.add_container(f'__{sdfg.name}_result', descriptor)
            result_copy = tn.CopyNode(target=target_access.container,
                                      memlet=Memlet(data=container,
                                                    subset=subsets.Range.from_array(descriptor),
                                                    other_subset=target_access.subset))
        else:
            fallback_to_callback(statement,
                                 state,
                                 'unsupported result binding for an SDFG call',
                                 category='inline-fallback:result-binding')
            return
        return_targets.append(container)

    state.emitter.emit(
        tn.SDFGCallNode(sdfg=sdfg,
                        call=tn.FrontendFunctionCall(callee_name=sdfg.name, arguments=arguments),
                        return_targets=return_targets))
    if result_copy is not None:
        state.emitter.emit(result_copy)
