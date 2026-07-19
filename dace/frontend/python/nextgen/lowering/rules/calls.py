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
from typing import Any, Dict, List, Optional, Tuple

from dace import data, subsets
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
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

    if not isinstance(callee, DaceProgram):
        sdfg = callee if isinstance(callee, SDFG) else _convertible_to_sdfg(callee, call, state)
        if sdfg is None:
            fallback_to_callback(statement,
                                 state,
                                 'SDFG-convertible callee could not produce an SDFG',
                                 category='inline-fallback:no-sdfg')
            return
        _lower_sdfg_call(target, call, sdfg, statement, state)
        return

    if callee.f in state.context.inline_stack:
        fallback_to_callback(statement, state, 'recursive @dace.program call', category='inline-fallback:recursion')
        return

    # Steps 1-3 (argument mapping, preprocessing, canonicalization) emit
    # nothing; any failure here falls back to the interpreter, preserving
    # totality. Failures during emission (step 4) are frontend bugs and raise.
    try:
        callee_body, parameter_bindings, callee_globals, argument_labels = _prepare_callee(call, callee, state)
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

    return_prefix = state.context.fresh_name(f'__{callee.name}_ret')
    scope = tn.FunctionCallScope(call=tn.FrontendFunctionCall(callee_name=callee.name, arguments=argument_labels),
                                 children=[])
    with state.context.inline_scope(callee.f, parameter_bindings, callee_globals, return_prefix) as return_names:
        with state.emitter.scope(scope):
            state.lower_body(callee_body)
        _strip_tail_returns(scope)  # Tail returns fall off the scope end
        returned = list(dict.fromkeys(return_names))
    _bind_call_results(target, returned, statement, state)


def _prepare_callee(call: ast.Call, callee: Any,
                    state: LoweringState) -> Tuple[List[ast.stmt], Dict[str, str], Dict[str, Any], Dict[str, str]]:
    """
    Map call arguments to callee parameters, then fetch (or produce) the
    callee's preprocessed and canonicalized parse through the per-program
    parse cache: a callee invoked from multiple call sites with the same
    specialization parses once.

    Data arguments bind parameters to the caller's repository containers (by
    reference); constant, symbolic, and compile-time-sequence arguments
    specialize the callee through its globals.

    :return: A 4-tuple of (canonical callee body — a fresh deep copy, since
             lowering mutates it, parameter-to-container bindings, resolved
             callee globals, argument label mapping).
    """
    argtypes, callee_globals, parameter_bindings, argument_labels, injected_defaults, spec_key = _map_arguments(
        call, callee, state)

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
    for callback_name, (original, _, _) in closure.callbacks.items():
        state.emitter.root.callback_mapping.setdefault(callback_name, original)
    for constant_name, value in closure.closure_constants.items():
        try:
            descriptor = data.create_datadescriptor(value)
        except (TypeError, ValueError):
            continue
        state.context.constants.setdefault(constant_name, (descriptor, value))
    # External arrays the callee references bind inside its inline scope,
    # deduplicated by qualified name across the whole program
    for reference_name, (qualified_name, descriptor, _, _) in closure.closure_arrays.items():
        container = state.context.register_closure_array(reference_name, qualified_name, descriptor)
        parameter_bindings[reference_name] = container

    # Lowering mutates the body (early-return restructuring, annotation
    # hints), so every call site works on its own copy — but objects embedded
    # by preprocessing (resolved dace programs, SDFGs, arbitrary constants)
    # must keep their IDENTITY across copies: resolution, convertibility
    # checks, and the cache key itself are identity-based, and deep-copying a
    # DaceProgram would clone its entire global namespace.
    memo: Dict[int, Any] = {}
    _seed_embedded_objects(parse.canonical_body, memo)
    body = copy.deepcopy(parse.canonical_body, memo)
    return body, parameter_bindings, parse.program_globals, argument_labels


def _seed_embedded_objects(statements: List[ast.stmt], memo: Dict[int, Any]) -> None:
    """Pre-map every non-literal ``ast.Constant`` value (an object embedded by
    preprocessing) to itself in a deepcopy memo, so copies share the object.
    Canonical leaf markers hide their contents from ``ast.walk``
    (``_fields = ()``), so their statement payloads walk explicitly."""
    from dace.frontend.python.nextgen.canonical import cpa
    from dace.frontend.python.nextgen.semantics.inference import is_literal_constant
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Constant) and not is_literal_constant(node.value):
                memo[id(node.value)] = node.value
            if isinstance(node, cpa.OpaqueStmt):
                _seed_embedded_objects(node.originals, memo)
            elif isinstance(node, cpa.ExplicitTasklet):
                _seed_embedded_objects(node.statements, memo)
                if node.original is not None:
                    _seed_embedded_objects([node.original], memo)
            elif isinstance(node, cpa.ExplicitConsume):
                _seed_embedded_objects(node.statements, memo)
                if node.original is not None:
                    _seed_embedded_objects([node.original], memo)


def _map_arguments(
        call: ast.Call, callee: Any, state: LoweringState
) -> Tuple[Dict[str, data.Data], Dict[str, Any], Dict[str, str], Dict[str, str], set, Tuple]:
    """
    Map call arguments to callee parameters against the caller's context.
    Cheap and caller-state-dependent (runs on every call site, unlike the
    cached parse).

    :return: A 6-tuple of (argument descriptors by parameter, callee globals
             with specialized values, parameter-to-container bindings,
             argument label mapping, injected default-argument names, and the
             hashable specialization key for the parse cache).
    """
    parameter_names = list(callee.argnames)
    if len(call.args) > len(parameter_names):
        raise UnsupportedFeatureError(f'Too many arguments in call to "{callee.name}"',
                                      state.context.filename,
                                      call,
                                      category='inline-fallback:arguments')
    provided: Dict[str, ast.expr] = dict(zip(parameter_names, call.args))
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in provided or keyword.arg not in parameter_names:
            raise UnsupportedFeatureError(f'Unsupported keyword argument in call to "{callee.name}"',
                                          state.context.filename,
                                          call,
                                          category='inline-fallback:arguments')
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
    specialization = []
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
            argtypes[parameter] = state.context.containers[container]  # By reference: shared repository
            parameter_bindings[parameter] = container
            specialization.append((parameter, 'descriptor', repr(argtypes[parameter])))
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
    return argtypes, callee_globals, parameter_bindings, argument_labels, injected_defaults, tuple(specialization)


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

    modules = {
        key: value.__name__
        for key, value in callee_globals.items() if hasattr(value, '__name__') and type(value).__name__ == 'module'
    }
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
    try:
        arguments = [_sdfg_call_argument(argument, state) for argument in call.args]
        keywords = {
            keyword.arg: _sdfg_call_argument(keyword.value, state)
            for keyword in call.keywords if keyword.arg is not None
        }
        sdfg = callee.__sdfg__(*arguments, **keywords)
    except Exception:
        return None
    return sdfg if isinstance(sdfg, SDFG) else None


def _sdfg_call_argument(argument: ast.expr, state: LoweringState) -> Any:
    """The data descriptor (data arguments) or compile-time value
    (constant/symbolic arguments) an SDFG-convertible's ``__sdfg__`` sees."""
    inferred = state.inference.infer(argument)
    if inferred.is_data:
        return inferred.descriptor
    return inferred.value


def _lower_sdfg_call(target: Optional[ast.expr], call: ast.Call, sdfg: Any, statement: ast.stmt,
                     state: LoweringState) -> None:
    """
    Emit an explicit :class:`SDFGCallNode` for an SDFG-valued callee. The SDFG
    stays a black box; only its return containers are registered (as copies)
    so the caller can consume the results.
    """
    from dace.frontend.python.nextgen.lowering.dispatch import fallback_to_callback

    arguments: Dict[str, str] = {}
    argument_names = list(getattr(sdfg, 'arg_names', None) or [])
    for name, argument in zip(argument_names, call.args):
        arguments[name] = astutils.unparse(argument)
    for keyword in call.keywords:
        if keyword.arg is not None:
            arguments[keyword.arg] = astutils.unparse(keyword.value)

    return_targets: List[str] = []
    if target is not None:
        return_descriptors = {
            name: descriptor
            for name, descriptor in sdfg.arrays.items() if name.startswith('__return')
        }
        if not isinstance(target, ast.Name) or len(return_descriptors) != 1:
            fallback_to_callback(statement,
                                 state,
                                 'unsupported result binding for an SDFG call',
                                 category='inline-fallback:result-binding')
            return
        descriptor = copy.deepcopy(next(iter(return_descriptors.values())))
        container = state.context.add_container(target.id, descriptor)
        state.context.bind(target.id, container)
        return_targets.append(container)

    state.emitter.emit(
        tn.SDFGCallNode(sdfg=sdfg,
                        call=tn.FrontendFunctionCall(callee_name=sdfg.name, arguments=arguments),
                        return_targets=return_targets))
