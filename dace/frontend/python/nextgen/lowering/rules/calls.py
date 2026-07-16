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
    dispatch.fallback_to_callback(statement, state, 'bare expression statement')


def is_sdfg_convertible(callee: Any) -> bool:
    """Whether a resolved callee is a nested dace program or an SDFG-valued object."""
    if callee is None:
        return False
    from dace.frontend.python import common as pycommon  # Deferred to avoid an import cycle
    from dace.sdfg import SDFG
    return isinstance(callee, (SDFG, pycommon.SDFGConvertible))


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
        sdfg = callee if isinstance(callee, SDFG) else _convertible_to_sdfg(callee)
        if sdfg is None:
            fallback_to_callback(statement, state, 'SDFG-convertible callee could not produce an SDFG')
            return
        _lower_sdfg_call(target, call, sdfg, statement, state)
        return

    if callee.f in state.context.inline_stack:
        fallback_to_callback(statement, state, 'recursive @dace.program call')
        return

    # Steps 1-3 (argument mapping, preprocessing, canonicalization) emit
    # nothing; any failure here falls back to the interpreter, preserving
    # totality. Failures during emission (step 4) are frontend bugs and raise.
    try:
        callee_body, parameter_bindings, callee_globals, argument_labels = _prepare_callee(call, callee, state)
    except Exception as reason:  # Unparseable callee, unsupported argument, ...
        fallback_to_callback(statement, state, f'cannot inline call to "{callee.name}": {reason}')
        return
    if _has_early_return(callee_body):
        fallback_to_callback(statement, state, f'early return in nested dace program "{callee.name}"')
        return

    return_prefix = state.context.fresh_name(f'__{callee.name}_ret')
    scope = tn.FunctionCallScope(call=tn.FrontendFunctionCall(callee_name=callee.name, arguments=argument_labels),
                                 children=[])
    with state.context.inline_scope(callee.f, parameter_bindings, callee_globals, return_prefix) as return_names:
        with state.emitter.scope(scope):
            state.lower_body(callee_body)
        returned = list(return_names)
    _bind_call_results(target, returned, statement, state)


def _prepare_callee(call: ast.Call, callee: Any,
                    state: LoweringState) -> Tuple[List[ast.stmt], Dict[str, str], Dict[str, Any], Dict[str, str]]:
    """
    Map call arguments to callee parameters, preprocess the callee, and
    canonicalize its body against the shared repository.

    Data arguments bind parameters to the caller's repository containers (by
    reference); constant, symbolic, and compile-time-sequence arguments
    specialize the callee through its globals.

    :return: A 4-tuple of (canonical callee body, parameter-to-container
             bindings, resolved callee globals, argument label mapping).
    """
    from dace.frontend.python import preprocessing  # Deferred to keep rule import light
    from dace.frontend.python.nextgen.canonical.passes import default_passes
    from dace.frontend.python.nextgen.pipeline import CanonicalizationPipeline, PipelineContext

    parameter_names = list(callee.argnames)
    if len(call.args) > len(parameter_names):
        raise UnsupportedFeatureError(f'Too many arguments in call to "{callee.name}"', state.context.filename, call)
    provided: Dict[str, ast.expr] = dict(zip(parameter_names, call.args))
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in provided or keyword.arg not in parameter_names:
            raise UnsupportedFeatureError(f'Unsupported keyword argument in call to "{callee.name}"',
                                          state.context.filename, call)
        provided[keyword.arg] = keyword.value

    callee_globals = dict(callee.global_vars)
    injected_defaults = set()
    argtypes: Dict[str, data.Data] = {}
    parameter_bindings: Dict[str, str] = {}
    argument_labels: Dict[str, str] = {}
    for parameter in parameter_names:
        if parameter not in provided:
            if parameter not in callee.default_args:
                raise UnsupportedFeatureError(f'Missing argument "{parameter}" in call to "{callee.name}"',
                                              state.context.filename, call)
            callee_globals[parameter] = callee.default_args[parameter]
            injected_defaults.add(parameter)
            continue
        argument = provided[parameter]
        argument_labels[parameter] = astutils.unparse(argument)
        inferred = state.inference.infer(argument)
        if inferred.is_pyobject:
            raise UnsupportedFeatureError(f'Opaque Python object passed to "{callee.name}"', state.context.filename,
                                          argument)
        if inferred.is_data:
            if not isinstance(argument, ast.Name):
                raise UnsupportedFeatureError(f'Unsupported data argument form in call to "{callee.name}"',
                                              state.context.filename, argument)
            container = state.context.container_of(argument.id, argument)
            argtypes[parameter] = state.context.containers[container]  # By reference: shared repository
            parameter_bindings[parameter] = container
        elif inferred.kind in ('constant', 'symbolic'):
            callee_globals[parameter] = inferred.value
        elif inferred.kind == 'static':
            constants = state.inference.sequence_constants(inferred.value)
            callee_globals[parameter] = tuple(constants) if inferred.value.kind == 'tuple' else constants
        else:
            raise UnsupportedFeatureError(f'Unsupported argument kind in call to "{callee.name}"',
                                          state.context.filename, argument)

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

    # Merge closure metadata into the shared repository/root
    for key, (original, _, _) in closure.callbacks.items():
        state.emitter.root.callback_mapping.setdefault(key, original)
    for constant_name, value in closure.closure_constants.items():
        try:
            descriptor = data.create_datadescriptor(value)
        except (TypeError, ValueError):
            continue
        state.context.constants.setdefault(constant_name, (descriptor, value))

    program_ast = parsed_ast.preprocessed_ast
    program_node = program_ast.body[0] if isinstance(program_ast, ast.Module) else program_ast
    if not isinstance(program_node, ast.FunctionDef):
        raise UnsupportedFeatureError(f'Preprocessing "{callee.name}" did not produce a function',
                                      state.context.filename, call)
    pipeline_context = PipelineContext(callee.name, parsed_ast.filename, parsed_ast.program_globals, argtypes)
    program = CanonicalizationPipeline(default_passes()).run(program_node, pipeline_context)
    return program.body, parameter_bindings, parsed_ast.program_globals, argument_labels


def _has_early_return(body: List[ast.stmt]) -> bool:
    """
    Whether the canonical callee body contains a return anywhere other than
    the tail of the top-level statement list (including returns swallowed by
    opaque statements, which cannot execute inside a callback).
    """
    from dace.frontend.python.nextgen.canonical.cpa import OpaqueStmt

    def _contains(statements: List[ast.stmt], allow_tail: bool) -> bool:
        for index, node in enumerate(statements):
            if isinstance(node, ast.Return):
                if not (allow_tail and index == len(statements) - 1):
                    return True
                continue
            if isinstance(node, OpaqueStmt):
                if any(isinstance(inner, ast.Return) for inner in ast.walk(node.original)):
                    return True
                continue
            for field in ('body', 'orelse'):
                child = getattr(node, field, None)
                if child and _contains(child, False):
                    return True
        return False

    return _contains(body, True)


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

    raise UnsupportedFeatureError('Unsupported assignment target for a nested program call', state.context.filename,
                                  statement)


def _convertible_to_sdfg(callee: Any) -> Optional[Any]:
    """Produce an SDFG from a non-DaceProgram SDFG-convertible, if possible."""
    try:
        sdfg = callee.__sdfg__()
    except Exception:
        return None
    from dace.sdfg import SDFG
    return sdfg if isinstance(sdfg, SDFG) else None


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
            fallback_to_callback(statement, state, 'unsupported result binding for an SDFG call')
            return
        descriptor = copy.deepcopy(next(iter(return_descriptors.values())))
        container = state.context.add_container(target.id, descriptor)
        state.context.bind(target.id, container)
        return_targets.append(container)

    state.emitter.emit(
        tn.SDFGCallNode(sdfg=sdfg,
                        call=tn.FrontendFunctionCall(callee_name=sdfg.name, arguments=arguments),
                        return_targets=return_targets))
