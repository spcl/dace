# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rule for :class:`OpaqueStmt` markers: statements that must execute in
the Python interpreter become fully specified :class:`PythonCallbackNode`\\ s.

The callback contract:

- ``input_names``/``output_names`` list the repository containers the callback
  reads and writes, derived from the statement's precomputed I/O sets.
- Written names that have no container yet are registered as ``pyobject``
  scalars so subsequent statements can reference (and pass through) the
  resulting Python objects.
- The node is a side-effect fence: later passes must not reorder memory
  accesses across it. (Enforced by the verifier through the presence of the
  reason and I/O metadata; the tree-to-SDFG lowering maps this onto the
  ``__pystate`` serialization edge of the stable frontend's callback ABI.)
"""
import ast
import builtins
import copy
import warnings
from typing import Optional

from dace import data, dtypes
from dace.config import Config
from dace.frontend.python import astutils
from dace.frontend.python.common import DaceSyntaxError
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.canonical.cpa import OpaqueStmt
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule


@rule(OpaqueStmt)
def lower_opaque(statement: OpaqueStmt, state: LoweringState) -> None:
    _warn_about_the_callback(statement, state)
    # A compile-time sequence has no container to pass, so the temporary ANF
    # hoisted it into would reach the interpreter as an undefined name. Put the
    # sequence expression back where the temporary stands; the containers it
    # names then become ordinary inputs below.
    static_expansions = _static_expansions(statement, state)
    referenced = {name for expression in static_expansions.values() for name in _referenced_names(expression)}

    source_to_repository: dict = {}
    input_names = []
    for name in sorted(set(statement.inputs) | referenced):
        binding = state.context.resolve(name)
        if binding is not None and binding.kind == 'container':
            if binding.container not in input_names:
                input_names.append(binding.container)
            source_to_repository[name] = binding.container

    output_names = []
    for name in sorted(statement.outputs):
        binding = state.context.resolve(name)
        if binding is not None and binding.kind == 'container':
            output_names.append(binding.container)
        else:
            # The callback produces a value inference cannot see: an
            # annotation on the original statement types it; otherwise
            # register an opaque scalar so later statements can bind and
            # pass the Python object through.
            descriptor = _declared_descriptor(name, statement, state)
            untyped_call = False
            if descriptor is None:
                _report_untyped_result(name, statement, state)
                descriptor = data.Scalar(dtypes.pyobject())
                # Only a CALL's result is worth reporting later: it is the one
                # opaque value the user can type, through an annotation or a
                # return hint. A dict literal has no return type to declare.
                untyped_call = _callee_name(statement) is not None
            container_name = state.context.add_container(name, descriptor)
            state.context.bind(name, container_name)
            if untyped_call:
                state.context.untyped_call_results.add(container_name)
            output_names.append(container_name)
        source_to_repository[name] = output_names[-1]

    reconstituted = [_reconstitute_source(original, state) for original in statement.originals]
    if static_expansions:
        reconstituted = [_substitute_names(original, static_expansions) for original in reconstituted]
    code = '\n'.join(ast.unparse(original) for original in reconstituted)
    renamed = _rename_to_repository(reconstituted, source_to_repository)
    _register_free_globals(renamed, set(input_names) | set(output_names), state)

    # Emission-time batching: if this scope's previous node is already a
    # callback, extend it instead of emitting a second one. Both statement
    # runs executed adjacently in the interpreter anyway, so merging changes
    # callback granularity, not semantics (and keeps the fence contract:
    # relative order within the merged run is preserved).
    children = state.emitter.current_scope.children
    previous = children[-1] if children else None
    if isinstance(previous, tn.PythonCallbackNode):
        _merge_into(previous, statement.reason, code, renamed, input_names, output_names, state)
        return

    function_name = state.context.fresh_name('__nextgen_callback')
    function_code, call_code = _outline(renamed, function_name, input_names, output_names, state)
    state.emitter.emit(
        tn.PythonCallbackNode(code=CodeBlock(code),
                              reason=statement.reason,
                              input_names=input_names,
                              output_names=output_names,
                              outlined_function_name=function_name,
                              outlined_function_code=function_code,
                              outlined_call_code=call_code))


def _merge_into(previous: tn.PythonCallbackNode, reason: str, code: str, renamed: list, input_names: list,
                output_names: list, state: LoweringState) -> None:
    """Extend an adjacent callback node with another statement run, chaining
    I/O (names the earlier run produced are not inputs of the merged run) and
    rebuilding the outlined scaffolding under the same callback name."""
    merged_inputs = list(previous.input_names)
    merged_inputs.extend(name for name in input_names
                         if name not in previous.output_names and name not in merged_inputs)
    merged_outputs = list(previous.output_names)
    merged_outputs.extend(name for name in output_names if name not in merged_outputs)

    previous_renamed = _outlined_body(previous)
    _register_free_globals(renamed, set(merged_inputs) | set(merged_outputs), state)
    previous.code = CodeBlock(f'{previous.code.as_string}\n{code}')
    previous.reason = '; '.join(dict.fromkeys([previous.reason, reason]))
    previous.input_names = merged_inputs
    previous.output_names = merged_outputs
    previous.outlined_function_code, previous.outlined_call_code = _outline(previous_renamed + renamed,
                                                                            previous.outlined_function_name,
                                                                            merged_inputs,
                                                                            merged_outputs,
                                                                            state,
                                                                            register=False)


def _outlined_body(node: tn.PythonCallbackNode) -> list:
    """Recover the repository-renamed statement run from a callback node's
    outlined function (dropping the synthesized trailing return)."""
    function_def = ast.parse(node.outlined_function_code.as_string).body[0]
    body = list(function_def.body)
    if body and isinstance(body[-1], ast.Return):
        body.pop()
    if body == [ast.Pass()] or (len(body) == 1 and isinstance(body[0], ast.Pass)):
        return []
    return body


def _outline(renamed: list,
             function_name: str,
             input_names: list,
             output_names: list,
             state: LoweringState,
             register: bool = True):
    """
    Build the outlined callback scaffolding (a function definition over the
    repository-named inputs and a call statement binding the outputs), and
    optionally register the callback name in the tree's callback mapping.

    The scaffolding references *repository* names so the tree-to-SDFG lowering
    can connect it directly; the node's ``code`` field keeps the source-level
    statement text.
    """
    from dace.frontend.python.nextgen.lowering.outliner import CallbackOutliner
    function_code, call_code = CallbackOutliner.outline(renamed,
                                                        callback_name=function_name,
                                                        input_names=input_names,
                                                        output_names=output_names)
    if register:
        state.emitter.root.callback_mapping[function_name] = function_name
    return function_code, call_code


def _static_expansions(statement: OpaqueStmt, state: LoweringState) -> dict:
    """
    The compile-time sequences among a statement's inputs, as the expressions
    that produced them.

    ``callback(a, [a, b])`` is A-normalized to ``__anf0 = [a, b]`` followed by
    ``callback(a, __anf0)``, and a list of arrays binds as a compile-time
    sequence rather than a container. There is consequently nothing to pass the
    interpreter under the name ``__anf0``, and the callback fails with
    ``NameError`` -- inside the callback, where it cannot propagate. Rebuilding
    the literal restores what the user wrote, and its elements are containers
    the callback can genuinely receive.

    :return: Source name -> the expression to put in its place.
    """
    expansions = {}
    for name in sorted(statement.inputs):
        binding = state.context.resolve(name)
        if binding is None or binding.kind != 'static':
            continue
        expression = _sequence_expression(state.context.static_values.get(name))
        if expression is not None:
            expansions[name] = expression
    return expansions


def _sequence_expression(sequence) -> Optional[ast.expr]:
    """The list/tuple display a compile-time sequence denotes, or None if its
    elements are not expressions this can rebuild."""
    elements = getattr(sequence, 'elements', None)
    if not elements:
        return None
    rebuilt = []
    for element in elements:
        if isinstance(element, ast.expr):
            rebuilt.append(copy.deepcopy(element))
        elif isinstance(element, (int, float, complex, bool, str)):
            rebuilt.append(ast.Constant(value=element))
        else:
            return None
    display = ast.Tuple if getattr(sequence, 'kind', 'list') == 'tuple' else ast.List
    return ast.fix_missing_locations(display(elts=rebuilt, ctx=ast.Load()))


def _referenced_names(expression: ast.expr) -> set:
    """The names an expression reads."""
    return {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}


def _substitute_names(statement: ast.stmt, expansions: dict) -> ast.stmt:
    """Replace each name in ``expansions`` with the expression it stands for."""

    class _Substituter(ast.NodeTransformer):

        def visit_Name(self, node: ast.Name) -> ast.expr:
            if isinstance(node.ctx, ast.Load) and node.id in expansions:
                return ast.copy_location(copy.deepcopy(expansions[node.id]), node)
            return node

    return ast.fix_missing_locations(_Substituter().visit(statement))


def _declared_descriptor(name: str, statement: OpaqueStmt, state: LoweringState) -> Optional[data.Data]:
    """
    The descriptor an annotation declares for an output of the opaque
    statement, or None if nothing types it.

    Two spellings elucidate a callback result, and both are offered by the
    error raised when neither is present (see
    :mod:`~dace.frontend.python.nextgen.lowering.mechanisms.opaque_values`):
    an annotation on the assignment itself (``a: dace.float64[20] =
    callee(b)``), and a return type hint on the callee (``def callee(b) ->
    dace.float64[20]``). The first is read off the statement; the second off
    the resolved callee object.
    """
    from dace.frontend.python.nextgen.lowering.rules.assign import annotation_descriptor
    for original in statement.originals:
        for node in ast.walk(original):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == name):
                continue
            descriptor = annotation_descriptor(getattr(node, 'annotation', None), state)
            if descriptor is not None:
                return descriptor
            descriptor = _return_hint_descriptor(node.value, state)
            if descriptor is not None:
                return descriptor
    return None


def _return_hint_descriptor(value: ast.expr, state: LoweringState) -> Optional[data.Data]:
    """
    The descriptor declared by the return type hint of the function ``value``
    calls, or None if it is not a call, the callee does not resolve, or it
    carries no usable hint.
    """
    if not isinstance(value, ast.Call):
        return None
    try:
        _, callee = state.inference.resolve_callee(value.func)
    except Exception:
        return None
    hint = getattr(callee, '__annotations__', {}).get('return') if callee is not None else None
    if hint is None:
        return None
    try:
        descriptor = data.create_datadescriptor(hint)
    except Exception:
        return None
    if not isinstance(descriptor, data.Data) or descriptor.dtype is None or isinstance(
            descriptor.dtype, dtypes.pyobject):
        return None
    return copy.deepcopy(descriptor)


def _callee_name(statement: OpaqueStmt) -> Optional[str]:
    """The name of the function the opaque statement calls, when it makes a
    single call -- what the user has to decorate or register to get rid of the
    callback."""
    names = set()
    for original in statement.originals:
        for node in ast.walk(original):
            if isinstance(node, ast.Call):
                names.add(getattr(node.func, 'qualname', None) or astutils.rname(node.func))
    return next(iter(names)) if len(names) == 1 else None


def _warn_about_the_callback(statement: OpaqueStmt, state: LoweringState) -> None:
    """
    Report that a statement will run in the Python interpreter.

    Not a diagnostic detail: a callback crosses into the interpreter on every
    execution of the statement, serializes around it, and cannot propagate an
    exception back out through the C callback boundary. The classic frontend
    raises a syntax error for most of what reaches here and warns for the rest,
    so nothing it accepted degraded silently; this frontend lowers a callback
    instead of refusing, which makes the warning the only signal the user gets.
    """
    callee = _callee_name(statement)
    subject = f'from function "{callee}"' if callee else 'for a statement'
    line = getattr(statement, 'lineno', 0)
    warnings.warn(f'Performance warning: Automatically creating callback to the Python interpreter '
                  f'{subject}\n  in File "{state.context.filename}", line {line}\n'
                  f'  reason: {statement.reason}\n'
                  'To lower it natively, place a @dace.program decorator on the callee, or register a '
                  'replacement through "dace.frontend.common.op_repository".')


def _report_untyped_result(name: str, statement: OpaqueStmt, state: LoweringState) -> None:
    """
    Report a callback result no annotation types, which therefore travels as an
    opaque Python object.

    :raises DaceSyntaxError: If ``frontend.typed_callbacks_only`` is set, the
                             configured refusal to accept exactly this.
    """
    callee = _callee_name(statement)
    subject = f'of function call "{callee}"' if callee else f'of the value assigned to "{name}"'
    line = getattr(statement, 'lineno', 0)
    message = (f'Cannot infer return type {subject}:\n'
               f'  in File "{state.context.filename}", line {line}\n'
               'To ensure that the return types can be inferred, try to extract the call to a '
               'separate statement and annotate the return values. For example: '
               'a: dace.int32 = call(b, c).\n'
               'To enforce only callbacks with explicit return types, set the '
               '`frontend.typed_callbacks_only` configuration entry to True.')
    if Config.get_bool('frontend', 'typed_callbacks_only'):
        raise DaceSyntaxError(None, statement, message)
    warnings.warn(message)


def _register_free_globals(statements: list, bound: set, state: LoweringState) -> None:
    """
    Register the values of free global names referenced by a callback run as
    named program constants, making the tree self-contained: callback code
    executes with a namespace built from ``root.constants`` (detected
    callables, modules, and other objects with no dataflow representation).
    Opaque (pyobject-typed) constants never reach generated code — they exist
    only for the callback execution namespace.
    """
    assigned = {
        node.id
        for statement in statements
        for node in ast.walk(statement) if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }
    for statement in statements:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            name = node.id
            if (name in bound or name in assigned or name in state.context.constants or name in state.context.containers
                    or name in state.context.symbols or hasattr(builtins, name)):
                continue
            if name in state.context.globals:
                value = state.context.globals[name]
            elif name in state.context.callback_callables:
                # A callback preprocessing detected: the call was rewritten to
                # a sanitized name that is a global of no scope, so the
                # callable only exists in the closure it came from.
                value = state.context.callback_callables[name]
            else:
                continue
            # Always opaque: these constants exist only for the callback
            # execution namespace and never reach generated code (typed
            # constants of supported code go through closure_constants).
            state.context.constants[name] = (data.Scalar(dtypes.pyobject()), value)


def _constant_reference(value: object, state: LoweringState) -> str:
    """The program-constant name of an embedded object, registering it under a
    fresh name if it is not a known constant yet."""
    for name, (_, existing) in state.context.constants.items():
        if existing is value:
            return name
    name = state.context.fresh_name('__nextgen_object')
    state.context.constants[name] = (data.Scalar(dtypes.pyobject()), value)
    return name


def _rename_to_repository(statements: list, source_to_repository: dict) -> list:
    """Copies of the statements with source names replaced by their repository
    container names."""

    class _Renamer(ast.NodeTransformer):

        def visit_Name(self, node: ast.Name) -> ast.Name:
            node.id = source_to_repository.get(node.id, node.id)
            return node

    return [ast.fix_missing_locations(_Renamer().visit(astutils.copy_tree(statement))) for statement in statements]


def _reconstitute_source(statement: ast.stmt, state: LoweringState) -> ast.stmt:
    """
    Return a copy of a statement in which canonicalization/preprocessing
    artifacts are restored to interpreter-executable source form:

    - objects embedded as constants by preprocessing's global resolution
      (dace programs, SDFGs, modules, ...) become their source-level
      qualified names again,
    - nested :class:`OpaqueStmt` markers (inside a rolled-back compound
      statement) are replaced by their original statements,
    - nested :class:`ExplicitTasklet` markers are restored to their original
      statement (interpreter-executable ``with dace.tasklet:`` blocks), or a
      synthesized equivalent with-block when no original exists.
    """
    from dace.frontend.python.nextgen.canonical.cpa import ExplicitTasklet, OpaqueStmt
    from dace.frontend.python.nextgen.semantics.inference import is_literal_constant

    class _Restorer(ast.NodeTransformer):

        def visit_Call(self, node: ast.Call) -> ast.expr:
            # Detected callables in callee position carry the *full call
            # expression* as their qualified name; restore only the callee.
            restored = self._restore_constant(node.func)
            if restored is not None:
                if isinstance(restored, ast.Call):
                    restored = restored.func
                node.func = ast.copy_location(restored, node.func)
            return self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> ast.expr:
            restored = self._restore_constant(node)
            if restored is None:
                return node
            return ast.copy_location(restored, node)

        def _restore_constant(self, node: ast.expr) -> Optional[ast.expr]:
            """The source expression of an embedded resolved object, or None
            for plain literals and non-constant expressions."""
            if not isinstance(node, ast.Constant) or is_literal_constant(node.value):
                return None
            qualname = getattr(node, 'qualname', None)
            if qualname is not None:
                try:
                    return ast.parse(qualname, mode='eval').body
                except SyntaxError:
                    pass
            # No parseable source form (e.g. a repr-derived name for a
            # resolved object attribute): bind the object as a named program
            # constant and reference it by that name.
            return ast.Name(id=_constant_reference(node.value, state), ctx=ast.Load())

        def visit_OpaqueStmt(self, node: OpaqueStmt) -> ast.stmt:
            return self.visit(node.original)

        def visit_ExplicitTasklet(self, node: ExplicitTasklet) -> ast.stmt:
            # A tasklet inside a rolled-back region replays through the
            # interpreter: restore the original statement (``with
            # dace.tasklet:`` blocks are interpreter-executable), or
            # synthesize an equivalent with-block for markers without a
            # standalone source form (desugared ``@dace.map`` bodies).
            if node.original is not None:
                return self.visit(node.original)
            context = ast.Attribute(value=ast.Name(id='dace', ctx=ast.Load()), attr='tasklet', ctx=ast.Load())
            with_block = ast.With(items=[ast.withitem(context_expr=context, optional_vars=None)],
                                  body=[self.visit(child) for child in node.statements])
            return ast.copy_location(with_block, node)

        def visit_ExplicitConsume(self, node) -> ast.stmt:
            # Consume markers always carry their original decorated
            # FunctionDef (the desugar deep-copies it before transforming the
            # body).
            return self.visit(node.original)

    return ast.fix_missing_locations(_Restorer().visit(astutils.copy_tree(statement)))
