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
import copy
from typing import Optional

from dace import data, dtypes
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.canonical.cpa import OpaqueStmt
from dace.frontend.python.nextgen.common import FrontendError
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule


@rule(OpaqueStmt)
def lower_opaque(statement: OpaqueStmt, state: LoweringState) -> None:
    source_to_repository: dict = {}
    input_names = []
    for name in sorted(statement.inputs):
        binding = state.context.resolve(name)
        if binding is not None and binding.kind == 'container':
            input_names.append(binding.container)
            source_to_repository[name] = binding.container

    output_names = []
    for name in sorted(statement.outputs):
        binding = state.context.resolve(name)
        if binding is not None and binding.kind == 'container':
            output_names.append(binding.container)
        else:
            # The callback produces a Python object we cannot type: register an
            # opaque scalar so later statements can bind and pass it through.
            container_name = state.context.add_container(name, data.Scalar(dtypes.pyobject()))
            state.context.bind(name, container_name)
            output_names.append(container_name)
        source_to_repository[name] = output_names[-1]

    reconstituted = [_reconstitute_source(original, state) for original in statement.originals]
    code = '\n'.join(ast.unparse(original) for original in reconstituted)
    renamed = _rename_to_repository(reconstituted, source_to_repository)

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
    # Reuses the staged frontend's outliner, which is builder-independent.
    from dace.frontend.python.schedule_tree.callback_support import CallbackOutliner
    function_code, call_code = CallbackOutliner.outline(renamed,
                                                        callback_name=function_name,
                                                        input_names=input_names,
                                                        output_names=output_names)
    if register:
        state.emitter.root.callback_mapping[function_name] = function_name
    return function_code, call_code


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

    return [ast.fix_missing_locations(_Renamer().visit(copy.deepcopy(statement))) for statement in statements]


def _reconstitute_source(statement: ast.stmt, state: LoweringState) -> ast.stmt:
    """
    Return a copy of a statement in which canonicalization/preprocessing
    artifacts are restored to interpreter-executable source form:

    - objects embedded as constants by preprocessing's global resolution
      (dace programs, SDFGs, modules, ...) become their source-level
      qualified names again,
    - nested :class:`OpaqueStmt` markers (inside a rolled-back compound
      statement) are replaced by their original statements,
    - nested :class:`ExplicitTasklet` markers cannot be reconstructed (their
      with-block form was consumed during canonicalization) and raise.
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
            raise FrontendError(
                'Cannot re-lower an explicit dataflow tasklet through the Python interpreter '
                '(e.g., inside a control-flow construct that fell back to a callback)', state.context.filename, node)

    return ast.fix_missing_locations(_Restorer().visit(copy.deepcopy(statement)))
