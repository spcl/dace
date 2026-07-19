# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Helpers for schedule-tree Python callback outlining.

Example:
    Wrapping ``it = iter(generator)`` as a callback can keep the original code
    block for compatibility while also producing an outlined scaffold such as::

        def __stree_callback_0():
            it = iter(generator)
            return it

        it = __stree_callback_0()

    The outlined scaffold is metadata for future callback lowering work; the
    schedule tree still preserves the original callback code text as well.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, Callable, Dict, Optional, Set, Tuple

from dace import data
from dace.properties import CodeBlock

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from dace.frontend.python import astutils
from dace.frontend.python.schedule_tree.callable_support import CallableResolver
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value

# The outliner's canonical implementation lives in the next-generation
# frontend; re-exported here for the legacy frontend until its removal.
from dace.frontend.python.nextgen.lowering.outliner import CallbackBody, CallbackOutliner  # noqa: F401


class CallbackHandler:
    """Own callback wrapping, callback assignments, and callback fallback policy."""

    def __init__(self, *, bindings: Dict[str, Any], callback_mutated_global_names: Set[str],
                 callable_resolver: CallableResolver, evaluation_context: Callable[[], Dict[str, Any]],
                 append_node: Callable[[tn.ScheduleTreeNode],
                                       None], register_binding: Callable[[str, data.Data, str],
                                                                         None], fresh_callback_name: Callable[[], str],
                 fresh_transient_name: Callable[[str], str], render_callback_code: Callable[[ast.AST], str],
                 collect_scope_declarations: Callable[[ast.AST],
                                                      Tuple[set[str],
                                                            set[str]]], raise_syntax_error: Callable[[ast.AST, str],
                                                                                                     None],
                 binding_kind_for_descriptor: Callable[[data.Data],
                                                       str], pyobject_scalar_descriptor: Callable[[], data.Scalar],
                 is_pyobject_scalar_descriptor: Callable[[Optional[data.Data]], bool],
                 is_iterator_protocol_call: Callable[[ast.AST], bool], is_iterator_next_call: Callable[[ast.AST],
                                                                                                       bool]) -> None:
        self.bindings = bindings
        self.callback_mutated_global_names = callback_mutated_global_names
        self.callable_resolver = callable_resolver
        self.evaluation_context = evaluation_context
        self.append_node = append_node
        self.register_binding = register_binding
        self.fresh_callback_name = fresh_callback_name
        self.fresh_transient_name = fresh_transient_name
        self.render_callback_code = render_callback_code
        self.collect_scope_declarations = collect_scope_declarations
        self.raise_syntax_error = raise_syntax_error
        self.binding_kind_for_descriptor = binding_kind_for_descriptor
        self.pyobject_scalar_descriptor = pyobject_scalar_descriptor
        self.is_pyobject_scalar_descriptor = is_pyobject_scalar_descriptor
        self.is_iterator_protocol_call = is_iterator_protocol_call
        self.is_iterator_next_call = is_iterator_next_call

    def reject_mutated_global_uses(self, node: Optional[ast.AST]) -> None:
        if node is None or not self.callback_mutated_global_names:
            return

        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id in self.callback_mutated_global_names:
                    self.raise_syntax_error(
                        child, 'Nested callback functions cannot reassign global names that are used in the enclosing '
                        f'program: {child.id}')

    def wrap_node(self, node: ast.AST, reason: str) -> None:
        node = ast.fix_missing_locations(astutils.copy_tree(node))
        try:
            code = CodeBlock(self.render_callback_code(node))
        except Exception:
            code = CodeBlock('pass')

        inputs, outputs = CallbackOutliner.analyze_name_flow(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            global_names, _ = self.collect_scope_declarations(node)
            self.callback_mutated_global_names.update(global_names)

        known_inputs = sorted(inputs & set(self.bindings))
        sorted_outputs = sorted(outputs)
        for output_name in sorted_outputs:
            binding = self.bindings.get(output_name)
            if binding is None or getattr(binding, 'descriptor', None) is None:
                self.register_binding(output_name, self.pyobject_scalar_descriptor(), 'scalar')

        callback_name = self.fresh_callback_name()
        outlined_function_code, outlined_call_code = CallbackOutliner.outline(node,
                                                                              callback_name=callback_name,
                                                                              input_names=known_inputs,
                                                                              output_names=sorted_outputs)
        self.append_node(
            tn.PythonCallbackNode(code=code,
                                  reason=reason,
                                  input_names=known_inputs,
                                  output_names=sorted_outputs,
                                  outlined_function_name=callback_name,
                                  outlined_function_code=outlined_function_code,
                                  outlined_call_code=outlined_call_code))

    def emit_assignment(self, name: str, value: ast.AST, reason: str, descriptor: data.Data) -> None:
        if reason == 'pyobject call' and self.is_pyobject_scalar_descriptor(descriptor) and self.is_iterator_next_call(
                value):
            import warnings
            warnings.warn('Could not infer the result type of iterator next() in schedule-tree lowering; '
                          'annotate the assignment target, e.g. val: dace.float64 = next(gen).')
        kind = self.binding_kind_for_descriptor(descriptor)
        self.register_binding(name, descriptor, kind)
        callback_assign = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=astutils.copy_tree(value))
        callback_assign = ast.copy_location(callback_assign, value)
        self.wrap_node(callback_assign, reason)

    def materialize_expression(self,
                               value: ast.AST,
                               reason: str,
                               descriptor: data.Data,
                               *,
                               prefix: str = '__stree_tmp') -> ast.AST:
        name = self.fresh_transient_name(prefix)
        self.emit_assignment(name, value, reason, descriptor)
        return ast.Name(id=name, ctx=ast.Load())

    def should_emit_pyobject_call_callback(self, value: ast.AST) -> bool:
        if not isinstance(value, ast.Call):
            return False
        if self.is_iterator_protocol_call(value):
            return True

        callee = self.callable_resolver.resolve_callable_value(value.func)
        if callee is not UNRESOLVED and inspect.isgeneratorfunction(callee):
            return True

        runtime_value = try_resolve_static_value(value, self.evaluation_context())
        if runtime_value is UNRESOLVED:
            return False
        if callable(runtime_value):
            return False
        return hasattr(runtime_value, '__next__') or hasattr(runtime_value, '__iter__')
