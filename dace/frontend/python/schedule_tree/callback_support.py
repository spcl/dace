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
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from dace import data
from dace.properties import CodeBlock

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from dace.frontend.python import astutils
from dace.frontend.python.schedule_tree.callable_support import CallableResolver
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value

CallbackBody = Union[ast.AST, Sequence[ast.stmt]]


class CallbackOutliner:
    """Build callback scaffolding and basic name-flow metadata.

    The helper accepts either a single AST node or a list of statements. This
    lets the current frontend keep wrapping individual callback statements while
    also providing an API that can later outline larger statement groups.
    """

    @staticmethod
    def analyze_name_flow(body: CallbackBody) -> Tuple[set[str], set[str]]:
        """Return ``(load_names, store_names)`` for a callback body."""
        inputs: set[str] = set()
        outputs: set[str] = set()
        for node in CallbackOutliner._body_nodes(body):
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    if isinstance(child.ctx, ast.Store):
                        outputs.add(child.id)
                    elif isinstance(child.ctx, ast.Load):
                        inputs.add(child.id)
                elif isinstance(child, ast.alias):
                    if child.asname:
                        outputs.add(child.asname)
                    else:
                        outputs.add(child.name.split('.')[0])
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    outputs.add(child.name)
                elif isinstance(child, ast.ExceptHandler) and isinstance(child.name, str):
                    outputs.add(child.name)
        return inputs, outputs

    @staticmethod
    def code_block(body: CallbackBody) -> CodeBlock:
        """Return a ``CodeBlock`` for the original callback body."""
        return CodeBlock(CallbackOutliner._body_nodes(body))

    @staticmethod
    def outline(body: CallbackBody, *, callback_name: str, input_names: Sequence[str],
                output_names: Sequence[str]) -> Tuple[CodeBlock, CodeBlock]:
        """Build outlined function and call-site scaffolding for ``body``."""
        input_names = list(input_names)
        output_names = list(output_names)
        function_body = CallbackOutliner._body_nodes(body)
        if output_names:
            returned = ast.Name(id=output_names[0], ctx=ast.Load())
            if len(output_names) > 1:
                returned = ast.Tuple(elts=[ast.Name(id=name, ctx=ast.Load()) for name in output_names], ctx=ast.Load())
            function_body.append(ast.Return(value=returned))

        function_def = ast.FunctionDef(name=callback_name,
                                       args=ast.arguments(posonlyargs=[],
                                                          args=[ast.arg(arg=name) for name in input_names],
                                                          vararg=None,
                                                          kwonlyargs=[],
                                                          kw_defaults=[],
                                                          kwarg=None,
                                                          defaults=[]),
                                       body=function_body or [ast.Pass()],
                                       decorator_list=[])
        function_code = CodeBlock([ast.fix_missing_locations(function_def)])

        call_expr = ast.Call(func=ast.Name(id=callback_name, ctx=ast.Load()),
                             args=[ast.Name(id=name, ctx=ast.Load()) for name in input_names],
                             keywords=[])
        if not output_names:
            call_stmt: ast.stmt = ast.Expr(value=call_expr)
        elif len(output_names) == 1:
            call_stmt = ast.Assign(targets=[ast.Name(id=output_names[0], ctx=ast.Store())], value=call_expr)
        else:
            call_stmt = ast.Assign(targets=[
                ast.Tuple(elts=[ast.Name(id=name, ctx=ast.Store()) for name in output_names], ctx=ast.Store())
            ],
                                   value=call_expr)
        call_code = CodeBlock([ast.fix_missing_locations(call_stmt)])
        return function_code, call_code

    @staticmethod
    def _body_nodes(body: CallbackBody) -> List[ast.stmt]:
        if isinstance(body, Sequence) and not isinstance(body, ast.AST):
            return [astutils.copy_tree(statement) for statement in body]
        if isinstance(body, ast.stmt):
            return [astutils.copy_tree(body)]
        if isinstance(body, ast.AST):
            return [ast.Expr(value=astutils.copy_tree(body))]
        return [ast.Pass()]


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
