# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""AST desugaring passes for the direct Python schedule-tree frontend."""

from __future__ import annotations

import ast
import copy
from typing import Any, Dict, List, Optional, Tuple

from dace.frontend.python.common import DaceSyntaxError
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value

_CALLBACK_REASON_ATTR = '_schedule_tree_callback_reason'


class _DynamicExpansionError(Exception):

    def __init__(self, node: ast.Call, *, is_sdfg_call: bool) -> None:
        self.node = node
        self.is_sdfg_call = is_sdfg_call


class ScheduleTreeExpansionDesugarer(ast.NodeTransformer):
    """Rewrites compile-time-expandable ``*args`` / ``**kwargs`` and starred unpacking.

    Static cases are rewritten into ordinary Python AST. Dynamic cases are marked
    for callback lowering, except SDFG-backed calls, which raise ``DaceSyntaxError``.
    """

    def __init__(self,
                 filename: str,
                 global_vars: Dict[str, Any],
                 callable_bindings: Optional[Dict[str, Any]] = None) -> None:
        self.filename = filename
        self.global_vars = copy.copy(global_vars)
        self.callable_bindings = dict(callable_bindings or {})
        self._expansion_bindings: Dict[str, ast.AST] = {}
        self._temp_counter = 0

    def visit_Module(self, node: ast.Module) -> ast.AST:
        saved = self._expansion_bindings
        self._expansion_bindings = {}
        node.body = self._rewrite_body(node.body)
        self._expansion_bindings = saved
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        saved = self._expansion_bindings
        self._expansion_bindings = {}
        node.body = self._rewrite_body(node.body)
        self._expansion_bindings = saved
        return node

    if hasattr(ast, 'AsyncFunctionDef'):

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            saved = self._expansion_bindings
            self._expansion_bindings = {}
            node.body = self._rewrite_body(node.body)
            self._expansion_bindings = saved
            return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        try:
            node.value = self._rewrite_expression(node.value)
        except _DynamicExpansionError as ex:
            self._invalidate_targets(node.targets)
            if ex.is_sdfg_call:
                self._raise_dynamic_sdfg_error(ex.node)
            return self._mark_callback(node, 'call expansion')

        if len(node.targets) == 1 and self._has_starred_target(node.targets[0]):
            expanded = self._expand_starred_assignment(node.targets[0], node.value)
            if expanded is None:
                self._invalidate_targets(node.targets)
                return self._mark_callback(node, 'starred unpacking')
            return self._rewrite_generated_assignments(expanded, template_node=node)

        for target in node.targets:
            self._update_target_binding(target, node.value)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if node.value is None:
            self._invalidate_target(node.target)
            return node

        try:
            node.value = self._rewrite_expression(node.value)
        except _DynamicExpansionError as ex:
            self._invalidate_target(node.target)
            if ex.is_sdfg_call:
                self._raise_dynamic_sdfg_error(ex.node)
            return self._mark_callback(node, 'call expansion')

        self._update_target_binding(node.target, node.value)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        try:
            node.value = self._rewrite_expression(node.value)
        except _DynamicExpansionError as ex:
            self._invalidate_target(node.target)
            if ex.is_sdfg_call:
                self._raise_dynamic_sdfg_error(ex.node)
            return self._mark_callback(node, 'call expansion')
        self._invalidate_target(node.target)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        try:
            node.value = self._rewrite_expression(node.value)
        except _DynamicExpansionError as ex:
            if ex.is_sdfg_call:
                self._raise_dynamic_sdfg_error(ex.node)
            return self._mark_callback(node, 'call expansion')
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if node.value is None:
            return node
        try:
            node.value = self._rewrite_expression(node.value)
        except _DynamicExpansionError as ex:
            if ex.is_sdfg_call:
                self._raise_dynamic_sdfg_error(ex.node)
            temp_name = self._fresh_name('__stree_retval')
            assign_stmt = ast.Assign(targets=[ast.Name(id=temp_name, ctx=ast.Store())], value=copy.deepcopy(node.value))
            assign_stmt = self._mark_callback(ast.copy_location(assign_stmt, node.value), 'call expansion')
            return_stmt = ast.copy_location(ast.Return(value=ast.Name(id=temp_name, ctx=ast.Load())), node)
            return [assign_stmt, ast.fix_missing_locations(return_stmt)]
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        try:
            node.test = self._rewrite_expression(node.test)
        except _DynamicExpansionError as ex:
            if ex.is_sdfg_call:
                self._raise_dynamic_sdfg_error(ex.node)
            return self._mark_callback(node, 'call expansion')
        node.body = self._rewrite_nested_body(node.body)
        node.orelse = self._rewrite_nested_body(node.orelse)
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        try:
            node.test = self._rewrite_expression(node.test)
        except _DynamicExpansionError as ex:
            if ex.is_sdfg_call:
                self._raise_dynamic_sdfg_error(ex.node)
            return self._mark_callback(node, 'call expansion')
        node.body = self._rewrite_nested_body(node.body)
        node.orelse = self._rewrite_nested_body(node.orelse)
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        try:
            node.iter = self._rewrite_expression(node.iter)
        except _DynamicExpansionError as ex:
            if ex.is_sdfg_call:
                self._raise_dynamic_sdfg_error(ex.node)
            return self._mark_callback(node, 'call expansion')
        node.body = self._rewrite_nested_body(node.body)
        node.orelse = self._rewrite_nested_body(node.orelse)
        self._invalidate_target(node.target)
        return node

    if hasattr(ast, 'AsyncFor'):

        def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
            try:
                node.iter = self._rewrite_expression(node.iter)
            except _DynamicExpansionError as ex:
                if ex.is_sdfg_call:
                    self._raise_dynamic_sdfg_error(ex.node)
                return self._mark_callback(node, 'call expansion')
            node.body = self._rewrite_nested_body(node.body)
            node.orelse = self._rewrite_nested_body(node.orelse)
            self._invalidate_target(node.target)
            return node

    def visit_With(self, node: ast.With) -> ast.AST:
        for item in node.items:
            try:
                item.context_expr = self._rewrite_expression(item.context_expr)
                if item.optional_vars is not None:
                    self._invalidate_target(item.optional_vars)
            except _DynamicExpansionError as ex:
                if ex.is_sdfg_call:
                    self._raise_dynamic_sdfg_error(ex.node)
                return self._mark_callback(node, 'call expansion')
        node.body = self._rewrite_nested_body(node.body)
        return node

    if hasattr(ast, 'AsyncWith'):

        def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
            for item in node.items:
                try:
                    item.context_expr = self._rewrite_expression(item.context_expr)
                    if item.optional_vars is not None:
                        self._invalidate_target(item.optional_vars)
                except _DynamicExpansionError as ex:
                    if ex.is_sdfg_call:
                        self._raise_dynamic_sdfg_error(ex.node)
                    return self._mark_callback(node, 'call expansion')
            node.body = self._rewrite_nested_body(node.body)
            return node

    def visit_Try(self, node: ast.Try) -> ast.AST:
        node.body = self._rewrite_nested_body(node.body)
        node.orelse = self._rewrite_nested_body(node.orelse)
        node.finalbody = self._rewrite_nested_body(node.finalbody)
        for handler in node.handlers:
            handler.body = self._rewrite_nested_body(handler.body)
        return node

    def _rewrite_body(self, body: List[ast.stmt]) -> List[ast.stmt]:
        result: List[ast.stmt] = []
        for statement in body:
            rewritten = self.visit(statement)
            if rewritten is None:
                continue
            if isinstance(rewritten, list):
                result.extend(rewritten)
            else:
                result.append(rewritten)
        return result

    def _rewrite_nested_body(self, body: List[ast.stmt]) -> List[ast.stmt]:
        saved = self._expansion_bindings
        self._expansion_bindings = copy.deepcopy(saved)
        rewritten = self._rewrite_body(body)
        self._expansion_bindings = saved
        return rewritten

    def _rewrite_generated_assignments(self, assignments: List[Tuple[ast.AST, ast.AST]],
                                       template_node: ast.stmt) -> List[ast.stmt]:
        result: List[ast.stmt] = []
        for target, value in assignments:
            assign_stmt = ast.Assign(targets=[copy.deepcopy(target)], value=copy.deepcopy(value))
            assign_stmt = ast.copy_location(assign_stmt, template_node)
            rewritten = self.visit(assign_stmt)
            if rewritten is None:
                continue
            if isinstance(rewritten, list):
                result.extend(rewritten)
            else:
                result.append(rewritten)
        return result

    def _rewrite_expression(self, node: ast.AST) -> ast.AST:
        outer = self

        class _ExpressionRewriter(ast.NodeTransformer):

            def visit_Call(self, call_node: ast.Call) -> ast.AST:
                call_node = self.generic_visit(call_node)
                if not outer._is_expanded_call(call_node):
                    return call_node
                expanded = outer._expand_call_if_static(call_node)
                if expanded is not None:
                    return expanded
                raise _DynamicExpansionError(call_node, is_sdfg_call=outer._is_sdfg_call(call_node))

        return _ExpressionRewriter().visit(copy.deepcopy(node))

    def _evaluation_context(self) -> Dict[str, Any]:
        context = copy.copy(self.global_vars)
        context.update(self.callable_bindings)
        return context

    def _resolve_callable_value(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            if node.id in self.callable_bindings:
                return self.callable_bindings[node.id]
            if node.id in self.global_vars:
                return self.global_vars[node.id]
        return try_resolve_static_value(node, self._evaluation_context())

    def _is_sdfg_call(self, node: ast.Call) -> bool:
        from dace import SDFG

        callee = self._resolve_callable_value(node.func)
        return callee is not UNRESOLVED and not hasattr(callee, '__schedule_tree__') and (isinstance(callee, SDFG) or
                                                                                          hasattr(callee, '__sdfg__'))

    @staticmethod
    def _is_expanded_call(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and (any(isinstance(arg, ast.Starred)
                                                   for arg in node.args) or any(keyword.arg is None
                                                                                for keyword in node.keywords))

    @staticmethod
    def _has_starred_target(target: ast.AST) -> bool:
        return any(isinstance(child, ast.Starred) for child in ast.walk(target))

    def _expand_call_if_static(self, node: ast.Call) -> Optional[ast.Call]:
        args: List[ast.AST] = []
        keywords: List[ast.keyword] = []
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                expanded = self._resolve_static_sequence_nodes(argument.value)
                if expanded is None:
                    return None
                args.extend(copy.deepcopy(value) for value in expanded)
            else:
                args.append(copy.deepcopy(argument))

        for keyword in node.keywords:
            if keyword.arg is None:
                expanded_items = self._resolve_static_mapping_items(keyword.value)
                if expanded_items is None:
                    return None
                keywords.extend(ast.keyword(arg=name, value=copy.deepcopy(value)) for name, value in expanded_items)
            else:
                keywords.append(ast.keyword(arg=keyword.arg, value=copy.deepcopy(keyword.value)))
        return ast.copy_location(ast.Call(func=copy.deepcopy(node.func), args=args, keywords=keywords), node)

    def _normalized_static_expansion_ast(self, node: ast.AST) -> Optional[ast.AST]:
        if isinstance(node, ast.Name):
            cached = self._expansion_bindings.get(node.id)
            return copy.deepcopy(cached) if cached is not None else None

        if isinstance(node, (ast.Tuple, ast.List)):
            elements: List[ast.AST] = []
            for element in node.elts:
                if isinstance(element, ast.Starred):
                    expanded = self._resolve_static_sequence_nodes(element.value)
                    if expanded is None:
                        return None
                    elements.extend(copy.deepcopy(value) for value in expanded)
                else:
                    elements.append(copy.deepcopy(element))
            sequence_type = ast.Tuple if isinstance(node, ast.Tuple) else ast.List
            return ast.copy_location(sequence_type(elts=elements, ctx=ast.Load()), node)

        if isinstance(node, ast.Dict):
            keys: List[Optional[ast.AST]] = []
            values: List[ast.AST] = []
            for key, value in zip(node.keys, node.values):
                if key is None:
                    expanded_items = self._resolve_static_mapping_items(value)
                    if expanded_items is None:
                        return None
                    for expanded_key, expanded_value in expanded_items:
                        keys.append(ast.copy_location(ast.Constant(expanded_key), value))
                        values.append(copy.deepcopy(expanded_value))
                    continue

                resolved_key = try_resolve_static_value(key, self._evaluation_context())
                if not isinstance(resolved_key, str):
                    return None
                keys.append(ast.copy_location(ast.Constant(resolved_key), key))
                values.append(copy.deepcopy(value))
            return ast.copy_location(ast.Dict(keys=keys, values=values), node)

        return None

    def _resolve_static_sequence_nodes(self, node: ast.AST) -> Optional[List[ast.AST]]:
        normalized = self._normalized_static_expansion_ast(node)
        if isinstance(normalized, (ast.Tuple, ast.List)):
            return [copy.deepcopy(element) for element in normalized.elts]
        return None

    def _resolve_static_mapping_items(self, node: ast.AST) -> Optional[List[Tuple[str, ast.AST]]]:
        normalized = self._normalized_static_expansion_ast(node)
        if not isinstance(normalized, ast.Dict):
            return None

        result: List[Tuple[str, ast.AST]] = []
        for key, value in zip(normalized.keys, normalized.values):
            resolved_key = try_resolve_static_value(key, self._evaluation_context())
            if not isinstance(resolved_key, str):
                return None
            result.append((resolved_key, copy.deepcopy(value)))
        return result

    def _expand_starred_assignment(self, target: ast.AST, value: ast.AST) -> Optional[List[Tuple[ast.AST, ast.AST]]]:
        if isinstance(target, ast.Name):
            return [(copy.deepcopy(target), copy.deepcopy(value))]

        if isinstance(target, ast.Starred):
            elements = self._resolve_static_sequence_nodes(value)
            if elements is None:
                return None
            list_value = ast.copy_location(
                ast.List(elts=[copy.deepcopy(element) for element in elements], ctx=ast.Load()), value)
            return [(copy.deepcopy(target.value), list_value)]

        if not isinstance(target, (ast.Tuple, ast.List)):
            return [(copy.deepcopy(target), copy.deepcopy(value))]

        elements = self._resolve_static_sequence_nodes(value)
        if elements is None:
            return None

        starred_indices = [index for index, element in enumerate(target.elts) if isinstance(element, ast.Starred)]
        if len(starred_indices) > 1:
            return None

        assignments: List[Tuple[ast.AST, ast.AST]] = []
        if not starred_indices:
            if len(target.elts) != len(elements):
                return None
            for subtarget, subvalue in zip(target.elts, elements):
                expanded = self._expand_starred_assignment(subtarget, subvalue)
                if expanded is None:
                    return None
                assignments.extend(expanded)
            return assignments

        starred_index = starred_indices[0]
        if len(elements) < len(target.elts) - 1:
            return None

        prefix_targets = target.elts[:starred_index]
        suffix_targets = target.elts[starred_index + 1:]
        prefix_values = elements[:starred_index]
        suffix_values = elements[len(elements) - len(suffix_targets):]
        middle_values = elements[starred_index:len(elements) - len(suffix_targets)]

        for subtarget, subvalue in zip(prefix_targets, prefix_values):
            expanded = self._expand_starred_assignment(subtarget, subvalue)
            if expanded is None:
                return None
            assignments.extend(expanded)

        middle_list = ast.copy_location(
            ast.List(elts=[copy.deepcopy(element) for element in middle_values], ctx=ast.Load()), value)
        expanded_middle = self._expand_starred_assignment(target.elts[starred_index], middle_list)
        if expanded_middle is None:
            return None
        assignments.extend(expanded_middle)

        for subtarget, subvalue in zip(suffix_targets, suffix_values):
            expanded = self._expand_starred_assignment(subtarget, subvalue)
            if expanded is None:
                return None
            assignments.extend(expanded)

        return assignments

    def _update_target_binding(self, target: ast.AST, value: ast.AST) -> None:
        normalized = self._normalized_static_expansion_ast(value)
        if isinstance(target, ast.Name) and normalized is not None:
            self._expansion_bindings[target.id] = normalized
            return
        self._invalidate_target(target)

    def _invalidate_targets(self, targets: List[ast.AST]) -> None:
        for target in targets:
            self._invalidate_target(target)

    def _invalidate_target(self, target: ast.AST) -> None:
        for child in ast.walk(target):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                self._expansion_bindings.pop(child.id, None)

    def _fresh_name(self, prefix: str) -> str:
        candidate = prefix
        while candidate in self._expansion_bindings or candidate in self.global_vars or candidate in self.callable_bindings:
            self._temp_counter += 1
            candidate = f'{prefix}{self._temp_counter}'
        return candidate

    def _mark_callback(self, statement: ast.stmt, reason: str) -> ast.stmt:
        setattr(statement, _CALLBACK_REASON_ATTR, reason)
        return ast.fix_missing_locations(statement)

    def _raise_dynamic_sdfg_error(self, node: ast.Call) -> None:
        raise DaceSyntaxError(self, node, 'Dynamic argument expansion is unsupported for SDFG calls in '
                              'schedule-tree lowering')


def desugar_schedule_tree_expansions(parsed_ast: ast.AST,
                                     *,
                                     filename: str,
                                     global_vars: Dict[str, Any],
                                     callable_bindings: Optional[Dict[str, Any]] = None) -> ast.AST:
    """Rewrite schedule-tree-specific argument expansion before AST lowering."""
    desugarer = ScheduleTreeExpansionDesugarer(filename, global_vars, callable_bindings=callable_bindings)
    return ast.fix_missing_locations(desugarer.visit(copy.deepcopy(parsed_ast)))


def callback_reason(node: ast.AST) -> Optional[str]:
    """Return the callback reason attached by schedule-tree desugaring, if any."""
    return getattr(node, _CALLBACK_REASON_ATTR, None)
