# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""AST desugaring passes for the direct Python schedule-tree frontend."""

from __future__ import annotations

import ast
import builtins as pybuiltins
import copy
import numbers
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dace import data
from dace.frontend.python.common import DaceSyntaxError
from dace.frontend.python.schedule_tree.callable_support import CallableResolver
from dace.frontend.python.schedule_tree.dunder_support import (rewrite_augassign, rewrite_subscript_assignment,
                                                               rewrite_subscript_delete, rewrite_sugared_expression)
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value

_CALLBACK_REASON_ATTR = '_schedule_tree_callback_reason'


class _DynamicExpansionError(Exception):

    def __init__(self, node: ast.Call, *, is_sdfg_call: bool) -> None:
        self.node = node
        self.is_sdfg_call = is_sdfg_call


class ScheduleTreeExpansionDesugarer(ast.NodeTransformer):
    """Rewrite schedule-tree-specific syntax into simpler AST forms.

    The pass handles compile-time-expandable ``*args`` / ``**kwargs``, starred
    unpacking, and tuple or list assignments that benefit from an explicit
    right-hand-side temporary.

    Example:
        ``A, B = B, A`` becomes ``__stree_tuple_tmp = (B, A)`` followed by
        ``(A, B) = __stree_tuple_tmp``.

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
        self.callable_resolver = CallableResolver(callable_bindings=self.callable_bindings,
                                                  evaluation_context=self._evaluation_context)
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

        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            rewritten = rewrite_subscript_assignment(node.targets[0], node.value, self.callable_resolver)
            if rewritten is not None:
                self._invalidate_targets(node.targets)
                return self.visit(rewritten)

        if len(node.targets) == 1 and self._has_starred_target(node.targets[0]):
            expanded = self._expand_starred_assignment(node.targets[0], node.value)
            if expanded is None:
                self._invalidate_targets(node.targets)
                return self._mark_callback(node, 'starred unpacking')
            return self._rewrite_generated_assignments(expanded, template_node=node)

        materialized = self._materialize_structured_assignment(node)
        if materialized is not None:
            return materialized

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
        rewritten = rewrite_augassign(node.target, node.op, node.value, self.callable_resolver)
        if rewritten is not None:
            self._invalidate_target(node.target)
            return self.visit(rewritten)

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

    def visit_Delete(self, node: ast.Delete) -> ast.AST:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Subscript):
            return node
        rewritten = rewrite_subscript_delete(node.targets[0], self.callable_resolver)
        if rewritten is None:
            return node
        self._invalidate_targets(node.targets)
        return self.visit(rewritten)

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

    def _rewrite_generated_statements(self, statements: List[ast.stmt]) -> List[ast.stmt]:
        result: List[ast.stmt] = []
        for statement in statements:
            rewritten = self.visit(statement)
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
                rewritten = rewrite_sugared_expression(call_node, outer.callable_resolver)
                if rewritten is not None:
                    call_node = self.visit(rewritten)
                if not outer._is_expanded_call(call_node):
                    return call_node
                expanded = outer._expand_call_if_static(call_node)
                if expanded is not None:
                    return expanded
                raise _DynamicExpansionError(call_node, is_sdfg_call=outer.callable_resolver.is_sdfg_call(call_node))

            def visit_BinOp(self, expr_node: ast.BinOp) -> ast.AST:
                expr_node = self.generic_visit(expr_node)
                rewritten = rewrite_sugared_expression(expr_node, outer.callable_resolver)
                return rewritten if rewritten is not None else expr_node

            def visit_UnaryOp(self, expr_node: ast.UnaryOp) -> ast.AST:
                expr_node = self.generic_visit(expr_node)
                rewritten = rewrite_sugared_expression(expr_node, outer.callable_resolver)
                return rewritten if rewritten is not None else expr_node

            def visit_Compare(self, expr_node: ast.Compare) -> ast.AST:
                expr_node = self.generic_visit(expr_node)
                rewritten = rewrite_sugared_expression(expr_node, outer.callable_resolver)
                return rewritten if rewritten is not None else expr_node

            def visit_Subscript(self, expr_node: ast.Subscript) -> ast.AST:
                expr_node = self.generic_visit(expr_node)
                rewritten = rewrite_sugared_expression(expr_node, outer.callable_resolver)
                return rewritten if rewritten is not None else expr_node

        return _ExpressionRewriter().visit(copy.deepcopy(node))

    def _evaluation_context(self) -> Dict[str, Any]:
        context = copy.copy(pybuiltins.__dict__)
        context.update(self.global_vars)
        context.update(self.callable_bindings)
        return context

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

    def _materialize_structured_assignment(self, node: ast.Assign) -> Optional[List[ast.stmt]]:
        if not any(isinstance(target, (ast.Tuple, ast.List)) for target in node.targets):
            return None

        if not isinstance(node.value, (ast.Tuple, ast.List)):
            return None

        normalized_value = self._normalized_static_expansion_ast(node.value)
        if not isinstance(normalized_value, (ast.Tuple, ast.List)):
            return None

        temp_name = self._fresh_name('__stree_tuple_tmp')
        temp_assign = ast.Assign(targets=[ast.Name(id=temp_name, ctx=ast.Store())], value=normalized_value)
        temp_assign = ast.copy_location(temp_assign, node)

        rewritten_assign = ast.Assign(targets=[copy.deepcopy(target) for target in node.targets],
                                      value=ast.Name(id=temp_name, ctx=ast.Load()))
        rewritten_assign = ast.copy_location(rewritten_assign, node)
        return self._rewrite_generated_statements([temp_assign, rewritten_assign])

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


class ScheduleTreeSubscriptIndexDesugarer(ast.NodeTransformer):
    """Outline nested subscript-index expressions into explicit temporaries.

    Examples:
        ``A[B[i]]`` becomes ``__stree_idx = B[i]`` followed by ``A[__stree_idx]``.

        ``A[f(g[i])]`` becomes a sequence such as ``__stree_idx = g[i]``,
        ``__stree_idx1 = f(__stree_idx)``, then ``A[__stree_idx1]``.
    """

    def __init__(self, global_vars: Dict[str, Any], callable_bindings: Optional[Dict[str, Any]] = None) -> None:
        self.global_vars = copy.copy(global_vars)
        self.callable_bindings = dict(callable_bindings or {})
        self._temp_counter = 0
        self._used_names: set[str] = set(self.global_vars) | set(self.callable_bindings)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self._seed_used_names(node)
        node.body = self._rewrite_body(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._seed_used_names(node)
        node.body = self._rewrite_body(node.body)
        return node

    if hasattr(ast, 'AsyncFunctionDef'):

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            self._seed_used_names(node)
            node.body = self._rewrite_body(node.body)
            return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        prologue, value = self._outline_expression(node.value)
        targets: List[ast.AST] = []
        for target in node.targets:
            outlined, rewritten = self._outline_expression(target)
            prologue.extend(outlined)
            targets.append(rewritten)
        node.targets = targets
        node.value = value
        return self._prepend_statements(node, prologue)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if node.value is None:
            return node
        prologue, value = self._outline_expression(node.value)
        target_prologue, target = self._outline_expression(node.target)
        node.value = value
        node.target = target
        return self._prepend_statements(node, prologue + target_prologue)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        prologue, target = self._outline_expression(node.target)
        value_prologue, value = self._outline_expression(node.value)
        node.target = target
        node.value = value
        return self._prepend_statements(node, prologue + value_prologue)

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if node.value is None:
            return node
        prologue, value = self._outline_expression(node.value)
        node.value = value
        return self._prepend_statements(node, prologue)

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        prologue, value = self._outline_expression(node.value)
        node.value = value
        return self._prepend_statements(node, prologue)

    def visit_If(self, node: ast.If) -> ast.AST:
        prologue, test = self._outline_expression(node.test)
        node.test = test
        node.body = self._rewrite_body(node.body)
        node.orelse = self._rewrite_body(node.orelse)
        return self._prepend_statements(node, prologue)

    def visit_While(self, node: ast.While) -> ast.AST:
        prologue, test = self._outline_expression(node.test)
        if prologue and node.orelse:
            return self._mark_callback(node, 'while loop test outlining with else')
        node.test = test
        node.body = self._rewrite_body(node.body)
        node.orelse = self._rewrite_body(node.orelse)
        if not prologue:
            return node

        guard = ast.If(test=ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(test)), body=[ast.Break()], orelse=[])
        guard = ast.fix_missing_locations(ast.copy_location(guard, node.test))
        rewritten = ast.While(test=ast.Constant(value=True), body=prologue + [guard] + node.body, orelse=[])
        rewritten = ast.fix_missing_locations(ast.copy_location(rewritten, node))
        return rewritten

    def visit_For(self, node: ast.For) -> ast.AST:
        prologue, iterator = self._outline_expression(node.iter)
        node.iter = iterator
        node.body = self._rewrite_body(node.body)
        node.orelse = self._rewrite_body(node.orelse)
        return self._prepend_statements(node, prologue)

    if hasattr(ast, 'AsyncFor'):

        def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
            prologue, iterator = self._outline_expression(node.iter)
            node.iter = iterator
            node.body = self._rewrite_body(node.body)
            node.orelse = self._rewrite_body(node.orelse)
            return self._prepend_statements(node, prologue)

    def visit_With(self, node: ast.With) -> ast.AST:
        prologue: List[ast.stmt] = []
        for item in node.items:
            item_prologue, context_expr = self._outline_expression(item.context_expr)
            prologue.extend(item_prologue)
            item.context_expr = context_expr
        node.body = self._rewrite_body(node.body)
        return self._prepend_statements(node, prologue)

    if hasattr(ast, 'AsyncWith'):

        def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
            prologue: List[ast.stmt] = []
            for item in node.items:
                item_prologue, context_expr = self._outline_expression(item.context_expr)
                prologue.extend(item_prologue)
                item.context_expr = context_expr
            node.body = self._rewrite_body(node.body)
            return self._prepend_statements(node, prologue)

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

    def _outline_expression(self,
                            node: Optional[ast.AST],
                            *,
                            in_index_context: bool = False,
                            hoist_safe: bool = True) -> Tuple[List[ast.stmt], Optional[ast.AST]]:
        if node is None:
            return [], None

        if isinstance(node, ast.Subscript):
            prologue, value = self._outline_expression(node.value,
                                                       in_index_context=in_index_context,
                                                       hoist_safe=hoist_safe)
            slice_prologue, slice_node = self._outline_subscript_slice(node.slice, hoist_safe=hoist_safe)
            rewritten = ast.copy_location(ast.Subscript(value=value, slice=slice_node, ctx=node.ctx), node)
            prologue.extend(slice_prologue)
            if hoist_safe and in_index_context and self._should_outline_index_expression(rewritten):
                return self._outline_to_temp(rewritten, prologue)
            return prologue, rewritten

        if isinstance(node, ast.Call):
            prologue, func = self._outline_expression(node.func,
                                                      in_index_context=in_index_context,
                                                      hoist_safe=hoist_safe)
            args: List[ast.AST] = []
            for arg in node.args:
                arg_prologue, rewritten_arg = self._outline_expression(arg,
                                                                       in_index_context=in_index_context,
                                                                       hoist_safe=hoist_safe)
                prologue.extend(arg_prologue)
                args.append(rewritten_arg)
            keywords: List[ast.keyword] = []
            for keyword in node.keywords:
                kw_prologue, rewritten_value = self._outline_expression(keyword.value,
                                                                        in_index_context=in_index_context,
                                                                        hoist_safe=hoist_safe)
                prologue.extend(kw_prologue)
                keywords.append(ast.keyword(arg=keyword.arg, value=rewritten_value))
            rewritten = ast.copy_location(ast.Call(func=func, args=args, keywords=keywords), node)
            if hoist_safe and in_index_context and self._should_outline_index_expression(rewritten):
                return self._outline_to_temp(rewritten, prologue)
            return prologue, rewritten

        if isinstance(node, ast.BinOp):
            prologue, left = self._outline_expression(node.left,
                                                      in_index_context=in_index_context,
                                                      hoist_safe=hoist_safe)
            right_prologue, right = self._outline_expression(node.right,
                                                             in_index_context=in_index_context,
                                                             hoist_safe=hoist_safe)
            prologue.extend(right_prologue)
            return prologue, ast.copy_location(ast.BinOp(left=left, op=copy.deepcopy(node.op), right=right), node)

        if isinstance(node, ast.UnaryOp):
            prologue, operand = self._outline_expression(node.operand,
                                                         in_index_context=in_index_context,
                                                         hoist_safe=hoist_safe)
            return prologue, ast.copy_location(ast.UnaryOp(op=copy.deepcopy(node.op), operand=operand), node)

        if isinstance(node, ast.BoolOp):
            prologue: List[ast.stmt] = []
            values: List[ast.AST] = []
            for index, value in enumerate(node.values):
                value_prologue, rewritten_value = self._outline_expression(value,
                                                                           in_index_context=in_index_context,
                                                                           hoist_safe=hoist_safe and index == 0)
                prologue.extend(value_prologue)
                values.append(rewritten_value)
            return prologue, ast.copy_location(ast.BoolOp(op=copy.deepcopy(node.op), values=values), node)

        if isinstance(node, ast.Compare):
            prologue, left = self._outline_expression(node.left,
                                                      in_index_context=in_index_context,
                                                      hoist_safe=hoist_safe)
            comparators: List[ast.AST] = []
            for index, comparator in enumerate(node.comparators):
                comparator_prologue, rewritten_comparator = self._outline_expression(comparator,
                                                                                     in_index_context=in_index_context,
                                                                                     hoist_safe=hoist_safe
                                                                                     and index == 0)
                prologue.extend(comparator_prologue)
                comparators.append(rewritten_comparator)
            return prologue, ast.copy_location(
                ast.Compare(left=left, ops=copy.deepcopy(node.ops), comparators=comparators), node)

        if isinstance(node, ast.IfExp):
            prologue, test = self._outline_expression(node.test,
                                                      in_index_context=in_index_context,
                                                      hoist_safe=hoist_safe)
            body_prologue, body = self._outline_expression(node.body,
                                                           in_index_context=in_index_context,
                                                           hoist_safe=False)
            orelse_prologue, orelse = self._outline_expression(node.orelse,
                                                               in_index_context=in_index_context,
                                                               hoist_safe=False)
            prologue.extend(body_prologue)
            prologue.extend(orelse_prologue)
            return prologue, ast.copy_location(ast.IfExp(test=test, body=body, orelse=orelse), node)

        if isinstance(node, ast.Attribute):
            prologue, value = self._outline_expression(node.value,
                                                       in_index_context=in_index_context,
                                                       hoist_safe=hoist_safe)
            return prologue, ast.copy_location(ast.Attribute(value=value, attr=node.attr, ctx=node.ctx), node)

        if isinstance(node, ast.Tuple):
            prologue: List[ast.stmt] = []
            elements: List[ast.AST] = []
            for element in node.elts:
                element_prologue, rewritten_element = self._outline_expression(element,
                                                                               in_index_context=in_index_context,
                                                                               hoist_safe=hoist_safe)
                prologue.extend(element_prologue)
                elements.append(rewritten_element)
            return prologue, ast.copy_location(ast.Tuple(elts=elements, ctx=node.ctx), node)

        if isinstance(node, ast.List):
            prologue: List[ast.stmt] = []
            elements: List[ast.AST] = []
            for element in node.elts:
                element_prologue, rewritten_element = self._outline_expression(element,
                                                                               in_index_context=in_index_context,
                                                                               hoist_safe=hoist_safe)
                prologue.extend(element_prologue)
                elements.append(rewritten_element)
            return prologue, ast.copy_location(ast.List(elts=elements, ctx=node.ctx), node)

        return [], copy.deepcopy(node)

    def _outline_subscript_slice(self, node: ast.AST, *, hoist_safe: bool = True) -> Tuple[List[ast.stmt], ast.AST]:
        if isinstance(node, ast.Slice):
            prologue, lower = self._outline_expression(node.lower, in_index_context=True, hoist_safe=hoist_safe)
            upper_prologue, upper = self._outline_expression(node.upper, in_index_context=True, hoist_safe=hoist_safe)
            step_prologue, step = self._outline_expression(node.step, in_index_context=True, hoist_safe=hoist_safe)
            prologue.extend(upper_prologue)
            prologue.extend(step_prologue)
            return prologue, ast.copy_location(ast.Slice(lower=lower, upper=upper, step=step), node)

        if isinstance(node, ast.Tuple):
            prologue: List[ast.stmt] = []
            elements: List[ast.AST] = []
            for element in node.elts:
                element_prologue, rewritten_element = self._outline_subscript_slice(element, hoist_safe=hoist_safe)
                prologue.extend(element_prologue)
                elements.append(rewritten_element)
            return prologue, ast.copy_location(ast.Tuple(elts=elements, ctx=node.ctx), node)

        return self._outline_expression(node, in_index_context=True, hoist_safe=hoist_safe)

    def _outline_to_temp(self, value: ast.AST, prologue: List[ast.stmt]) -> Tuple[List[ast.stmt], ast.AST]:
        temp_name = self._fresh_name('__stree_idx')
        assign = ast.Assign(targets=[ast.Name(id=temp_name, ctx=ast.Store())], value=copy.deepcopy(value))
        assign = ast.fix_missing_locations(ast.copy_location(assign, value))
        prologue.append(assign)
        return prologue, ast.copy_location(ast.Name(id=temp_name, ctx=ast.Load()), value)

    def _prepend_statements(self, node: ast.stmt, prologue: List[ast.stmt]) -> ast.AST:
        if not prologue:
            return node
        return prologue + [node]

    def _mark_callback(self, statement: ast.stmt, reason: str) -> ast.stmt:
        setattr(statement, _CALLBACK_REASON_ATTR, reason)
        return ast.fix_missing_locations(statement)

    def _should_outline_index_expression(self, node: ast.AST) -> bool:
        if not isinstance(node, (ast.Call, ast.Subscript)):
            return False
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'slice':
            context = self._evaluation_context()
            for arg in node.args:
                if try_resolve_static_value(arg, context) is UNRESOLVED:
                    return True
            for keyword in node.keywords:
                if try_resolve_static_value(keyword.value, context) is UNRESOLVED:
                    return True
            return False
        return try_resolve_static_value(node, self._evaluation_context()) is UNRESOLVED

    def _seed_used_names(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                self._used_names.add(child.id)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._used_names.add(child.name)
            elif isinstance(child, ast.arg):
                self._used_names.add(child.arg)

    def _fresh_name(self, prefix: str) -> str:
        candidate = prefix
        while candidate in self._used_names:
            self._temp_counter += 1
            candidate = f'{prefix}{self._temp_counter}'
        self._used_names.add(candidate)
        return candidate

    def _evaluation_context(self) -> Dict[str, Any]:
        context = copy.copy(self.global_vars)
        context.update(self.callable_bindings)
        return context


class _DescriptorTrackingEnvironment:

    def __init__(self,
                 global_vars: Dict[str, Any],
                 *,
                 known_descriptors: Optional[Dict[str, data.Data]] = None,
                 seed_bindings: Optional[Dict[str, Any]] = None,
                 callable_bindings: Optional[Dict[str, Any]] = None) -> None:
        self.global_vars = copy.copy(global_vars)
        self.callable_bindings = dict(callable_bindings or {})
        self.static_bindings: Dict[str, Any] = {}
        self.sequence_lengths: Dict[str, int] = {}
        self.descriptor_bindings: Dict[str, data.Data] = {}

        for name, value in self.global_vars.items():
            self.static_bindings[name] = value
            if isinstance(value, (list, tuple)):
                self.sequence_lengths[name] = len(value)
            if isinstance(value, data.Data):
                self.descriptor_bindings[name] = copy.deepcopy(value)

        for name, descriptor in (known_descriptors or {}).items():
            self.descriptor_bindings[name] = copy.deepcopy(descriptor)

        for name, binding in (seed_bindings or {}).items():
            descriptor = getattr(binding, 'descriptor', None)
            if isinstance(descriptor, data.Data):
                self.descriptor_bindings[name] = copy.deepcopy(descriptor)
            structure = getattr(binding, 'structure', None)
            if isinstance(structure, (list, tuple)):
                self.sequence_lengths[name] = len(structure)

    def child(self, *, cleared_names: Sequence[str] = ()) -> '_DescriptorTrackingEnvironment':
        cloned = _DescriptorTrackingEnvironment(self.global_vars, callable_bindings=self.callable_bindings)
        cloned.static_bindings = copy.copy(self.static_bindings)
        cloned.sequence_lengths = dict(self.sequence_lengths)
        cloned.descriptor_bindings = {
            name: copy.deepcopy(descriptor)
            for name, descriptor in self.descriptor_bindings.items()
        }
        for name in cleared_names:
            cloned.static_bindings.pop(name, None)
            cloned.sequence_lengths.pop(name, None)
        return cloned

    def evaluation_context(self) -> Dict[str, Any]:
        context = copy.copy(pybuiltins.__dict__)
        context.update(self.global_vars)
        context.update(self.callable_bindings)
        context.update(self.static_bindings)
        return context

    def descriptor_for_base(self, node: ast.AST) -> Optional[data.Data]:
        if isinstance(node, ast.Name):
            descriptor = self.descriptor_bindings.get(node.id)
            if isinstance(descriptor, data.Data) and getattr(descriptor, 'shape', None) is not None:
                return copy.deepcopy(descriptor)
        return None

    def sequence_length_for_base(self, node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Name) and node.id in self.sequence_lengths:
            return self.sequence_lengths[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return len(node.elts)
        value = try_resolve_static_value(node, self.evaluation_context())
        if isinstance(value, (list, tuple)):
            return len(value)
        return None

    def evaluate_descriptor(self, node: ast.AST) -> Optional[data.Data]:
        descriptor = try_resolve_static_value(node, self.evaluation_context())
        if isinstance(descriptor, data.Data):
            return copy.deepcopy(descriptor)
        return None

    def update_target_binding(self, target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, ast.Name):
            resolved = try_resolve_static_value(value, self.evaluation_context())
            if resolved is UNRESOLVED:
                self.static_bindings.pop(target.id, None)
            else:
                self.static_bindings[target.id] = resolved

            if isinstance(value, (ast.Tuple, ast.List)):
                self.sequence_lengths[target.id] = len(value.elts)
            elif isinstance(resolved, (list, tuple)):
                self.sequence_lengths[target.id] = len(resolved)
            else:
                self.sequence_lengths.pop(target.id, None)

            if isinstance(value, ast.Name) and value.id in self.descriptor_bindings:
                self.descriptor_bindings[target.id] = copy.deepcopy(self.descriptor_bindings[value.id])
            return

        self.invalidate_target(target)

    def invalidate_target(self, target: ast.AST) -> None:
        self.invalidate_names(self.stored_names(target))

    def invalidate_names(self, names: Sequence[str]) -> None:
        for name in names:
            self.static_bindings.pop(name, None)
            self.sequence_lengths.pop(name, None)
            self.descriptor_bindings.pop(name, None)

    @staticmethod
    def stored_names(target: ast.AST) -> set[str]:
        names: set[str] = set()
        for child in ast.walk(target):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
        return names

    def assigned_names(self, body: Sequence[ast.stmt]) -> set[str]:
        names: set[str] = set()
        for statement in body:
            if isinstance(statement, ast.ExceptHandler) and statement.name:
                names.add(statement.name)
            names.update(self.stored_names(statement))
        return names


class ScheduleTreeNegativeIndexNormalizer(ast.NodeTransformer):
    """Rewrite definitely negative indices into extent-relative expressions.

    Examples:
        ``A[-1]`` becomes ``A[N - 1]`` when ``A`` has known extent ``N``.

        ``t[-i]`` becomes ``t[3 - i]`` for a statically known 3-element tuple or
        list when ``i`` is known positive.
    """

    def __init__(self,
                 global_vars: Dict[str, Any],
                 *,
                 known_descriptors: Optional[Dict[str, data.Data]] = None,
                 seed_bindings: Optional[Dict[str, Any]] = None,
                 callable_bindings: Optional[Dict[str, Any]] = None) -> None:
        self.global_vars = copy.copy(global_vars)
        self.callable_bindings = dict(callable_bindings or {})
        self._env = _DescriptorTrackingEnvironment(global_vars,
                                                   known_descriptors=known_descriptors,
                                                   seed_bindings=seed_bindings,
                                                   callable_bindings=callable_bindings)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = self._rewrite_in_child_scope(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_function_scope(node)

    if hasattr(ast, 'AsyncFunctionDef'):

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            return self._visit_function_scope(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node.value = self.visit(node.value)
        node.targets = [self.visit(target) for target in node.targets]
        for target in node.targets:
            self._env.update_target_binding(target, node.value)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if node.value is not None:
            node.value = self.visit(node.value)
        node.target = self.visit(node.target)
        descriptor = self._env.evaluate_descriptor(node.annotation)
        if isinstance(node.target, ast.Name) and descriptor is not None:
            self._env.descriptor_bindings[node.target.id] = descriptor
        if node.value is None:
            self._env.invalidate_target(node.target)
            return node
        self._env.update_target_binding(node.target, node.value)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        node.target = self.visit(node.target)
        node.value = self.visit(node.value)
        self._env.invalidate_target(node.target)
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        node.value = self.visit(node.value)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        node.test = self.visit(node.test)
        node.body = self._rewrite_in_child_scope(node.body)
        node.orelse = self._rewrite_in_child_scope(node.orelse)
        self._env.invalidate_names(self._env.assigned_names(node.body + node.orelse))
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        node.test = self.visit(node.test)
        node.body = self._rewrite_in_child_scope(node.body)
        node.orelse = self._rewrite_in_child_scope(node.orelse)
        self._env.invalidate_names(self._env.assigned_names(node.body + node.orelse))
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        node.body = self._rewrite_in_child_scope(node.body)
        node.orelse = self._rewrite_in_child_scope(node.orelse)
        self._env.invalidate_target(node.target)
        self._env.invalidate_names(self._env.assigned_names(node.body + node.orelse))
        return node

    if hasattr(ast, 'AsyncFor'):

        def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
            node.target = self.visit(node.target)
            node.iter = self.visit(node.iter)
            node.body = self._rewrite_in_child_scope(node.body)
            node.orelse = self._rewrite_in_child_scope(node.orelse)
            self._env.invalidate_target(node.target)
            self._env.invalidate_names(self._env.assigned_names(node.body + node.orelse))
            return node

    def visit_With(self, node: ast.With) -> ast.AST:
        for item in node.items:
            item.context_expr = self.visit(item.context_expr)
            if item.optional_vars is not None:
                item.optional_vars = self.visit(item.optional_vars)
        node.body = self._rewrite_in_child_scope(node.body)
        invalidated = self._env.assigned_names(node.body)
        for item in node.items:
            if item.optional_vars is not None:
                invalidated.update(self._env.stored_names(item.optional_vars))
        self._env.invalidate_names(invalidated)
        return node

    if hasattr(ast, 'AsyncWith'):

        def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
            for item in node.items:
                item.context_expr = self.visit(item.context_expr)
                if item.optional_vars is not None:
                    item.optional_vars = self.visit(item.optional_vars)
            node.body = self._rewrite_in_child_scope(node.body)
            invalidated = self._env.assigned_names(node.body)
            for item in node.items:
                if item.optional_vars is not None:
                    invalidated.update(self._env.stored_names(item.optional_vars))
            self._env.invalidate_names(invalidated)
            return node

    def visit_Try(self, node: ast.Try) -> ast.AST:
        node.body = self._rewrite_in_child_scope(node.body)
        node.orelse = self._rewrite_in_child_scope(node.orelse)
        node.finalbody = self._rewrite_in_child_scope(node.finalbody)
        for handler in node.handlers:
            handler.body = self._rewrite_in_child_scope(handler.body)
        invalidated = self._env.assigned_names(node.body + node.orelse + node.finalbody)
        for handler in node.handlers:
            invalidated.update(self._env.assigned_names(handler.body))
            if handler.name:
                invalidated.add(handler.name)
        self._env.invalidate_names(invalidated)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node = self.generic_visit(node)
        rewritten = self._rewrite_subscript(node)
        return ast.fix_missing_locations(ast.copy_location(rewritten, node))

    def _visit_function_scope(self, node: ast.AST) -> ast.AST:
        args = getattr(node, 'args', None)
        cleared_names = [arg.arg for arg in args.args] if args is not None else []
        node.body = self._rewrite_in_child_scope(node.body, cleared_names=cleared_names)
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

    def _rewrite_in_child_scope(self, body: List[ast.stmt], *, cleared_names: Sequence[str] = ()) -> List[ast.stmt]:
        saved_env = self._env
        self._env = self._env.child(cleared_names=cleared_names)
        try:
            return self._rewrite_body(body)
        finally:
            self._env = saved_env

    def _rewrite_subscript(self, node: ast.Subscript) -> ast.Subscript:
        descriptor = self._env.descriptor_for_base(node.value)
        if descriptor is not None:
            rewritten_slice = self._rewrite_descriptor_slice(node.slice, tuple(descriptor.shape))
            return ast.copy_location(ast.Subscript(value=node.value, slice=rewritten_slice, ctx=node.ctx), node)

        sequence_length = self._env.sequence_length_for_base(node.value)
        if sequence_length is not None:
            rewritten_slice = self._rewrite_index_with_extent(node.slice, ast.Constant(sequence_length))
            return ast.copy_location(ast.Subscript(value=node.value, slice=rewritten_slice, ctx=node.ctx), node)

        return node

    def _rewrite_descriptor_slice(self, slice_node: ast.AST, shape: Tuple[Any, ...]) -> ast.AST:
        if isinstance(slice_node, ast.Tuple):
            extents = self._slice_extents(slice_node.elts, shape)
            if extents is None:
                return slice_node
            elements = [
                self._rewrite_index_with_extent(element, extent) if extent is not None else element
                for element, extent in zip(slice_node.elts, extents)
            ]
            return ast.copy_location(ast.Tuple(elts=elements, ctx=slice_node.ctx), slice_node)

        if not shape:
            return slice_node
        return self._rewrite_index_with_extent(slice_node, self._extent_ast(shape[0]))

    def _slice_extents(self, elements: Sequence[ast.AST], shape: Tuple[Any, ...]) -> Optional[List[Optional[ast.AST]]]:
        if sum(1 for element in elements if self._is_ellipsis(element)) > 1:
            return None

        consumed = sum(1 for element in elements if not self._is_newaxis(element) and not self._is_ellipsis(element))
        ellipsis_dims = max(len(shape) - consumed, 0)
        dim_index = 0
        extents: List[Optional[ast.AST]] = []

        for element in elements:
            if self._is_newaxis(element):
                extents.append(None)
                continue
            if self._is_ellipsis(element):
                extents.append(None)
                dim_index += ellipsis_dims
                continue
            if dim_index >= len(shape):
                return None
            extents.append(self._extent_ast(shape[dim_index]))
            dim_index += 1
        return extents

    def _rewrite_index_with_extent(self, node: ast.AST, extent: ast.AST) -> ast.AST:
        if isinstance(node, ast.Slice):
            lower = self._rewrite_negative_expression(node.lower, extent)
            upper = self._rewrite_negative_expression(node.upper, extent)
            return ast.copy_location(ast.Slice(lower=lower, upper=upper, step=node.step), node)

        if isinstance(node, ast.List):
            return ast.copy_location(
                ast.List(elts=[self._rewrite_index_with_extent(element, extent) for element in node.elts],
                         ctx=node.ctx), node)

        if isinstance(node, ast.Tuple):
            return ast.copy_location(
                ast.Tuple(elts=[self._rewrite_index_with_extent(element, extent) for element in node.elts],
                          ctx=node.ctx), node)

        return self._rewrite_negative_expression(node, extent)

    def _rewrite_negative_expression(self, node: Optional[ast.AST], extent: ast.AST) -> Optional[ast.AST]:
        if node is None:
            return None

        resolved = try_resolve_static_value(node, self._env.evaluation_context())
        if not self._is_definitely_negative_value(resolved):
            return node

        magnitude = self._negative_magnitude(node, resolved)
        return ast.copy_location(ast.BinOp(left=copy.deepcopy(extent), op=ast.Sub(), right=magnitude), node)

    @staticmethod
    def _is_definitely_negative_value(value: Any) -> bool:
        if value is UNRESOLVED or isinstance(value, bool):
            return False
        is_negative = getattr(value, 'is_negative', None)
        if is_negative is True:
            return True
        try:
            return (value < 0) == True
        except Exception:
            return False

    @staticmethod
    def _negative_magnitude(node: ast.AST, resolved: Any) -> ast.AST:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return copy.deepcopy(node.operand)
        if isinstance(resolved, numbers.Integral) and not isinstance(resolved, bool):
            return ast.Constant(value=abs(int(resolved)))
        return ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(node))

    @staticmethod
    def _is_newaxis(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value is None

    @staticmethod
    def _is_ellipsis(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value is Ellipsis

    @staticmethod
    def _extent_ast(extent: Any) -> ast.AST:
        if isinstance(extent, ast.AST):
            return copy.deepcopy(extent)
        return ast.parse(str(extent), mode='eval').body


def desugar_schedule_tree_expansions(parsed_ast: ast.AST,
                                     *,
                                     filename: str,
                                     global_vars: Dict[str, Any],
                                     known_descriptors: Optional[Dict[str, data.Data]] = None,
                                     seed_bindings: Optional[Dict[str, Any]] = None,
                                     callable_bindings: Optional[Dict[str, Any]] = None) -> ast.AST:
    """Rewrite schedule-tree-specific syntax before AST lowering."""
    expanded = ScheduleTreeExpansionDesugarer(filename, global_vars,
                                              callable_bindings=callable_bindings).visit(copy.deepcopy(parsed_ast))
    canonical = ScheduleTreeNegativeIndexNormalizer(global_vars,
                                                    known_descriptors=known_descriptors,
                                                    seed_bindings=seed_bindings,
                                                    callable_bindings=callable_bindings).visit(expanded)
    outlined = ScheduleTreeSubscriptIndexDesugarer(global_vars, callable_bindings=callable_bindings).visit(canonical)
    return ast.fix_missing_locations(outlined)


def callback_reason(node: ast.AST) -> Optional[str]:
    """Return the callback reason attached by schedule-tree desugaring, if any."""
    return getattr(node, _CALLBACK_REASON_ATTR, None)
