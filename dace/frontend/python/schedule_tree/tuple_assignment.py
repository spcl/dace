"""Tuple and list assignment lowering for the direct schedule-tree frontend."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from dace.frontend.python import astutils

_CONTAINER_INIT_ATTR = '_schedule_tree_container_init_only'
_ELEMENT_ASSIGNMENT_ATTR = '_schedule_tree_tuple_element_assignment'


@dataclass
class _PackedSequence:
    container: ast.Name
    elements: List[Union[ast.AST, '_PackedSequence']]
    is_list: bool


_SourceValue = Union[ast.AST, _PackedSequence]


def is_container_initialization(node: ast.AST) -> bool:
    """Return True when *node* only establishes tuple/list descriptor metadata."""
    return bool(getattr(node, _CONTAINER_INIT_ATTR, False))


def is_tuple_element_assignment(node: ast.AST) -> bool:
    """Return True when *node* copies one packed tuple/list element."""
    return bool(getattr(node, _ELEMENT_ASSIGNMENT_ATTR, False))


class ScheduleTreeTupleAssignmentLowerer(ast.NodeTransformer):
    """Lower tuple/list packing and unpacking into element assignments.

    The pass keeps Python assignment ordering explicit by first materializing the
    literal right-hand side into fresh element temporaries, then assigning
    destructured targets from those temporaries. For example, ``A, B = B, A``
    becomes a metadata-only tuple initializer, two element assignments, then two
    ordinary name assignments from the frozen element values. Destructuring from
    non-literal values, including function returns, is left to the regular DaCe
    frontend lowering.
    """

    def __init__(self) -> None:
        self._packed_sequences: Dict[str, _PackedSequence] = {}
        self._used_names: set[str] = set()
        self._temp_counter = 0

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self._seed_used_names(node)
        saved = self._packed_sequences
        self._packed_sequences = {}
        node.body = self._rewrite_body(node.body)
        self._packed_sequences = saved
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._seed_used_names(node)
        saved = self._packed_sequences
        self._packed_sequences = {}
        node.body = self._rewrite_body(node.body)
        self._packed_sequences = saved
        return node

    if hasattr(ast, 'AsyncFunctionDef'):

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            self._seed_used_names(node)
            saved = self._packed_sequences
            self._packed_sequences = {}
            node.body = self._rewrite_body(node.body)
            self._packed_sequences = saved
            return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self._is_sequence_literal(node.value):
            return self._pack_named_sequence(node.targets[0].id, node.value, node)

        if any(isinstance(target, (ast.Tuple, ast.List)) for target in node.targets):
            source = self._assignment_source(node.value, node)
            if source is None:
                self._invalidate_targets(node.targets)
                return node

            prefix, source_value = source
            lowered: List[ast.stmt] = list(prefix)
            for target in node.targets:
                if isinstance(target, (ast.Tuple, ast.List)):
                    assignments = self._lower_destructuring_target(target, source_value, node)
                    if assignments is None:
                        self._invalidate_target(target)
                        return node
                    lowered.extend(assignments)
                else:
                    lowered.append(
                        ast.copy_location(
                            ast.Assign(targets=[astutils.copy_tree(target)], value=self._source_expr(source_value)),
                            node))
                    self._invalidate_target(target)
            return lowered

        node.value = self._rewrite_expression(node.value)
        for target in node.targets:
            self._invalidate_target(target)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if node.value is not None:
            node.value = self._rewrite_expression(node.value)
        self._invalidate_target(node.target)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        node.value = self._rewrite_expression(node.value)
        self._invalidate_target(node.target)
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        node.value = self._rewrite_expression(node.value)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        node.test = self._rewrite_expression(node.test)
        node.body = self._rewrite_child_body(node.body)
        node.orelse = self._rewrite_child_body(node.orelse)
        self._invalidate_assigned_names(node.body + node.orelse)
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        node.test = self._rewrite_expression(node.test)
        node.body = self._rewrite_child_body(node.body)
        node.orelse = self._rewrite_child_body(node.orelse)
        self._invalidate_assigned_names(node.body + node.orelse)
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        node.iter = self._rewrite_expression(node.iter)
        self._invalidate_target(node.target)
        node.body = self._rewrite_child_body(node.body)
        node.orelse = self._rewrite_child_body(node.orelse)
        return node

    if hasattr(ast, 'AsyncFor'):

        def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
            node.iter = self._rewrite_expression(node.iter)
            self._invalidate_target(node.target)
            node.body = self._rewrite_child_body(node.body)
            node.orelse = self._rewrite_child_body(node.orelse)
            return node

    def visit_With(self, node: ast.With) -> ast.AST:
        for item in node.items:
            item.context_expr = self._rewrite_expression(item.context_expr)
            if item.optional_vars is not None:
                self._invalidate_target(item.optional_vars)
        node.body = self._rewrite_child_body(node.body)
        return node

    if hasattr(ast, 'AsyncWith'):

        def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
            for item in node.items:
                item.context_expr = self._rewrite_expression(item.context_expr)
                if item.optional_vars is not None:
                    self._invalidate_target(item.optional_vars)
            node.body = self._rewrite_child_body(node.body)
            return node

    def visit_Try(self, node: ast.Try) -> ast.AST:
        node.body = self._rewrite_child_body(node.body)
        node.orelse = self._rewrite_child_body(node.orelse)
        node.finalbody = self._rewrite_child_body(node.finalbody)
        for handler in node.handlers:
            handler.body = self._rewrite_child_body(handler.body)
        self._invalidate_assigned_names(node.body + node.orelse + node.finalbody)
        return node

    def _rewrite_body(self, body: Sequence[ast.stmt]) -> List[ast.stmt]:
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

    def _rewrite_child_body(self, body: Sequence[ast.stmt]) -> List[ast.stmt]:
        saved = self._packed_sequences
        self._packed_sequences = copy.deepcopy(saved)
        rewritten = self._rewrite_body(body)
        self._packed_sequences = saved
        return rewritten

    def _pack_named_sequence(self, name: str, value: ast.AST, template: ast.AST) -> List[ast.stmt]:
        statements, sequence = self._pack_sequence(name, value, template)
        self._packed_sequences[name] = sequence
        return statements

    def _pack_sequence(self, name: str, value: ast.AST, template: ast.AST) -> tuple[List[ast.stmt], _PackedSequence]:
        init = ast.copy_location(
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=astutils.copy_tree(value)), template)
        setattr(init, _CONTAINER_INIT_ATTR, True)

        statements: List[ast.stmt] = [init]
        elements: List[Union[ast.AST, _PackedSequence]] = []
        for index, element in enumerate(value.elts):
            if self._is_sequence_literal(element):
                element_name = self._fresh_name(f'{name}_{index}')
                nested_statements, nested_sequence = self._pack_sequence(element_name, element, template)
                statements.extend(nested_statements)
                elements.append(nested_sequence)
                continue

            element_name = self._fresh_name(f'{name}_{index}')
            element_target = ast.Name(id=element_name, ctx=ast.Store())
            element_value = self._rewrite_expression(element)
            statements.append(self._element_assignment(element_target, element_value, template))
            elements.append(ast.Name(id=element_name, ctx=ast.Load()))

        sequence = _PackedSequence(container=ast.Name(id=name, ctx=ast.Load()),
                                   elements=elements,
                                   is_list=isinstance(value, ast.List))
        return (statements, sequence)

    def _assignment_source(self, value: ast.AST, template: ast.AST) -> Optional[Tuple[List[ast.stmt], _SourceValue]]:
        if isinstance(value, ast.Name):
            packed = self._packed_sequences.get(value.id)
            if packed is not None:
                return ([], copy.deepcopy(packed))
        if self._is_sequence_literal(value):
            temp_name = self._fresh_name('__stree_unpack_tmp')
            statements, packed = self._pack_sequence(temp_name, value, template)
            self._packed_sequences[temp_name] = packed
            return (statements, packed)
        return None

    def _lower_destructuring_target(self, target: ast.AST, source: _SourceValue,
                                    template: ast.AST) -> Optional[List[ast.stmt]]:
        if isinstance(target, ast.Name):
            value = self._source_expr(source)
            if isinstance(source, _PackedSequence):
                self._packed_sequences[target.id] = copy.deepcopy(source)
            else:
                self._packed_sequences.pop(target.id, None)
            return [self._element_assignment(astutils.copy_tree(target), value, template)]

        if not isinstance(target, (ast.Tuple, ast.List)):
            self._invalidate_target(target)
            return [self._element_assignment(astutils.copy_tree(target), self._source_expr(source), template)]

        if any(isinstance(element, ast.Starred) for element in target.elts):
            return None

        elements = self._source_elements(source, len(target.elts))
        if elements is None:
            return None

        result: List[ast.stmt] = []
        for subtarget, subsource in zip(target.elts, elements):
            lowered = self._lower_destructuring_target(subtarget, subsource, template)
            if lowered is None:
                return None
            result.extend(lowered)
        return result

    def _source_elements(self, source: _SourceValue, expected_length: int) -> Optional[List[_SourceValue]]:
        if isinstance(source, _PackedSequence):
            if len(source.elements) != expected_length:
                return None
            return [astutils.copy_tree(element) for element in source.elements]

        return [
            ast.Subscript(value=astutils.copy_tree(source), slice=ast.Constant(value=index), ctx=ast.Load())
            for index in range(expected_length)
        ]

    @staticmethod
    def _source_expr(source: _SourceValue) -> ast.AST:
        if isinstance(source, _PackedSequence):
            return astutils.copy_tree(source.container)
        return astutils.copy_tree(source)

    @staticmethod
    def _element_assignment(target: ast.AST, value: ast.AST, template: ast.AST) -> ast.Assign:
        assignment = ast.copy_location(ast.Assign(targets=[target], value=value), template)
        setattr(assignment, _ELEMENT_ASSIGNMENT_ATTR, True)
        return assignment

    def _rewrite_expression(self, node: ast.AST) -> ast.AST:
        outer = self

        class _ExpressionRewriter(ast.NodeTransformer):

            def visit_Subscript(self, subscript: ast.Subscript) -> ast.AST:
                subscript = self.generic_visit(subscript)
                replacement = outer._packed_subscript_replacement(subscript)
                return replacement if replacement is not None else subscript

        return _ExpressionRewriter().visit(astutils.copy_tree(node))

    def _packed_subscript_replacement(self, node: ast.Subscript) -> Optional[ast.AST]:
        if not isinstance(node.value, ast.Name):
            return None
        packed = self._packed_sequences.get(node.value.id)
        if packed is None:
            return None
        index = self._constant_int(node.slice)
        if index is None or index < 0 or index >= len(packed.elements):
            return None
        element = packed.elements[index]
        return self._source_expr(element)

    @staticmethod
    def _constant_int(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp):
            left = ScheduleTreeTupleAssignmentLowerer._constant_int(node.left)
            right = ScheduleTreeTupleAssignmentLowerer._constant_int(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
        return None

    @staticmethod
    def _is_sequence_literal(node: ast.AST) -> bool:
        return isinstance(node,
                          (ast.Tuple, ast.List)) and not any(isinstance(element, ast.Starred) for element in node.elts)

    def _invalidate_targets(self, targets: Sequence[ast.AST]) -> None:
        for target in targets:
            self._invalidate_target(target)

    def _invalidate_target(self, target: ast.AST) -> None:
        for child in ast.walk(target):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                self._packed_sequences.pop(child.id, None)

    def _invalidate_assigned_names(self, statements: Sequence[ast.stmt]) -> None:
        for statement in statements:
            for child in ast.walk(statement):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    self._packed_sequences.pop(child.id, None)

    def _fresh_name(self, prefix: str) -> str:
        candidate = prefix
        while candidate in self._used_names or candidate in self._packed_sequences:
            self._temp_counter += 1
            candidate = f'{prefix}_{self._temp_counter}'
        self._used_names.add(candidate)
        return candidate

    def _seed_used_names(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                self._used_names.add(child.id)


def lower_tuple_assignments(parsed_ast: ast.AST) -> ast.AST:
    """Lower tuple/list packing and destructuring assignments in *parsed_ast*."""
    return ScheduleTreeTupleAssignmentLowerer().visit(astutils.copy_tree(parsed_ast))
