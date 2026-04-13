# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Helpers for lowering Python ``match`` statements to simpler AST forms."""

import ast
import copy
from typing import Dict, List, Optional, Tuple


class UnsupportedMatchPatternError(TypeError):
    """Raised when a match pattern cannot be lowered to an if-chain."""


_Bindings = List[Tuple[str, ast.AST]]


def lower_match_to_statements(node: ast.Match, subject_expr: ast.AST) -> List[ast.stmt]:
    """Lower a supported ``match`` node to equivalent ``if`` statements.

    Supported patterns are limited to value, singleton, wildcard, capture,
    alias, guarded cases, ``or`` patterns without bindings, and fixed-length
    sequence patterns without starred items. Class and mapping patterns, and
    more advanced structural patterns, still fall back to callbacks.
    """
    lowered: List[ast.stmt] = []

    for case in reversed(node.cases):
        condition, bindings = _lower_pattern(case.pattern, subject_expr)
        binding_map = {name: copy.deepcopy(expr) for name, expr in bindings}

        if case.guard is not None:
            guard = _substitute_capture_loads(case.guard, binding_map)
            condition = guard if condition is None else ast.BoolOp(op=ast.And(), values=[condition, guard])

        body: List[ast.stmt] = [
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=copy.deepcopy(expr))
            for name, expr in bindings
        ]
        body.extend(copy.deepcopy(case.body))

        if condition is None:
            lowered = body
        else:
            lowered = [ast.If(test=condition, body=body, orelse=lowered)]

    return [ast.fix_missing_locations(stmt) for stmt in lowered]


def _lower_pattern(pattern: ast.pattern, subject_expr: ast.AST) -> Tuple[Optional[ast.AST], _Bindings]:
    if isinstance(pattern, ast.MatchValue):
        return _eq_condition(subject_expr, pattern.value), []

    if isinstance(pattern, ast.MatchSingleton):
        return ast.Compare(left=copy.deepcopy(subject_expr), ops=[ast.Is()],
                           comparators=[ast.Constant(pattern.value)]), []

    if isinstance(pattern, ast.MatchSequence):
        if any(isinstance(subpattern, ast.MatchStar) for subpattern in pattern.patterns):
            raise UnsupportedMatchPatternError('sequence patterns with starred items are not supported yet')

        conditions: List[ast.AST] = [_fixed_length_sequence_condition(subject_expr, len(pattern.patterns))]
        bindings: _Bindings = []
        for index, subpattern in enumerate(pattern.patterns):
            element_expr = ast.Subscript(value=copy.deepcopy(subject_expr), slice=ast.Constant(index), ctx=ast.Load())
            element_condition, element_bindings = _lower_pattern(subpattern, element_expr)
            if element_condition is not None:
                conditions.append(element_condition)
            bindings.extend(element_bindings)
        return _combine_conditions(conditions), bindings

    if isinstance(pattern, ast.MatchAs):
        if pattern.pattern is None:
            if pattern.name is None:
                return None, []
            return None, [(pattern.name, copy.deepcopy(subject_expr))]

        condition, bindings = _lower_pattern(pattern.pattern, subject_expr)
        if pattern.name is not None:
            bindings = bindings + [(pattern.name, copy.deepcopy(subject_expr))]
        return condition, bindings

    if isinstance(pattern, ast.MatchOr):
        alternatives = [_lower_pattern(alt, subject_expr) for alt in pattern.patterns]
        if any(bindings for _, bindings in alternatives):
            raise UnsupportedMatchPatternError('or-patterns with bindings are not supported yet')
        if any(condition is None for condition, _ in alternatives):
            return None, []
        return ast.BoolOp(op=ast.Or(), values=[condition for condition, _ in alternatives]), []

    if isinstance(pattern, (ast.MatchClass, ast.MatchMapping, ast.MatchStar)):
        raise UnsupportedMatchPatternError(f'Unsupported match pattern: {type(pattern).__name__}')

    raise UnsupportedMatchPatternError(f'Unsupported match pattern: {type(pattern).__name__}')


def _eq_condition(left: ast.AST, right: ast.AST) -> ast.Compare:
    return ast.Compare(left=copy.deepcopy(left), ops=[ast.Eq()], comparators=[copy.deepcopy(right)])


def _fixed_length_sequence_condition(subject_expr: ast.AST, length: int) -> ast.AST:
    return _combine_conditions([
        ast.BoolOp(op=ast.Or(),
                   values=[
                       ast.Call(func=ast.Name(id='isinstance', ctx=ast.Load()),
                                args=[copy.deepcopy(subject_expr),
                                      ast.Name(id='tuple', ctx=ast.Load())],
                                keywords=[]),
                       ast.Call(func=ast.Name(id='isinstance', ctx=ast.Load()),
                                args=[copy.deepcopy(subject_expr),
                                      ast.Name(id='list', ctx=ast.Load())],
                                keywords=[]),
                   ]),
        ast.Compare(left=ast.Call(func=ast.Name(id='len', ctx=ast.Load()),
                                  args=[copy.deepcopy(subject_expr)],
                                  keywords=[]),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(length)]),
    ])


def _combine_conditions(conditions: List[ast.AST]) -> ast.AST:
    if len(conditions) == 1:
        return conditions[0]
    return ast.BoolOp(op=ast.And(), values=conditions)


def _substitute_capture_loads(node: ast.AST, bindings: Dict[str, ast.AST]) -> ast.AST:

    class _CaptureSubstituter(ast.NodeTransformer):

        def visit_Name(self, inner: ast.Name) -> ast.AST:
            if isinstance(inner.ctx, ast.Load) and inner.id in bindings:
                return copy.deepcopy(bindings[inner.id])
            return inner

    return ast.fix_missing_locations(_CaptureSubstituter().visit(copy.deepcopy(node)))
