# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Helpers for recovering and inlining lambda expressions."""

import ast
import copy
import inspect
from typing import Dict, Optional

from dace.frontend.python import astutils


def extract_lambda_ast(func) -> Optional[ast.Lambda]:
    """Recover the AST for a Python lambda when its source is available."""
    if not inspect.isfunction(func) or getattr(func, '__name__', None) != '<lambda>':
        return None

    try:
        src_ast, _, _, _ = astutils.function_to_ast(func)
    except Exception:
        return None

    target_lineno = getattr(func.__code__, 'co_firstlineno', None)
    candidates = [node for node in ast.walk(src_ast) if isinstance(node, ast.Lambda)]
    if target_lineno is not None:
        exact = [node for node in candidates if getattr(node, 'lineno', None) == target_lineno]
        if exact:
            candidates = exact
    if len(candidates) != 1:
        return None

    return ast.fix_missing_locations(copy.deepcopy(candidates[0]))


def inline_lambda_call(lambda_node: ast.Lambda, call_node: ast.Call) -> ast.AST:
    """Inline a call to ``lambda_node`` by substituting actual arguments."""
    bindings = _bind_lambda_arguments(lambda_node.args, call_node)

    class _LambdaInliner(ast.NodeTransformer):

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id in bindings:
                return copy.deepcopy(bindings[node.id])
            return node

        def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
            return node

    return ast.fix_missing_locations(_LambdaInliner().visit(copy.deepcopy(lambda_node.body)))


def _bind_lambda_arguments(args: ast.arguments, call_node: ast.Call) -> Dict[str, ast.AST]:
    if args.vararg is not None or args.kwarg is not None or args.kwonlyargs:
        raise TypeError('Only simple positional/keyword lambda arguments are supported')

    parameters = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    default_offset = len(parameters) - len(defaults)

    bindings: Dict[str, ast.AST] = {}
    positional = list(call_node.args)
    if len(positional) > len(parameters):
        raise TypeError('Too many positional arguments for lambda call')

    for parameter, actual in zip(parameters, positional):
        bindings[parameter.arg] = copy.deepcopy(actual)

    remaining_keywords = {kw.arg: kw.value for kw in call_node.keywords if kw.arg is not None}
    for index, parameter in enumerate(parameters[len(positional):], start=len(positional)):
        if parameter.arg in remaining_keywords:
            bindings[parameter.arg] = copy.deepcopy(remaining_keywords.pop(parameter.arg))
            continue
        default_index = index - default_offset
        if default_index >= 0:
            bindings[parameter.arg] = copy.deepcopy(defaults[default_index])
            continue
        raise TypeError(f'Missing argument {parameter.arg!r} for lambda call')

    for parameter in parameters[:len(positional)]:
        if parameter.arg in remaining_keywords:
            raise TypeError(f'Multiple values for argument {parameter.arg!r} in lambda call')

    if remaining_keywords:
        raise TypeError(f'Unexpected keyword arguments in lambda call: {sorted(remaining_keywords)}')

    return bindings
