# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Helpers for recovering and inlining lambda expressions.

Example:
    If ``f`` is known to be ``lambda a, b: a + b``, then a call such as
    ``f(A, B)`` can be rewritten directly to ``A + B`` before schedule-tree
    lowering continues.
"""

import ast
import inspect
from typing import Any, Dict, Optional

import sympy

from dace import dtypes, symbolic
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

    return ast.fix_missing_locations(astutils.copy_tree(candidates[0]))


def inline_lambda_call(lambda_node: ast.Lambda, call_node: ast.Call) -> ast.AST:
    """Inline a call to ``lambda_node`` by substituting actual arguments."""
    bindings = _bind_lambda_arguments(lambda_node.args, call_node)

    class _LambdaInliner(ast.NodeTransformer):

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id in bindings:
                return astutils.copy_tree(bindings[node.id])
            return node

        def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
            return node

    return ast.fix_missing_locations(_LambdaInliner().visit(astutils.copy_tree(lambda_node.body)))


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
        bindings[parameter.arg] = astutils.copy_tree(actual)

    remaining_keywords = {kw.arg: kw.value for kw in call_node.keywords if kw.arg is not None}
    for index, parameter in enumerate(parameters[len(positional):], start=len(positional)):
        if parameter.arg in remaining_keywords:
            bindings[parameter.arg] = astutils.copy_tree(remaining_keywords.pop(parameter.arg))
            continue
        default_index = index - default_offset
        if default_index >= 0:
            bindings[parameter.arg] = astutils.copy_tree(defaults[default_index])
            continue
        raise TypeError(f'Missing argument {parameter.arg!r} for lambda call')

    for parameter in parameters[:len(positional)]:
        if parameter.arg in remaining_keywords:
            raise TypeError(f'Multiple values for argument {parameter.arg!r} in lambda call')

    if remaining_keywords:
        raise TypeError(f'Unexpected keyword arguments in lambda call: {sorted(remaining_keywords)}')

    return bindings


class LambdaResolver:
    """Resolve known lambda values and inline their call sites.

    The resolver keeps track of lambda values that are visible by name in the
    current lowering scope and can recover AST for global or closure-backed
    lambdas when source is available.

    Example:
        Given a binding ``f = lambda a, b: a + b``, calling
        ``inline_known_lambda_calls(...)`` on the AST for ``f(A, B)`` returns an
        expression equivalent to ``A + B``.
    """

    def __init__(self,
                 globals_env: Dict[str, Any],
                 lambda_bindings: Dict[str, ast.Lambda],
                 callable_bindings: Dict[str, Any],
                 *,
                 cache: Optional[Dict[str, Optional[ast.Lambda]]] = None) -> None:
        self.globals = globals_env
        self.lambda_bindings = lambda_bindings
        self.callable_bindings = callable_bindings
        self._global_lambda_cache = cache if cache is not None else {}

    def update_binding(self, name: str, value: ast.AST) -> None:
        lambda_node = self.resolve_known_lambda_node(value)
        if lambda_node is None:
            self.lambda_bindings.pop(name, None)
            return
        self.lambda_bindings[name] = lambda_node

    def bind_value(self, name: str, value: Any) -> None:
        lambda_node = self.resolve_global_lambda_node(value)
        if lambda_node is None:
            self.lambda_bindings.pop(name, None)
            return
        self.lambda_bindings[name] = lambda_node
        self._global_lambda_cache[name] = astutils.copy_tree(lambda_node)

    def resolve_known_lambda_node(self, node: ast.AST) -> Optional[ast.Lambda]:
        if isinstance(node, ast.Lambda):
            return astutils.copy_tree(node)
        if not isinstance(node, ast.Name):
            return None
        if node.id in self.lambda_bindings:
            return astutils.copy_tree(self.lambda_bindings[node.id])
        if node.id in self.callable_bindings:
            lambda_node = self.resolve_global_lambda_node(self.callable_bindings[node.id])
            return astutils.copy_tree(lambda_node) if lambda_node is not None else None
        if node.id in self._global_lambda_cache:
            cached = self._global_lambda_cache[node.id]
            return astutils.copy_tree(cached) if cached is not None else None
        value = self.globals.get(node.id)
        lambda_node = self.resolve_global_lambda_node(value) if value is not None else None
        self._global_lambda_cache[node.id] = astutils.copy_tree(lambda_node) if lambda_node is not None else None
        return astutils.copy_tree(lambda_node) if lambda_node is not None else None

    def resolve_global_lambda_node(self, value: Any) -> Optional[ast.Lambda]:
        lambda_node = extract_lambda_ast(value)
        if lambda_node is None:
            return None

        lambda_globals = _resolve_lambda_environment(value)
        for name, captured_value in lambda_globals.items():
            self.globals.setdefault(name, captured_value)

        inline_globals = {
            name: captured_value
            for name, captured_value in lambda_globals.items() if _can_inline_lambda_capture(captured_value)
        }
        if not inline_globals:
            return astutils.copy_tree(lambda_node)

        return _rewrite_lambda_free_names(lambda_node, inline_globals)

    def inline_known_lambda_calls(self, node: ast.AST) -> ast.AST:
        resolver = self

        class _KnownLambdaInliner(ast.NodeTransformer):

            def visit_Call(self, call_node: ast.Call) -> ast.AST:
                rewritten = ast.Call(
                    func=self.visit(call_node.func),
                    args=[self.visit(arg) for arg in call_node.args],
                    keywords=[ast.keyword(arg=kw.arg, value=self.visit(kw.value)) for kw in call_node.keywords])
                lambda_node = resolver.resolve_known_lambda_node(rewritten.func)
                if lambda_node is None:
                    return ast.copy_location(rewritten, call_node)
                try:
                    inlined = inline_lambda_call(lambda_node, rewritten)
                except TypeError:
                    return ast.copy_location(rewritten, call_node)
                return ast.copy_location(self.visit(inlined), call_node)

        return ast.fix_missing_locations(_KnownLambdaInliner().visit(astutils.copy_tree(node)))


def _resolve_lambda_environment(value: Any) -> Dict[str, Any]:
    try:
        closure_vars = inspect.getclosurevars(value)
    except Exception:
        closure_vars = None

    if closure_vars is not None:
        resolved = dict(closure_vars.globals)
        resolved.update(closure_vars.nonlocals)
        return resolved

    resolved = {}
    globals_env = getattr(value, '__globals__', {})
    closure = getattr(value, '__closure__', None)
    freevars = getattr(getattr(value, '__code__', None), 'co_freevars', ())
    if closure is not None:
        for name, cell in zip(freevars, closure):
            try:
                resolved[name] = cell.cell_contents
            except ValueError:
                resolved[name] = None

    for name in _lambda_loaded_names(extract_lambda_ast(value)):
        if name not in resolved and name in globals_env:
            resolved[name] = globals_env[name]
    return resolved


def _can_inline_lambda_capture(value: Any) -> bool:
    if isinstance(value, ast.AST):
        return True
    if isinstance(value, (symbolic.symbol, sympy.Basic)):
        return True
    if dtypes.isconstant(value):
        return True
    if isinstance(value, tuple):
        return all(_can_inline_lambda_capture(element) for element in value)
    if isinstance(value, list):
        return all(_can_inline_lambda_capture(element) for element in value)
    if isinstance(value, dict):
        return all(
            _can_inline_lambda_capture(key) and _can_inline_lambda_capture(element) for key, element in value.items())
    return False


def _rewrite_lambda_free_names(lambda_node: ast.Lambda, env: Dict[str, Any]) -> ast.Lambda:
    rewriter = _LambdaFreeNameRewriter(env)
    return ast.fix_missing_locations(rewriter.visit(astutils.copy_tree(lambda_node)))


def _value_to_ast(value: Any, template_node: ast.AST) -> Optional[ast.AST]:
    if isinstance(value, ast.AST):
        return ast.copy_location(astutils.copy_tree(value), template_node)

    if isinstance(value, symbolic.symbol):
        return ast.copy_location(ast.Name(id=value.name, ctx=ast.Load()), template_node)

    if isinstance(value, sympy.Basic):
        return ast.copy_location(ast.parse(symbolic.symstr(value), mode='eval').body, template_node)

    if isinstance(value, list):
        elements = [_value_to_ast(element, template_node) for element in value]
        if any(element is None for element in elements):
            return None
        return ast.copy_location(ast.List(elts=elements, ctx=ast.Load()), template_node)

    if isinstance(value, tuple):
        elements = [_value_to_ast(element, template_node) for element in value]
        if any(element is None for element in elements):
            return None
        return ast.copy_location(ast.Tuple(elts=elements, ctx=ast.Load()), template_node)

    if isinstance(value, dict):
        keys = []
        values = []
        for key, item in value.items():
            key_ast = _value_to_ast(key, template_node)
            value_ast = _value_to_ast(item, template_node)
            if key_ast is None or value_ast is None:
                return None
            keys.append(key_ast)
            values.append(value_ast)
        return ast.copy_location(ast.Dict(keys=keys, values=values), template_node)

    if dtypes.isconstant(value):
        return astutils.create_constant(value, template_node)

    return None


class _LambdaFreeNameRewriter(ast.NodeTransformer):

    def __init__(self, env: Dict[str, Any]) -> None:
        self.env = env
        self.scope_stack = []

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        self.scope_stack.append(_lambda_parameter_names(node.args))
        try:
            node.body = self.visit(node.body)
            return node
        finally:
            self.scope_stack.pop()

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            if self.scope_stack and isinstance(node.ctx, ast.Store):
                self.scope_stack[-1].add(node.id)
            return node

        if any(node.id in scope for scope in reversed(self.scope_stack)):
            return node
        if node.id not in self.env:
            return node

        replacement = _value_to_ast(self.env[node.id], node)
        return replacement if replacement is not None else node

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        return self._visit_comprehension(node, 'elt')

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        return self._visit_comprehension(node, 'elt')

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        return self._visit_comprehension(node, 'elt')

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        pushed = self._push_comprehension_scopes(node.generators)
        try:
            node.key = self.visit(node.key)
            node.value = self.visit(node.value)
            return node
        finally:
            self._pop_comprehension_scopes(pushed)

    def _visit_comprehension(self, node: ast.AST, field: str) -> ast.AST:
        pushed = self._push_comprehension_scopes(node.generators)
        try:
            setattr(node, field, self.visit(getattr(node, field)))
            return node
        finally:
            self._pop_comprehension_scopes(pushed)

    def _push_comprehension_scopes(self, generators) -> int:
        pushed = 0
        for generator in generators:
            generator.iter = self.visit(generator.iter)
            bound_names = _store_target_names(generator.target)
            self.scope_stack.append(bound_names)
            pushed += 1
            generator.ifs = [self.visit(condition) for condition in generator.ifs]
        return pushed

    def _pop_comprehension_scopes(self, pushed: int) -> None:
        for _ in range(pushed):
            self.scope_stack.pop()


class _LambdaLoadedNameCollector(ast.NodeVisitor):

    def __init__(self) -> None:
        self.scope_stack = []
        self.loaded_names = set()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.scope_stack.append(_lambda_parameter_names(node.args))
        try:
            self.visit(node.body)
        finally:
            self.scope_stack.pop()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not any(node.id in scope for scope in reversed(self.scope_stack)):
            self.loaded_names.add(node.id)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, lambda: self.visit(node.elt))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, lambda: self.visit(node.elt))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, lambda: self.visit(node.elt))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, lambda: (self.visit(node.key), self.visit(node.value)))

    def _visit_comprehension(self, generators, visit_result) -> None:
        pushed = 0
        for generator in generators:
            self.visit(generator.iter)
            self.scope_stack.append(_store_target_names(generator.target))
            pushed += 1
            for condition in generator.ifs:
                self.visit(condition)
        try:
            visit_result()
        finally:
            for _ in range(pushed):
                self.scope_stack.pop()


def _lambda_loaded_names(lambda_node: Optional[ast.Lambda]) -> set[str]:
    if lambda_node is None:
        return set()
    collector = _LambdaLoadedNameCollector()
    collector.visit(astutils.copy_tree(lambda_node))
    return collector.loaded_names


def _lambda_parameter_names(args: ast.arguments) -> set[str]:
    names = {arg.arg for arg in args.posonlyargs}
    names.update(arg.arg for arg in args.args)
    names.update(arg.arg for arg in args.kwonlyargs)
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _store_target_names(target: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(target):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
    return names
