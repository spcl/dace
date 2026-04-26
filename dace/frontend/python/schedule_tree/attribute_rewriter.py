# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Helpers for rewriting attribute access on user-defined Python objects.

The direct schedule-tree frontend keeps ordinary attribute syntax such as
``obj.value`` unchanged for plain objects, but it makes descriptor behavior and
custom attribute hooks explicit when lowering user-defined objects.

Example:
    A descriptor-backed assignment such as ``holder.arr = A`` is rewritten to
    ``type(holder).__dict__['arr'].__set__(holder, A)`` so later lowering sees
    the same runtime behavior directly in the AST.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, Callable, Dict, Optional

from dace import data, dtypes, symbolic
from dace.frontend.python import astutils
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED, try_resolve_static_value


class AttributeRewriter:
    """Rewrite selected attribute reads and writes into explicit method calls.

    The rewriter handles user-defined objects whose attribute behavior is not
    plain field access. In particular, it expands descriptor loads and stores,
    and classes that override ``__getattribute__``, ``__getattr__``, or
    ``__setattr__``.

    Example:
        ``holder.arr`` can be rewritten to
        ``type(holder).__dict__['arr'].__get__(holder, type(holder))`` when
        ``arr`` is a descriptor on ``holder``'s class.
    """

    def __init__(self, evaluation_context: Callable[[], Dict[str, Any]]) -> None:
        self._evaluation_context = evaluation_context

    def rewrite_expression(self, node: ast.AST) -> ast.AST:
        """Return a copy of *node* with rewritten attribute reads."""

        class _AttributeLoadRewriter(ast.NodeTransformer):

            def __init__(self, rewriter: 'AttributeRewriter') -> None:
                self.rewriter = rewriter

            def visit_Attribute(self, attr_node: ast.Attribute) -> ast.AST:
                attr_node.value = self.visit(attr_node.value)
                rewritten = self.rewriter._rewrite_load(attr_node)
                if rewritten is None:
                    return attr_node
                return ast.copy_location(rewritten, attr_node)

        try:
            working = astutils.copy_tree(node)
        except Exception:
            working = node
        rewritten = _AttributeLoadRewriter(self).visit(working)
        return ast.fix_missing_locations(rewritten)

    def rewrite_assignment(self, target: ast.AST, value: ast.AST) -> Optional[ast.AST]:
        """Rewrite ``target = value`` when *target* is a special attribute write."""
        if not isinstance(target, ast.Attribute):
            return None

        base_value = try_resolve_static_value(target.value, self._evaluation_context())
        if base_value is UNRESOLVED or self._is_builtin_like_base(base_value):
            return None

        owner_expr = self._type_expr(astutils.copy_tree(target.value))
        obj_expr = astutils.copy_tree(target.value)
        objtype = type(base_value)
        rewritten_value = self.rewrite_expression(value)

        try:
            static_attr = inspect.getattr_static(base_value, target.attr)
        except AttributeError:
            static_attr = None

        if static_attr is not None and self._is_descriptor(static_attr) and hasattr(static_attr, '__set__'):
            descriptor_expr = self._descriptor_expr(astutils.copy_tree(target.value), target.attr)
            return ast.Call(func=ast.Attribute(value=descriptor_expr, attr='__set__', ctx=ast.Load()),
                            args=[obj_expr, rewritten_value],
                            keywords=[])

        setattr_method = objtype.__dict__.get('__setattr__')
        if setattr_method is not None and setattr_method is not object.__setattr__:
            return ast.Call(func=ast.Attribute(value=astutils.copy_tree(owner_expr), attr='__setattr__',
                                               ctx=ast.Load()),
                            args=[obj_expr, ast.Constant(target.attr), rewritten_value],
                            keywords=[])

        return None

    def _rewrite_load(self, node: ast.Attribute) -> Optional[ast.AST]:
        if not isinstance(node.ctx, ast.Load):
            return None

        base_value = try_resolve_static_value(node.value, self._evaluation_context())
        if base_value is UNRESOLVED or self._is_builtin_like_base(base_value):
            return None

        owner_expr = self._type_expr(astutils.copy_tree(node.value))
        obj_expr = astutils.copy_tree(node.value)
        objtype = type(base_value)

        try:
            static_attr = inspect.getattr_static(base_value, node.attr)
        except AttributeError:
            static_attr = None

        if static_attr is not None and self._is_descriptor(static_attr) and hasattr(static_attr, '__get__'):
            descriptor_expr = self._descriptor_expr(astutils.copy_tree(node.value), node.attr)
            return ast.Call(func=ast.Attribute(value=descriptor_expr, attr='__get__', ctx=ast.Load()),
                            args=[obj_expr, astutils.copy_tree(owner_expr)],
                            keywords=[])

        getattribute = objtype.__dict__.get('__getattribute__')
        if getattribute is not None and getattribute is not object.__getattribute__:
            return ast.Call(func=ast.Attribute(value=astutils.copy_tree(owner_expr),
                                               attr='__getattribute__',
                                               ctx=ast.Load()),
                            args=[obj_expr, ast.Constant(node.attr)],
                            keywords=[])

        if static_attr is None and '__getattr__' in objtype.__dict__:
            return ast.Call(func=ast.Attribute(value=astutils.copy_tree(owner_expr), attr='__getattr__',
                                               ctx=ast.Load()),
                            args=[obj_expr, ast.Constant(node.attr)],
                            keywords=[])

        return None

    @staticmethod
    def _type_expr(value_expr: ast.AST) -> ast.AST:
        return ast.Call(func=ast.Name(id='type', ctx=ast.Load()), args=[value_expr], keywords=[])

    def _descriptor_expr(self, value_expr: ast.AST, attr_name: str) -> ast.AST:
        return ast.Subscript(value=ast.Attribute(value=self._type_expr(value_expr), attr='__dict__', ctx=ast.Load()),
                             slice=ast.Constant(attr_name),
                             ctx=ast.Load())

    @staticmethod
    def _is_descriptor(value: Any) -> bool:
        return any(hasattr(value, attr) for attr in ('__get__', '__set__', '__delete__'))

    @staticmethod
    def _is_builtin_like_base(value: Any) -> bool:
        if dtypes.ismodule(value):
            return True
        if isinstance(value, (dtypes.typeclass, symbolic.symbol, symbolic.SymExpr, symbolic.sympy.Basic, data.Data)):
            return True
        module_name = getattr(type(value), '__module__', '')
        return module_name.startswith(('numpy', 'dace', 'sympy', 'builtins'))
