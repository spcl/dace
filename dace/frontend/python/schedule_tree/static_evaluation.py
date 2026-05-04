# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Static AST evaluation helpers for schedule-tree inference.

These helpers resolve a narrow subset of Python AST without executing user code.
They are intended for parser-time metadata recovery such as shapes, dtypes, and
simple literal/container reasoning.
"""

import ast
import inspect
import operator
import types
from typing import Any, Dict

import numpy as np

from dace import data
from dace.frontend.python import astutils

UNRESOLVED = object()

_SAFE_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}
_SAFE_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}
_SAFE_COMPARE_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}


def try_resolve_static_value(node: ast.AST, env: Dict[str, Any]) -> Any:
    """Resolve a static value from ``node`` or return ``UNRESOLVED``."""
    return _StaticValueResolver(env).resolve(node)


class _StaticValueResolver:

    def __init__(self, env: Dict[str, Any]) -> None:
        self.env = env

    def resolve(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            return self.env.get(node.id, UNRESOLVED)

        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value)
            if base is UNRESOLVED:
                return UNRESOLVED
            if _supports_simple_object_attribute_lookup(base):
                return _resolve_simple_object_attribute(base, node.attr)
            if not _supports_attribute_lookup(base):
                return UNRESOLVED
            try:
                return getattr(base, node.attr)
            except Exception:
                return UNRESOLVED

        if isinstance(node, ast.Tuple):
            values = [self.resolve(element) for element in node.elts]
            if any(value is UNRESOLVED for value in values):
                return UNRESOLVED
            return tuple(values)

        if isinstance(node, ast.List):
            values = [self.resolve(element) for element in node.elts]
            if any(value is UNRESOLVED for value in values):
                return UNRESOLVED
            return values

        if isinstance(node, ast.Set):
            values = [self.resolve(element) for element in node.elts]
            if any(value is UNRESOLVED for value in values):
                return UNRESOLVED
            try:
                return set(values)
            except TypeError:
                return UNRESOLVED

        if isinstance(node, ast.Dict):
            resolved = {}
            for key_node, value_node in zip(node.keys, node.values):
                if key_node is None:
                    return UNRESOLVED
                key = self.resolve(key_node)
                value = self.resolve(value_node)
                if key is UNRESOLVED or value is UNRESOLVED:
                    return UNRESOLVED
                try:
                    resolved[key] = value
                except TypeError:
                    return UNRESOLVED
            return resolved

        if isinstance(node, ast.Slice):
            lower = None if node.lower is None else self.resolve(node.lower)
            upper = None if node.upper is None else self.resolve(node.upper)
            step = None if node.step is None else self.resolve(node.step)
            if lower is UNRESOLVED or upper is UNRESOLVED or step is UNRESOLVED:
                return UNRESOLVED
            return slice(lower, upper, step)

        if isinstance(node, ast.Subscript):
            base = self.resolve(node.value)
            index = self.resolve(node.slice)
            if base is UNRESOLVED or index is UNRESOLVED or not _supports_subscript(base):
                return UNRESOLVED
            try:
                return base[index]
            except Exception:
                return UNRESOLVED

        if isinstance(node, ast.UnaryOp):
            operand = self.resolve(node.operand)
            if operand is UNRESOLVED:
                return UNRESOLVED
            operator_fn = _SAFE_UNARY_OPERATORS.get(type(node.op))
            if operator_fn is None:
                return UNRESOLVED
            try:
                return operator_fn(operand)
            except Exception:
                return UNRESOLVED

        if isinstance(node, ast.BinOp):
            left = self.resolve(node.left)
            right = self.resolve(node.right)
            if left is UNRESOLVED or right is UNRESOLVED:
                return UNRESOLVED
            operator_fn = _SAFE_BINARY_OPERATORS.get(type(node.op))
            if operator_fn is None:
                return UNRESOLVED
            try:
                return operator_fn(left, right)
            except Exception:
                return UNRESOLVED

        if isinstance(node, ast.BoolOp):
            values = [self.resolve(value) for value in node.values]
            if any(value is UNRESOLVED for value in values):
                return UNRESOLVED
            if isinstance(node.op, ast.And):
                result = values[0]
                for value in values[1:]:
                    result = result and value
                return result
            if isinstance(node.op, ast.Or):
                result = values[0]
                for value in values[1:]:
                    result = result or value
                return result
            return UNRESOLVED

        if isinstance(node, ast.Compare):
            left = self.resolve(node.left)
            if left is UNRESOLVED:
                return UNRESOLVED
            current = left
            for op, comparator_node in zip(node.ops, node.comparators):
                right = self.resolve(comparator_node)
                if right is UNRESOLVED:
                    return UNRESOLVED
                operator_fn = _SAFE_COMPARE_OPERATORS.get(type(op))
                if operator_fn is None:
                    return UNRESOLVED
                try:
                    if not operator_fn(current, right):
                        return False
                except Exception:
                    return UNRESOLVED
                current = right
            return True

        if isinstance(node, ast.IfExp):
            condition = self.resolve(node.test)
            if condition is UNRESOLVED:
                return UNRESOLVED
            branch = node.body if condition else node.orelse
            return self.resolve(branch)

        if isinstance(node, ast.Call):
            return self._resolve_builtin_container_call(node)

        return UNRESOLVED

    def _resolve_builtin_container_call(self, node: ast.Call) -> Any:
        if node.keywords:
            return UNRESOLVED

        call_name = astutils.rname(node.func)
        if call_name not in {'tuple', 'list'}:
            return UNRESOLVED

        args = [self.resolve(arg) for arg in node.args]
        if any(arg is UNRESOLVED for arg in args):
            return UNRESOLVED

        if call_name == 'tuple':
            if len(args) == 0:
                return tuple()
            if len(args) != 1:
                return UNRESOLVED
            try:
                return tuple(args[0])
            except TypeError:
                return UNRESOLVED

        if len(args) == 0:
            return []
        if len(args) != 1:
            return UNRESOLVED
        try:
            return list(args[0])
        except TypeError:
            return UNRESOLVED


def _supports_attribute_lookup(value: Any) -> bool:
    if isinstance(value, (types.ModuleType, type, data.Data, np.ndarray, np.generic)):
        return True
    return type(value).__module__.startswith('numpy')


def _supports_simple_object_attribute_lookup(value: Any) -> bool:
    if _supports_attribute_lookup(value):
        return False

    objtype = type(value)
    getattribute = objtype.__dict__.get('__getattribute__')
    if getattribute is not None and getattribute is not object.__getattribute__:
        return False
    if '__getattr__' in objtype.__dict__:
        return False
    return True


def _resolve_simple_object_attribute(value: Any, attr_name: str) -> Any:
    try:
        static_attr = inspect.getattr_static(value, attr_name)
    except AttributeError:
        return UNRESOLVED

    if any(hasattr(static_attr, attr) for attr in ('__get__', '__set__', '__delete__')):
        return UNRESOLVED
    return static_attr


def _supports_subscript(value: Any) -> bool:
    return isinstance(value, (list, tuple, str, bytes, dict, np.ndarray))
