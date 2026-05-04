"""Helpers for lowering Python syntax sugar to explicit dunder calls."""

from __future__ import annotations

import ast
import builtins as pybuiltins
import inspect
import math
from typing import Dict, Optional, Tuple

from dace.frontend.python import astutils
from dace.frontend.python.schedule_tree.callable_support import CallableResolver
from dace.frontend.python.schedule_tree.static_evaluation import UNRESOLVED

_BINARY_DUNDERS: Dict[type[ast.operator], Tuple[str, str, Optional[str]]] = {
    ast.Add: ('__add__', '__radd__', '__iadd__'),
    ast.Sub: ('__sub__', '__rsub__', '__isub__'),
    ast.Mult: ('__mul__', '__rmul__', '__imul__'),
    ast.MatMult: ('__matmul__', '__rmatmul__', '__imatmul__'),
    ast.Div: ('__truediv__', '__rtruediv__', '__itruediv__'),
    ast.FloorDiv: ('__floordiv__', '__rfloordiv__', '__ifloordiv__'),
    ast.Mod: ('__mod__', '__rmod__', '__imod__'),
    ast.Pow: ('__pow__', '__rpow__', '__ipow__'),
    ast.LShift: ('__lshift__', '__rlshift__', '__ilshift__'),
    ast.RShift: ('__rshift__', '__rrshift__', '__irshift__'),
    ast.BitOr: ('__or__', '__ror__', '__ior__'),
    ast.BitXor: ('__xor__', '__rxor__', '__ixor__'),
    ast.BitAnd: ('__and__', '__rand__', '__iand__'),
}
_BINARY_DUNDERS = {key: value for key, value in _BINARY_DUNDERS.items() if value is not None}

_UNARY_DUNDERS: Dict[type[ast.unaryop], str] = {
    ast.UAdd: '__pos__',
    ast.USub: '__neg__',
    ast.Invert: '__invert__',
}

_COMPARE_DUNDERS: Dict[type[ast.cmpop], Tuple[Optional[str], Optional[str]]] = {
    ast.Eq: ('__eq__', '__eq__'),
    ast.NotEq: ('__ne__', '__ne__'),
    ast.Lt: ('__lt__', '__gt__'),
    ast.LtE: ('__le__', '__ge__'),
    ast.Gt: ('__gt__', '__lt__'),
    ast.GtE: ('__ge__', '__le__'),
}

_UNARY_CALL_DUNDERS = {
    pybuiltins.hash: '__hash__',
    pybuiltins.repr: '__repr__',
    pybuiltins.str: '__str__',
    pybuiltins.bool: '__bool__',
    pybuiltins.int: '__int__',
    pybuiltins.float: '__float__',
    pybuiltins.bytes: '__bytes__',
    pybuiltins.complex: '__complex__',
    pybuiltins.len: '__len__',
    pybuiltins.iter: '__iter__',
    pybuiltins.reversed: '__reversed__',
    pybuiltins.next: '__next__',
    pybuiltins.abs: '__abs__',
    pybuiltins.dir: '__dir__',
    math.trunc: '__trunc__',
    math.floor: '__floor__',
    math.ceil: '__ceil__',
}


def rewrite_sugared_expression(node: ast.AST, callable_resolver: CallableResolver) -> Optional[ast.AST]:
    """Return an explicit dunder-call AST for a sugared expression when possible."""
    if isinstance(node, ast.Call):
        return _rewrite_call(node, callable_resolver)

    if isinstance(node, ast.BinOp):
        return _rewrite_binop(node.left, node.op, node.right, callable_resolver)

    if isinstance(node, ast.UnaryOp):
        method_name = _UNARY_DUNDERS.get(type(node.op))
        if method_name is None:
            return None
        return _call_on_operand(node.operand, method_name, (), callable_resolver, template=node)

    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        return _rewrite_compare(node.left, node.ops[0], node.comparators[0], callable_resolver, node)

    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        return _rewrite_subscript(node, callable_resolver)

    return None


def rewrite_subscript_assignment(target: ast.Subscript, value: ast.AST,
                                 callable_resolver: CallableResolver) -> Optional[ast.Expr]:
    call = _call_on_operand(target.value, '__setitem__', (target.slice, value), callable_resolver, template=target)
    if call is None:
        return None
    return ast.copy_location(ast.Expr(value=call), target)


def rewrite_subscript_delete(target: ast.Subscript, callable_resolver: CallableResolver) -> Optional[ast.Expr]:
    call = _call_on_operand(target.value, '__delitem__', (target.slice, ), callable_resolver, template=target)
    if call is None:
        return None
    return ast.copy_location(ast.Expr(value=call), target)


def rewrite_augassign(target: ast.AST, op: ast.operator, value: ast.AST,
                      callable_resolver: CallableResolver) -> Optional[ast.stmt]:
    if isinstance(target, ast.Subscript):
        current_value = ast.copy_location(
            ast.Subscript(value=astutils.copy_tree(target.value),
                          slice=astutils.copy_tree(target.slice),
                          ctx=ast.Load()), target)
        getter_call = _rewrite_subscript(current_value, callable_resolver)
        updated = _rewrite_augassign_value(getter_call, op, value, template=target) if getter_call is not None else None
        if updated is None:
            updated = _rewrite_binop(current_value, op, value, callable_resolver, prefer_inplace=True)
        if updated is None:
            return None
        return rewrite_subscript_assignment(target, updated, callable_resolver)

    load_target = _load_context_copy(target)
    if load_target is None:
        return None

    updated = _rewrite_binop(load_target, op, value, callable_resolver, prefer_inplace=True)
    if updated is None:
        return None

    assign = ast.Assign(targets=[astutils.copy_tree(target)], value=updated)
    return ast.copy_location(assign, target)


def _rewrite_call(node: ast.Call, callable_resolver: CallableResolver) -> Optional[ast.Call]:
    if isinstance(node.func, ast.Attribute) and node.func.attr == '__call__':
        return None
    value = callable_resolver.resolve_static_value(node.func)
    if value is UNRESOLVED or not callable(value):
        return None

    builtin_rewritten = _rewrite_builtin_call(node, value, callable_resolver)
    if builtin_rewritten is not None:
        return builtin_rewritten

    from dace import SDFG
    from dace import data

    if isinstance(value, (SDFG, data.Data)) or inspect.isclass(value):
        return None
    if (inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value)
            or inspect.ismethoddescriptor(value) or getattr(value, '_schedule_tree_inline_callable', False)
            or hasattr(value, '__schedule_tree__') or hasattr(value, '__sdfg__')):
        return None
    return _call_on_operand(node.func,
                            '__call__',
                            tuple(node.args),
                            callable_resolver,
                            template=node,
                            keywords=node.keywords)


def _rewrite_builtin_call(node: ast.Call, builtin_value, callable_resolver: CallableResolver) -> Optional[ast.Call]:
    if builtin_value in _UNARY_CALL_DUNDERS and len(node.args) == 1 and not node.keywords:
        return _call_on_operand(node.args[0], _UNARY_CALL_DUNDERS[builtin_value], (), callable_resolver, template=node)

    if builtin_value is pybuiltins.format and not node.keywords and len(node.args) in {1, 2}:
        format_spec = ast.Constant(value='') if len(node.args) == 1 else node.args[1]
        return _call_on_operand(node.args[0], '__format__', (format_spec, ), callable_resolver, template=node)

    if builtin_value is pybuiltins.round and not node.keywords and len(node.args) in {1, 2}:
        return _call_on_operand(node.args[0], '__round__', tuple(node.args[1:]), callable_resolver, template=node)

    if builtin_value is pybuiltins.divmod and not node.keywords and len(node.args) == 2:
        rewritten = _call_on_operand(node.args[0], '__divmod__', (node.args[1], ), callable_resolver, template=node)
        if rewritten is not None:
            return rewritten
        return _call_on_operand(node.args[1], '__rdivmod__', (node.args[0], ), callable_resolver, template=node)

    if builtin_value is pybuiltins.isinstance and not node.keywords and len(node.args) == 2:
        return _call_on_operand(node.args[1], '__instancecheck__', (node.args[0], ), callable_resolver, template=node)

    if builtin_value is pybuiltins.issubclass and not node.keywords and len(node.args) == 2:
        return _call_on_operand(node.args[1], '__subclasscheck__', (node.args[0], ), callable_resolver, template=node)

    return None


def _rewrite_binop(left: ast.AST,
                   op: ast.operator,
                   right: ast.AST,
                   callable_resolver: CallableResolver,
                   *,
                   prefer_inplace: bool = False) -> Optional[ast.Call]:
    dunders = _BINARY_DUNDERS.get(type(op))
    if dunders is None:
        return None

    direct_name, reflected_name, inplace_name = dunders
    if prefer_inplace and inplace_name is not None:
        rewritten = _call_on_operand(left, inplace_name, (right, ), callable_resolver, template=left)
        if rewritten is not None:
            return rewritten

    rewritten = _call_on_operand(left, direct_name, (right, ), callable_resolver, template=left)
    if rewritten is not None:
        return rewritten

    return _call_on_operand(right, reflected_name, (left, ), callable_resolver, template=right)


def _rewrite_augassign_value(left: ast.AST, op: ast.operator, right: ast.AST, *,
                             template: ast.AST) -> Optional[ast.Call]:
    dunders = _BINARY_DUNDERS.get(type(op))
    if dunders is None:
        return None

    direct_name, _, inplace_name = dunders
    method_name = inplace_name or direct_name
    if method_name is None:
        return None
    return _build_method_call(left, method_name, (right, ), template=template)


def _rewrite_compare(left: ast.AST, op: ast.cmpop, right: ast.AST, callable_resolver: CallableResolver,
                     template: ast.AST) -> Optional[ast.AST]:
    if isinstance(op, ast.In):
        return _call_on_operand(right, '__contains__', (left, ), callable_resolver, template=template)
    if isinstance(op, ast.NotIn):
        contains = _call_on_operand(right, '__contains__', (left, ), callable_resolver, template=template)
        if contains is None:
            return None
        return ast.copy_location(ast.UnaryOp(op=ast.Not(), operand=contains), template)

    dunders = _COMPARE_DUNDERS.get(type(op))
    if dunders is None:
        return None

    direct_name, reflected_name = dunders
    if direct_name is not None:
        rewritten = _call_on_operand(left, direct_name, (right, ), callable_resolver, template=template)
        if rewritten is not None:
            return rewritten
    if reflected_name is not None:
        return _call_on_operand(right, reflected_name, (left, ), callable_resolver, template=template)
    return None


def _rewrite_subscript(node: ast.Subscript, callable_resolver: CallableResolver) -> Optional[ast.Call]:
    owner = _resolve_static_owner(node.value, callable_resolver)
    if owner is not UNRESOLVED and inspect.isclass(owner):
        rewritten = _call_on_operand(node.value, '__class_getitem__', (node.slice, ), callable_resolver, template=node)
        if rewritten is not None:
            return rewritten
    return _call_on_operand(node.value, '__getitem__', (node.slice, ), callable_resolver, template=node)


def _call_on_operand(operand: ast.AST,
                     method_name: str,
                     args: Tuple[ast.AST, ...],
                     callable_resolver: CallableResolver,
                     *,
                     template: ast.AST,
                     keywords: Optional[list[ast.keyword]] = None) -> Optional[ast.Call]:
    method_value = _resolve_dunder_method(operand, method_name, callable_resolver)
    if not _is_parseable_dunder_value(method_value):
        return None
    return _build_method_call(operand, method_name, args, template=template, keywords=keywords)


def _build_method_call(operand: ast.AST,
                       method_name: str,
                       args: Tuple[ast.AST, ...],
                       *,
                       template: ast.AST,
                       keywords: Optional[list[ast.keyword]] = None) -> ast.Call:
    method = ast.copy_location(ast.Attribute(value=astutils.copy_tree(operand), attr=method_name, ctx=ast.Load()),
                               template)
    return ast.copy_location(
        ast.Call(func=method,
                 args=[astutils.copy_tree(arg) for arg in args],
                 keywords=[astutils.copy_tree(keyword) for keyword in (keywords or [])]), template)


def _resolve_dunder_method(operand: ast.AST, method_name: str, callable_resolver: CallableResolver):
    owner = _resolve_static_owner(operand, callable_resolver)
    if owner is UNRESOLVED:
        return UNRESOLVED
    try:
        return getattr(owner, method_name)
    except Exception:
        return UNRESOLVED


def _resolve_static_owner(node: ast.AST, callable_resolver: CallableResolver):
    if isinstance(node, ast.Attribute):
        owner = _resolve_static_owner(node.value, callable_resolver)
        if owner is UNRESOLVED:
            return UNRESOLVED
        try:
            return getattr(owner, node.attr)
        except Exception:
            return UNRESOLVED
    return callable_resolver.resolve_static_value(node)


def _is_parseable_dunder_value(value) -> bool:
    if value is UNRESOLVED or not callable(value):
        return False
    if getattr(value, '_schedule_tree_inline_callable', False):
        return True
    if hasattr(value, '__schedule_tree__') or hasattr(value, '__sdfg__') or hasattr(value, '_generate_schedule_tree'):
        return True
    module_name = getattr(value.__func__ if inspect.ismethod(value) else value, '__module__', '')
    if isinstance(module_name, str) and module_name.startswith(('dace.frontend.python', 'sympy', 'numpy')):
        return False
    if inspect.ismethod(value) or inspect.isfunction(value):
        return True
    function = getattr(value, '__func__', None)
    if inspect.isfunction(function):
        return True
    wrapped = getattr(value, 'f', None)
    return callable(wrapped)


def _load_context_copy(target: ast.AST) -> Optional[ast.AST]:
    copied = astutils.copy_tree(target)
    if isinstance(copied, ast.Name):
        copied.ctx = ast.Load()
        return copied
    if isinstance(copied, ast.Attribute):
        copied.ctx = ast.Load()
        return copied
    if isinstance(copied, ast.Subscript):
        copied.ctx = ast.Load()
        return copied
    return None
