# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Compile-time evaluation of ``for``-loop iterator expressions.

A ``for`` header names an object to iterate over. Most of them are runtime
values the frontend cannot see (a list, a generator, an array), but the DaCe
iteration constructs — ``dace.map[...]`` today, ``dace.consume(...)`` later —
are ordinary Python objects living in the program's closure, and the operators
written on them are ordinary Python operators on those objects::

    for i, j in dace.map[0:N, 0:M] @ dace.ScheduleType.GPU_Device:

This module evaluates such headers *by running them*: names resolve through
the program globals to the real objects, and subscripts and operators are
applied by invoking the real dunder methods
(:meth:`~dace.frontend.python.interface.MapGenerator.__matmul__` and friends).
The frontends therefore hold no syntactic knowledge of which operators an
iteration construct supports or what they mean — giving
:class:`~dace.frontend.python.interface.MapGenerator` a new operator, or
registering a new construct in :func:`construct_types`, is enough for it to
work in loop preprocessing, in the canonical grammar, and in lowering.

Evaluation happens in a mixed domain: everything is a real Python object
except subscript indices, which stay ASTs. ``dace.map[0:N]`` has bounds that
only lowering can interpret (symbols, data-dependent reads), and evaluating
them here would both fail and throw away the information lowering needs.
Iteration constructs are written to accept that: ``MapGenerator`` stores its
range without inspecting it.

Evaluation is guarded on both ends — the expression is only evaluated while it
is rooted in a registered construct, its leaves are names and literals so that
resolving a header never *calls* anything, and any failure yields "not an
iteration construct" rather than an error — so a header this module declines is
simply handled elsewhere (an ordinary Python iterator, which becomes a
callback).
"""
import ast
import operator
from typing import Any, Callable, Dict, Optional, Tuple

from dace.frontend.python import astutils

#: Sentinel for an expression that could not be evaluated at compile time.
UNRESOLVED = object()

#: Every binary AST operator, mapped to the function invoking its dunder.
BINARY_OPERATORS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.MatMult: operator.matmul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
}

#: Every unary AST operator that dispatches to a dunder (``not`` does not).
UNARY_OPERATORS: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}


def construct_types() -> Tuple[type, ...]:
    """
    The compile-time objects a ``for`` loop may iterate over in an SDFG scope.

    Registering a type here — together with the class that produces it, in
    :func:`construct_factories`, if it is made by subscripting one — is the
    entire interface between the frontend and an iteration construct. What
    operators the construct supports, and what they do to it, is the
    construct's own business.
    """
    from dace.frontend.python import interface  # Deferred to avoid an import cycle during package initialization
    return (interface.MapGenerator, )


def construct_factories() -> Tuple[type, ...]:
    """Classes whose subscript produces an iteration construct (``dace.map[...]``)."""
    from dace.frontend.python import interface  # Deferred to avoid an import cycle during package initialization
    return (interface.map, )


def is_construct(value: Any) -> bool:
    """Whether a value is an iteration construct, or the class that produces one."""
    if value is UNRESOLVED:
        return False
    return isinstance(value, construct_types()) or any(value is factory for factory in construct_factories())


def iteration_object(node: ast.expr, global_vars: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """
    The iteration construct a ``for``-loop header denotes, or None if the
    header is not one (an ordinary Python iterator, or an expression that is
    not resolvable at parse time).

    :param node: The iterator expression (``For.iter``).
    :param global_vars: The program's global namespace. When omitted, only
        headers rooted in the ``dace`` module itself resolve.
    """
    value = evaluate(node, global_vars if global_vars is not None else {})
    return value if isinstance(value, construct_types()) else None


def evaluate(node: ast.expr, global_vars: Dict[str, Any]) -> Any:
    """
    Evaluate an iterator expression in the mixed object/AST domain described
    in the module docstring, returning :data:`UNRESOLVED` if it is not a
    compile-time expression over an iteration construct.
    """
    if isinstance(node, ast.Subscript):
        base = evaluate(node.value, global_vars)
        if not is_construct(base):
            return UNRESOLVED
        # The index deliberately stays an AST: see the module docstring.
        return _apply(operator.getitem, base, node.slice)
    if isinstance(node, ast.BinOp):
        left = evaluate(node.left, global_vars)
        function = BINARY_OPERATORS.get(type(node.op))
        if function is None or not is_construct(left):
            return UNRESOLVED
        right = evaluate(node.right, global_vars)
        if right is UNRESOLVED:
            return UNRESOLVED
        return _apply(function, left, right)
    if isinstance(node, ast.UnaryOp):
        operand = evaluate(node.operand, global_vars)
        function = UNARY_OPERATORS.get(type(node.op))
        if function is None or not is_construct(operand):
            return UNRESOLVED
        return _apply(function, operand)
    return _closure_value(node, global_vars)


def _apply(function: Callable[..., Any], *arguments: Any) -> Any:
    """
    Invoke a construct's dunder. A construct that rejects the operand it was
    given (``dace.map[0:N] @ "not a schedule"``) makes the header not an
    iteration construct, which is reported as such downstream rather than as
    an error from inside the evaluator.
    """
    try:
        return function(*arguments)
    except Exception:
        return UNRESOLVED


def _closure_value(node: ast.expr, global_vars: Dict[str, Any]) -> Any:
    """
    Evaluate a leaf of the header (the construct's root, or an operand such as
    ``dace.ScheduleType.GPU_Device``) against the program's closure.

    Leaves are restricted to names, attribute chains and literals: resolving a
    header must never *call* anything, since a header that turns out not to be
    an iteration construct is one this module was only asking about.
    """
    if not isinstance(node, (ast.Name, ast.Attribute, ast.Constant)):
        return UNRESOLVED
    try:
        return astutils.evalnode(node, global_vars)
    except SyntaxError:
        pass
    # Preprocessing rewrites aliased module imports to the real module name,
    # which may be absent from the caller's globals.
    if 'dace' not in global_vars:
        import dace  # Deferred to avoid an import cycle during package initialization
        try:
            return astutils.evalnode(node, {**global_vars, 'dace': dace})
        except SyntaxError:
            pass
    return UNRESOLVED
