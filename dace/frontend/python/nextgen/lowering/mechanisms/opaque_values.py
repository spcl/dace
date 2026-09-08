# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Reporting for uses of opaque Python objects that no lowering can satisfy.

A callback whose return type is neither annotated nor inferable produces a
``pyobject`` container. That is the correct result -- the value really is an
arbitrary Python object -- and passing it along to further callbacks is fine:
it never leaves the interpreter.

What is not fine is asking the *generated program* to make sense of one. A
branch condition becomes C++, an operand becomes a tasklet's connector, and a
``pyobject`` is a ``void *`` there. Falling back to a callback does not help,
because the position needing the value is not itself a statement that could run
in the interpreter -- so the failure lands either in the C++ compiler
(``invalid operands of types 'double' and 'pyobject'``) or, worse, inside a
callback whose exception cannot cross the C ABI.

The frontend therefore refuses, and names the two ways out: annotate the
binding (``a: dace.float64[20] = callee(b)``) or give the callee a return type
hint. Refusals here are :class:`~dace.frontend.python.common.DaceSyntaxError`,
the class the stable frontend raises for the same programs.

Post-ANF every such use is a use of a *named* binding, so recognizing one is a
lookup of the name's descriptor rather than an analysis of the expression.
"""
import ast
from typing import List

from dace import dtypes
from dace.frontend.python.common import DaceSyntaxError
from dace.frontend.python.nextgen.lowering.registry import LoweringState


def opaque_names(expression: ast.expr, state: LoweringState) -> List[str]:
    """
    The source-level names read in ``expression`` that are bound to opaque
    Python objects.

    :param expression: The expression to inspect.
    :param state: The active lowering state.
    :return: The offending names, sorted, without duplicates.
    """
    return sorted({
        node.id
        for node in ast.walk(expression)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and _is_opaque(node.id, state)
    })


def sliced_opaque_names(expression: ast.expr, state: LoweringState) -> List[str]:
    """
    The source-level names in ``expression`` that are bound to opaque Python
    objects *and* subscripted (``a[:20]`` where ``a`` came back untyped).

    Separated from :func:`opaque_names` because an opaque value that is merely
    passed on can still travel to another callback, whereas a subscript of one
    cannot be given a shape or a dtype by any later step.

    :param expression: The expression to inspect.
    :param state: The active lowering state.
    :return: The offending names, sorted, without duplicates.
    """
    return sorted({
        node.value.id
        for node in ast.walk(expression)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and _is_opaque(node.value.id, state)
    })


def reject_opaque_condition(test: ast.expr, statement: ast.stmt, state: LoweringState) -> None:
    """
    Refuse a branch or loop condition that reads an opaque Python object.

    A condition is evaluated by the generated program, not by the interpreter,
    so it cannot be deferred to a callback the way an ordinary statement can.

    :param test: The condition expression.
    :param statement: The statement carrying it, for the error location.
    :param state: The active lowering state.
    :raises DaceSyntaxError: If the condition reads an opaque Python object.
    """
    names = opaque_names(test, state)
    if not names:
        return
    offending = _listed([state.context.describe_expression(name) for name in names])
    raise DaceSyntaxError(
        None, statement, f'Trying to operate on a callback with an unknown return type: the condition '
        f'"{state.context.describe_expression(test)}" reads {offending}, whose type could not be inferred. '
        f'A condition is evaluated by the compiled program, so it needs a concrete type. {_REMEDY}')


def reject_opaque_slice(expression: ast.expr, statement: ast.stmt, state: LoweringState) -> None:
    """
    Refuse a subscript of an opaque Python object.

    :param expression: The expression to inspect.
    :param statement: The statement carrying it, for the error location.
    :param state: The active lowering state.
    :raises DaceSyntaxError: If the expression subscripts an opaque Python object.
    """
    names = sliced_opaque_names(expression, state)
    if not names:
        return
    offending = _listed([state.context.describe_expression(name) for name in names])
    raise DaceSyntaxError(
        None, statement, f'A callback result with an unknown return type cannot be sliced: {offending} in '
        f'"{state.context.describe_expression(expression)}". Slicing needs a shape, and none could be inferred. '
        f'{_REMEDY}')


_REMEDY = ('Annotate the binding (for example "a: dace.float64[20] = callee(b)") or give the callee a return type '
           'hint, so the result has a type the compiled program can use.')


def _is_opaque(source_name: str, state: LoweringState) -> bool:
    """
    Whether a source-level name holds the result of a call whose return type
    could not be inferred.

    Deliberately narrower than "is a ``pyobject``". Plenty of opaque values are
    not a problem: a dict literal, or a binding the join merge could not
    reconcile, becomes a ``pyobject`` too, and a statement consuming one -- even
    subscripting it -- lowers to a callback that writes an ordinary container,
    which works. Those also have nothing the user could annotate. Only a call
    result carries both a real remedy and a real hazard, so only it is reported.
    """
    binding = state.context.resolve(source_name)
    if binding is None or binding.kind != 'container' or binding.container not in state.context.containers:
        return False
    if binding.container not in state.context.untyped_call_results:
        return False
    return isinstance(state.context.containers[binding.container].dtype, dtypes.pyobject)


def _listed(names: List[str]) -> str:
    """Render offending names for a message: ``"a"`` / ``"a" and "b"``."""
    quoted = [f'"{name}"' for name in names]
    if len(quoted) == 1:
        return quoted[0]
    return f'{", ".join(quoted[:-1])} and {quoted[-1]}'
