# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Narrowing of DaCe's loosely typed unions to the concrete type extended-only code works with.

``symbolic.SymbolicType`` is ``sympy.Basic | SymExpr`` and subset bounds are additionally Python ints, so
arithmetic, ``free_symbols`` and ``is_number`` on them do not type-check; ``Memlet.subset`` is
``Subset | None`` while most passes only handle a ``Range``. Extended-only code narrows once, here, instead of
probing at every call site.
"""

import sympy

from dace import subsets, symbolic
from dace.sdfg import nodes

#: Anything a subset bound, loop bound or parsed expression can be (``sympy.Basic`` also covers sympy's own stubs,
#: which declare ``Basic`` where an ``Expr`` is returned).
SymbolicLike = sympy.Basic | symbolic.SymExpr | int | float | str


def as_expr(value: SymbolicLike) -> sympy.Expr:
    """``value`` as a sympy ``Expr``; a :class:`~dace.symbolic.SymExpr` contributes its main expression.

    :param value: A symbolic expression, Python number or expression string.
    :returns: The sympy expression.
    :raises TypeError: If ``value`` parses to a non-``Expr`` sympy object (a relational, for example).
    """
    expr = value.expr if isinstance(value, symbolic.SymExpr) else value
    if isinstance(expr, (int, float, str)):
        expr = symbolic.pystr_to_symbolic(expr)
    if not isinstance(expr, sympy.Expr):
        raise TypeError(f"{expr!r} is not a sympy expression")
    return expr


def as_basic(value: SymbolicLike) -> sympy.Basic:
    """``value`` as a sympy ``Basic``; a :class:`~dace.symbolic.SymExpr` contributes its main expression.

    Unlike :func:`as_expr` this keeps relationals and booleans (``a < b``, ``Eq(a, b)``, ``True``), which carry
    ``free_symbols``, ``args`` and ``atoms`` but are no ``Expr``.

    :param value: A symbolic expression, boolean expression, Python number or expression string.
    :returns: The sympy object.
    :raises TypeError: If ``value`` is none of those.
    """
    expr = value.expr if isinstance(value, symbolic.SymExpr) else value
    if isinstance(expr, (int, float, str)):
        expr = symbolic.pystr_to_symbolic(expr)
    if not isinstance(expr, sympy.Basic):
        raise TypeError(f"{expr!r} is not a sympy object")
    return expr


def simplified(value: SymbolicLike) -> sympy.Expr:
    """``symbolic.simplify`` of ``value``, narrowed to ``sympy.Expr``."""
    return as_expr(symbolic.simplify(as_expr(value)))


def as_range(subset: subsets.Subset | None) -> subsets.Range:
    """``subset`` as a :class:`~dace.subsets.Range`.

    :param subset: A memlet subset.
    :returns: The same object.
    :raises TypeError: If ``subset`` is ``None`` or a ``SubsetUnion``.
    """
    if not isinstance(subset, subsets.Range):
        raise TypeError(f"expected a Range subset, got {type(subset).__name__}")
    return subset


def as_map_entry(node: nodes.Node | None) -> nodes.MapEntry:
    """``node`` as a :class:`~dace.sdfg.nodes.MapEntry`; scope lookups return the abstract ``EntryNode | None``."""
    if not isinstance(node, nodes.MapEntry):
        raise TypeError(f"expected a MapEntry, got {type(node).__name__}")
    return node


def free_symbols(expr: sympy.Basic) -> set[sympy.Symbol]:
    """Free symbols of ``expr`` narrowed from sympy's ``set[Basic]`` to ``set[Symbol]``.

    :param expr: A sympy expression, relational or boolean.
    :returns: Its free symbols.
    :raises TypeError: If a free symbol is not a ``sympy.Symbol`` (an ``Indexed`` object, for example).
    """
    result: set[sympy.Symbol] = set()
    for s in expr.free_symbols:
        if not isinstance(s, sympy.Symbol):
            raise TypeError(f"free symbol {s!r} of {expr} is not a sympy.Symbol")
        result.add(s)
    return result


def free_symbol_names(value: symbolic.SymbolicType | int | float) -> set[str]:
    """Names of the free symbols of a subset bound or stride; a Python number has none.

    A :class:`~dace.symbolic.SymExpr` contributes the symbols of its main expression.
    """
    expr = value.expr if isinstance(value, symbolic.SymExpr) else value
    if isinstance(expr, sympy.Basic):
        return {str(s) for s in expr.free_symbols}
    return set()
