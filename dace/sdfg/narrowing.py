# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Narrowing of DaCe's loosely typed unions to the concrete type extended-only code works with.

``symbolic.SymbolicType`` is ``sympy.Basic | SymExpr`` and subset bounds are additionally Python ints, so
arithmetic, ``free_symbols`` and ``is_number`` on them do not type-check; ``Memlet.subset`` is
``Subset | None`` while most passes only handle a ``Range``. Extended-only code narrows once, here, instead of
probing at every call site.
"""
from typing import Union

import sympy

from dace import dtypes, subsets, symbolic

#: Anything a subset bound, loop bound or parsed expression can be.
SymbolicLike = Union[symbolic.SymbolicType, int, str]


def as_expr(value: SymbolicLike) -> sympy.Expr:
    """``value`` as a sympy ``Expr``; a :class:`~dace.symbolic.SymExpr` contributes its main expression.

    :param value: A symbolic expression, Python int or expression string.
    :returns: The sympy expression.
    :raises TypeError: If ``value`` parses to a non-``Expr`` sympy object (a relational, for example).
    """
    if isinstance(value, symbolic.SymExpr):
        value = value.expr
    elif isinstance(value, (int, str)):
        value = symbolic.pystr_to_symbolic(value)
    if not isinstance(value, sympy.Expr):
        raise TypeError(f'{value!r} is not a sympy expression')
    return value


def simplified(value: SymbolicLike) -> sympy.Expr:
    """``symbolic.simplify`` of ``value``, narrowed to ``sympy.Expr``."""
    return as_expr(symbolic.simplify(as_expr(value)))


def as_range(subset: subsets.Subset | None) -> subsets.Range:
    """``subset`` as a :class:`~dace.subsets.Range` (an ``Indices`` is one).

    :param subset: A memlet subset.
    :returns: The same object.
    :raises TypeError: If ``subset`` is ``None`` or a ``SubsetUnion``.
    """
    if not isinstance(subset, subsets.Range):
        raise TypeError(f'expected a Range subset, got {type(subset).__name__}')
    return subset


def as_typeclass(dtype: object) -> dtypes.typeclass:
    """``dtype`` as a :class:`~dace.dtypes.typeclass`.

    ``dace.int64`` and friends are typeclass instances at run time but are declared as array classes for
    annotation syntax (``dace.float64[N]``), so a type checker rejects them where a typeclass is expected.

    :param dtype: ``dace.int64``, ``dace.float32``, ... or any typeclass.
    :returns: The same object.
    :raises TypeError: If ``dtype`` is not a typeclass.
    """
    if not isinstance(dtype, dtypes.typeclass):
        raise TypeError(f'expected a dace typeclass, got {dtype!r}')
    return dtype


def free_symbols(expr: sympy.Expr) -> set[sympy.Symbol]:
    """Free symbols of ``expr`` narrowed from sympy's ``set[Basic]`` to ``set[Symbol]``.

    :param expr: A sympy expression.
    :returns: Its free symbols.
    :raises TypeError: If a free symbol is not a ``sympy.Symbol`` (an ``Indexed`` object, for example).
    """
    result: set[sympy.Symbol] = set()
    for s in expr.free_symbols:
        if not isinstance(s, sympy.Symbol):
            raise TypeError(f'free symbol {s!r} of {expr} is not a sympy.Symbol')
        result.add(s)
    return result


def coeff_of(expr: sympy.Expr, symbol: sympy.Expr, power: int = 1) -> sympy.Expr:
    """``expr.coeff(symbol, power)`` narrowed from sympy's ``Expr | None`` (it returns ``0`` for no match)."""
    coeff = expr.coeff(symbol, power)
    if coeff is None:
        raise TypeError(f'{expr} has no coefficient for {symbol}**{power}')
    return coeff


def ndrange_exprs(subset: subsets.Subset) -> list[tuple[sympy.Expr, sympy.Expr, sympy.Expr]]:
    """``subset.ndrange()`` with every ``(begin, end, step)`` entry narrowed to a sympy ``Expr``."""
    return [(as_expr(begin), as_expr(end), as_expr(step)) for begin, end, step in subset.ndrange()]
