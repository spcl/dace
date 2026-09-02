# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SymExpr`` floor division builds ``int_floor``.

``SymExpr.__floordiv__`` used to apply Python ``//`` to its sympy operands, which sympy evaluates
as ``floor(a / b)`` over rationals: ``SymExpr(N - 1, N) // 8`` became ``floor(N/8 - 1/8)``. That is
the wrong tree -- a pass matching ``int_floor``, or any ``subs`` or equality against the integer
form, no longer sees it -- and the split rationals truncate term by term in C++.
"""
import pytest
import sympy

from dace.symbolic import SymExpr, symbol

N = symbol('N')
M = symbol('M')


@pytest.mark.parametrize('divisor', [8, '8', M, SymExpr(M - 1, M)], ids=['int', 'str', 'symbolic', 'symexpr'])
def test_floordiv_builds_int_floor(divisor):
    """Every operand kind takes the integer-division form, on both the main and approx expression."""
    result = SymExpr(N - 1, N) // divisor
    assert result.expr.func.__name__ == 'int_floor'
    assert result.approx.func.__name__ == 'int_floor'


@pytest.mark.parametrize('divisor', [8, '8', M, SymExpr(M - 1, M)], ids=['int', 'str', 'symbolic', 'symexpr'])
def test_floordiv_leaves_no_sympy_floor(divisor):
    """No ``floor`` over rationals survives, on either expression."""
    result = SymExpr(N - 1, N) // divisor
    assert not result.expr.has(sympy.floor)
    assert not result.approx.has(sympy.floor)


def test_floordiv_still_folds_an_exact_division():
    """``int_floor`` folding is unchanged: an exact division is not a rounding operation."""
    result = SymExpr(4 * N, 8 * N) // 4
    assert result.expr == N
    assert result.approx == 2 * N


if __name__ == '__main__':
    for d in (8, '8', M, SymExpr(M - 1, M)):
        test_floordiv_builds_int_floor(d)
        test_floordiv_leaves_no_sympy_floor(d)
    test_floordiv_still_folds_an_exact_division()
