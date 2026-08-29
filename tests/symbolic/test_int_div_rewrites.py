# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``int_floor`` / ``int_ceil`` rewrite rules.

``int_ceil`` used to fold only when BOTH operands were numbers, unlike ``int_floor`` which already
folded a unit denominator. That asymmetry leaked: ``strides_from_layout`` pads with the default
``alignment=1``, so every symbolic descriptor came back carrying an ``int_ceil(N, 1)`` that never
folded back to ``N``. Both functions now share the unit-denominator and exact-division rules.
"""
import pytest
import sympy

from dace import dtypes
from dace.symbolic import (int_ceil, int_floor, pystr_to_symbolic, symbol, symstr, sympy_intdiv_fix)

N = pystr_to_symbolic('N')
M = pystr_to_symbolic('M')


def test_unit_denominator_folds_away():
    """Dividing by 1 rounds nothing up, and int_floor already folds it."""
    assert int_ceil(N, 1) == N
    assert int_floor(N, 1) == N
    assert int_ceil(N * M + 3, 1) == N * M + 3


@pytest.mark.parametrize('fn', [int_floor, int_ceil])
def test_exact_division_yields_the_quotient(fn):
    """Exact division is not a rounding operation, so neither node should survive it.

    Returning the quotient rather than the other rounding function keeps the result comparable and
    simplifiable -- ``int_floor(4*N, 4)`` and ``N`` are the same number and should compare equal.
    """
    assert fn(4 * N, 4) == N
    assert fn(8 * N + 16, 8) == N + 2
    assert fn(12 * N + 6, 3) == 4 * N + 2


@pytest.mark.parametrize('fn', [int_floor, int_ceil])
def test_inexact_division_is_left_symbolic(fn):
    """A numerator that may leave a remainder must keep the rounding node."""
    assert fn(N, 8).func is fn
    assert fn(4 * N + 2, 4).func is fn
    # A symbolic denominator cannot be shown to divide, so it is left alone too.
    assert fn(N, M).func is fn


@pytest.mark.parametrize('rounding', [sympy.floor, sympy.ceiling])
def test_rounding_of_a_non_integer_expression_is_kept(rounding):
    """A rounding call with nothing to divide by is not an integer division.

    ``floor(e / b)`` used to match with ``b = 1`` for any non-atomic argument, so the rounding was
    rewritten to ``int_floor(arg, 1)`` and printed as a division by one -- silently dropped
    (spcl/dace#2524).
    """
    x = pystr_to_symbolic('x')
    expr = rounding(sympy.sin(x))
    assert sympy_intdiv_fix(expr) == expr


@pytest.mark.parametrize('rounding,call', [(sympy.floor, 'floor'), (sympy.ceiling, 'ceil')])
def test_rounding_of_a_non_integer_expression_lowers_to_the_math_call(rounding, call):
    """The kept rounding must reach C++ as the matching math-library call.

    ``ceil`` rather than the sympy name: the runtime's one-argument ``ceiling`` overload returns
    ``DBL_MAX`` for a floating argument, so the bare name would emit a poison value.
    """
    x = pystr_to_symbolic('x')
    assert symstr(rounding(sympy.sin(x)), cpp_mode=True) == '(%s(sin(x)))' % call


def test_ceiling_of_index_arithmetic_is_integer_typed():
    """A ceiling over integer symbols is an extent, and must reach C++ with an integer type.

    ``ceil`` returns a double. As a map's range end that makes the emitted loop compare an integer
    induction variable against a double, which is not OpenMP's canonical form, and the compiler
    rejects the enclosing ``omp for`` outright with "invalid controlling predicate" -- npbench's
    ``stockham_fft``, whose radix extent is ``R ** (K - 1)``.
    """
    r = symbol('R', dtype=dtypes.int64)
    k = symbol('K', dtype=dtypes.int64)
    emitted = symstr(sympy.ceiling(r**(k - 1)), cpp_mode=True)
    assert '(int64_t)' in emitted, emitted
    assert 'ceil(' in emitted, emitted


def test_ceiling_of_a_float_valued_call_on_integer_symbols_stays_floating():
    """Integrality is decided by the OPERATIONS, not the operands.

    Every atom of ``sin(n)`` is an integer symbol and the value is a double, so reading the atoms
    would type this one as an index and truncate it.
    """
    n = symbol('n', dtype=dtypes.int64)
    assert symstr(sympy.ceiling(sympy.sin(n)), cpp_mode=True) == '(ceil(sin(n)))'


@pytest.mark.parametrize('rounding,name', [(sympy.floor, 'int_floor'), (sympy.ceiling, 'int_ceil')])
def test_division_by_an_integer_still_rewrites(rounding, name):
    """The integer-division rewrite is unchanged where a real denominator is present."""
    assert sympy_intdiv_fix(rounding(N / 8)).func.__name__ == name


@pytest.mark.parametrize('x,y,expected', [(17, 8, 3), (16, 8, 2), (1, 8, 1), (0, 8, 0)])
def test_numeric_operands_fold(x, y, expected):
    assert int_ceil(x, y) == expected


if __name__ == '__main__':
    test_unit_denominator_folds_away()
    for fn in (int_floor, int_ceil):
        test_exact_division_yields_the_quotient(fn)
        test_inexact_division_is_left_symbolic(fn)
    for args in [(17, 8, 3), (16, 8, 2), (1, 8, 1), (0, 8, 0)]:
        test_numeric_operands_fold(*args)
    test_ceiling_of_index_arithmetic_is_integer_typed()
    test_ceiling_of_a_float_valued_call_on_integer_symbols_stays_floating()
