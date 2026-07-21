# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``int_floor`` / ``int_ceil`` rewrite rules.

``int_ceil`` used to fold only when BOTH operands were numbers, unlike ``int_floor`` which already
folded a unit denominator. That asymmetry leaked: ``strides_from_layout`` pads with the default
``alignment=1``, so every symbolic descriptor came back carrying an ``int_ceil(N, 1)`` that never
folded back to ``N``. Both functions now share the unit-denominator and exact-division rules.
"""
import pytest

from dace.symbolic import int_ceil, int_floor, pystr_to_symbolic

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
