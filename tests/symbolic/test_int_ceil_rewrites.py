# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``int_ceil`` rewrite rules.

``int_ceil`` used to fold only when BOTH operands were numbers, unlike ``int_floor`` which already
folded a unit denominator. That asymmetry leaked: ``strides_from_layout`` pads with the default
``alignment=1``, so every symbolic descriptor came back carrying an ``int_ceil(N, 1)`` that never
folded back to ``N``.
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


def test_exact_division_becomes_int_floor():
    """When the denominator divides the numerator there is nothing to round up."""
    assert int_ceil(4 * N, 4) == int_floor(4 * N, 4)
    assert int_ceil(8 * N + 16, 8) == int_floor(8 * N + 16, 8)


def test_inexact_division_is_left_symbolic():
    """A numerator that may leave a remainder must keep the ceiling."""
    assert int_ceil(N, 8).func is int_ceil
    assert int_ceil(4 * N + 2, 4).func is int_ceil
    # A symbolic denominator cannot be shown to divide, so it is left alone too.
    assert int_ceil(N, M).func is int_ceil


@pytest.mark.parametrize('x,y,expected', [(17, 8, 3), (16, 8, 2), (1, 8, 1), (0, 8, 0)])
def test_numeric_operands_fold(x, y, expected):
    assert int_ceil(x, y) == expected


if __name__ == '__main__':
    test_unit_denominator_folds_away()
    test_exact_division_becomes_int_floor()
    test_inexact_division_is_left_symbolic()
    for args in [(17, 8, 3), (16, 8, 2), (1, 8, 1), (0, 8, 0)]:
        test_numeric_operands_fold(*args)
