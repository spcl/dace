# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the alignment helpers ``symbolic.align`` and ``symbolic.is_multiple``."""

import pytest
import sympy

import dace
from dace import symbolic

N = dace.symbol("N")


@pytest.mark.parametrize(
    "value,alignment,expected", [(0, 16, 0), (1, 16, 16), (16, 16, 16), (17, 16, 32), (sympy.Integer(33), 8, 40)]
)
def test_align_constant(value, alignment, expected):
    result = symbolic.align(value, alignment)
    assert isinstance(result, int)
    assert result == expected


def test_align_symbolic():
    aligned = symbolic.align(4 * N, 16)
    assert aligned == symbolic.int_ceil(4 * N, 16) * 16
    assert aligned.subs(N, 5) == 32
    # An expression that is already a multiple needs no rounding
    assert symbolic.align(16 * N, 16) == 16 * N


@pytest.mark.parametrize(
    "value,alignment,expected",
    [
        (0, 16, True),
        (48, 16, True),
        (sympy.Integer(40), 16, False),
        (16 * N, 16, True),
        (32 * N + 16, 16, True),
        (4 * N, 16, False),
        (N, 1, True),
    ],
)
def test_is_multiple(value, alignment, expected):
    assert symbolic.is_multiple(value, alignment) is expected


if __name__ == "__main__":
    for args in [(0, 16, 0), (1, 16, 16), (16, 16, 16), (17, 16, 32), (sympy.Integer(33), 8, 40)]:
        test_align_constant(*args)
    test_align_symbolic()
    for args in [
        (0, 16, True),
        (48, 16, True),
        (sympy.Integer(40), 16, False),
        (16 * N, 16, True),
        (32 * N + 16, 16, True),
        (4 * N, 16, False),
        (N, 1, True),
    ]:
        test_is_multiple(*args)
