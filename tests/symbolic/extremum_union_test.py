# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``extremum_union``: the bound of a bounding-box union keeps only the arguments the facts cannot drop."""

import sympy

from dace import symbolic

X, Y = sympy.symbols("x y", integer=True)
NO_FACTS = symbolic.Facts.none()


def test_a_shifted_nested_minimum_is_dominated_by_its_own_arguments():
    """gromacs's unrolled copies: ``Min(2047, y, Min(2047, y) + 1)`` per copy grew every union by two arguments."""
    nested = sympy.Min(2047, Y, sympy.Min(2047, Y) + 1, evaluate=False)
    assert symbolic.extremum_union(sympy.Min, sympy.Min(2047, X), nested, NO_FACTS) == sympy.Min(2047, X, Y)


def test_a_provably_larger_argument_replaces_the_kept_one():
    assert symbolic.extremum_union(sympy.Max, X, X + 1, NO_FACTS) == X + 1
    assert symbolic.extremum_union(sympy.Min, 3, 5, NO_FACTS) == 3


def test_an_argument_already_kept_is_not_repeated():
    assert symbolic.extremum_union(sympy.Max, sympy.Max(X, Y), Y, NO_FACTS) == sympy.Max(X, Y)


def test_incomparable_arguments_are_all_kept():
    assert symbolic.extremum_union(sympy.Max, X, Y, NO_FACTS) == sympy.Max(X, Y)


if __name__ == "__main__":
    test_a_shifted_nested_minimum_is_dominated_by_its_own_arguments()
    test_a_provably_larger_argument_replaces_the_kept_one()
    test_an_argument_already_kept_is_not_repeated()
    test_incomparable_arguments_are_all_kept()
