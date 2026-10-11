# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Bounds :func:`dace.symbolic.ask` proves through ``Min``, ``Max`` and the rounding heads, which sympy's
assumption engine does not bound: a cost model reading a peeled one-trip loop as "maybe big" pins every map
in its body sequential."""

import dace
from dace import symbolic


def test_min_bounds_a_peeled_first_trip():
    """``14:Min(14, klev - 1) + 1`` is at most one iteration, whatever ``klev`` is."""
    count = symbolic.pystr_to_symbolic("Min(14, klev - 1) - 13")
    assert symbolic.ask("negative", count - 256) is True
    assert symbolic.numeric_bounds(count)[1] == 1
    assert symbolic.ask("nonnegative", count) is None, "klev < 14 leaves the loop empty or negative"


def test_max_bounds_from_below():
    assert symbolic.ask("positive", "Max(3, N) - 2") is True
    assert symbolic.ask("negative", "-Max(3, N) + 2") is True


def test_int_ceil_is_at_most_one_above_the_quotient():
    assert symbolic.ask("nonnegative", "N/64 + 1 - int_ceil(N, 64)") is True
    assert symbolic.ask("nonnegative", "int_ceil(N, 64) - N/64") is True


def test_int_floor_is_at_most_the_quotient():
    assert symbolic.ask("nonnegative", "N/8 - int_floor(N, 8)") is True
    assert symbolic.ask("positive", "int_floor(N, 8) - N/8") is False


def test_floor_ceiling_and_mod():
    assert symbolic.ask("nonnegative", "ceiling(N/3) - N/3") is True
    assert symbolic.ask("nonnegative", "N/3 - floor(N/3)") is True
    assert symbolic.ask("nonnegative", "Mod(N, 8)") is True
    assert symbolic.ask("positive", "Mod(N, 8) - 8") is False


def test_an_unbounded_symbol_stays_undecided():
    assert symbolic.ask("negative", "N - 3") is None
    assert symbolic.numeric_bounds(symbolic.pystr_to_symbolic("N + Min(N, 4)")) == (None, None)


def test_int_floor_and_int_ceil_know_their_sign():
    n = dace.symbol("n", positive=True, integer=True)
    assert symbolic.int_floor(n, 4).is_nonnegative is True
    assert symbolic.int_ceil(n, 4).is_positive is True


def test_divide_fix_writes_dace_int_floor():
    """The printing fix that turns ``tid * (1/2)`` into a floor division mints DaCe's ``int_floor``, which
    carries integrality and sign, not an undefined function of the same name."""
    fixed = symbolic.sympy_divide_fix(symbolic.pystr_to_symbolic("tid/2"))
    assert isinstance(fixed, symbolic.int_floor), type(fixed)
    assert fixed.is_integer is True


def test_an_undefined_symbol_has_no_bounds():
    """An expression over an ``UndefinedSymbol`` is no number; ``expand`` around one never returns."""
    undefined = symbolic.UndefinedSymbol()
    assert symbolic.numeric_bounds(undefined * 5 + 1) == (None, None)
    assert symbolic.ask("nonnegative", undefined * 5 + 1) is None


if __name__ == "__main__":
    test_min_bounds_a_peeled_first_trip()
    test_max_bounds_from_below()
    test_int_ceil_is_at_most_one_above_the_quotient()
    test_int_floor_is_at_most_the_quotient()
    test_floor_ceiling_and_mod()
    test_an_unbounded_symbol_stays_undecided()
    test_int_floor_and_int_ceil_know_their_sign()
    test_divide_fix_writes_dace_int_floor()
    test_an_undefined_symbol_has_no_bounds()
