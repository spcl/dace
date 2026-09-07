# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Offsetting a subset must fold a symbol against itself, whatever minted the two instances.

A memlet parsed from a string mints its symbols with the default dtype and no assumptions, while
a map parameter carries the frontend's -- ``int64``, ``nonnegative``. SymPy compares assumptions,
so the two ``k`` atoms are different objects and ``k - k`` stays in the subset instead of folding
to ``0``. In DaCe a symbol IS its name, so both denote the same value and the difference is zero;
left in the graph it also stops the SDFG from surviving a serialization round trip, because only
one of the two atoms is written with its dtype and the other comes back as a third symbol.
"""

import pytest

import dace
from dace import subsets, symbolic


def differently_minted_pair():
    """The same name from the two paths that disagree: a map parameter and a parsed memlet."""
    from_map = symbolic.symbol('k', dace.int64, nonnegative=True)
    from_string = dace.Memlet('A[k]').subset.min_element()[0]
    assert from_map - from_string != 0, 'the two paths agree now, so this test guards nothing'
    return from_map, from_string


def test_offset_folds_a_symbol_against_its_differently_minted_twin():
    from_map, _ = differently_minted_pair()
    rng = subsets.Range([(from_map, from_map, 1)])
    rng.offset(dace.Memlet('A[k]').subset, negative=True)
    assert rng == subsets.Range([(0, 0, 1)]), f'k - k did not fold: {rng}'


def test_offset_new_folds_it_too():
    from_map, _ = differently_minted_pair()
    rng = subsets.Range([(from_map, from_map, 1)]).offset_new(dace.Memlet('A[k]').subset, negative=True)
    assert rng == subsets.Range([(0, 0, 1)]), f'k - k did not fold: {rng}'


def test_an_offset_by_another_symbol_is_left_alone():
    """The fold is only for one name on two atoms -- a real difference must survive."""
    rng = subsets.Range([(symbolic.symbol('k', dace.int64, nonnegative=True), ) * 2 + (1, )])
    rng.offset(dace.Memlet('A[j]').subset, negative=True)
    assert rng != subsets.Range([(0, 0, 1)]), 'k - j was folded away'
    assert {str(s) for s in rng.free_symbols} == {'k', 'j'}, f'unexpected symbols in {rng}'


def test_the_guard_reports_a_duplicated_name():
    from_map, from_string = differently_minted_pair()
    assert symbolic.has_duplicate_symbol_names(from_map - from_string)
    assert not symbolic.has_duplicate_symbol_names(from_map - symbolic.symbol('j'))
    assert not symbolic.has_duplicate_symbol_names(5)


def test_compose_folds_it_too():
    """``Range.compose`` adds a bound of one subset to a bound of the other, same as ``offset``."""
    from_map, _ = differently_minted_pair()
    outer = subsets.Range([(from_map, from_map, 1)])
    composed = outer.compose(dace.Memlet('A[0:M - k]').subset)
    assert not symbolic.has_duplicate_symbol_names(composed.min_element()[0]), \
        f'composition kept one name on two symbols: {composed}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
