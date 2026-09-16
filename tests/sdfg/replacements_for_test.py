# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that narrowing a symbolic replacement to the entries that can apply leaves ``subs`` unchanged. """

import pytest

import dace
from dace import symbolic
from dace.sdfg.replace import replacements_for

N = dace.symbol('N', dtype=dace.int64)
M = dace.symbol('M', dtype=dace.int64)
K = dace.symbol('K', dtype=dace.int64)
A = dace.symbol('A')
i = dace.symbol('i')


def unrelated(count: int) -> dict:
    return {dace.symbol(f'u{n}'): dace.symbol(f'w{n}') for n in range(count)}


def test_an_expression_without_a_replaced_name_needs_no_entry():
    repl = unrelated(20)

    assert replacements_for((N + 1, ), repl) == {}


def test_only_the_entries_naming_a_symbol_of_the_expression_are_kept():
    repl = {**unrelated(20), N: K}

    assert list(replacements_for((N + 1, ), repl)) == [N]


@pytest.mark.parametrize('repl', [{N: M, M: K}, {M: K, N: M}, {N: M + 1, M: N}])
def test_a_chain_keeps_the_entries_the_first_substitution_brings_in(repl):
    """``subs`` applies the entries in sequence, so ``M: K`` still reaches an ``M`` that ``N: M`` introduced."""
    expr = 2 * N - 1
    full = {**unrelated(5), **repl}

    narrowed = replacements_for((expr, ), full)

    assert list(narrowed) == [key for key in full if key in repl]
    assert expr.subs(narrowed) == expr.subs(full)


def test_a_subscripted_container_counts_as_a_name_of_the_expression():
    """``Subscript.free_symbols`` leaves out the container, but ``subs`` renames it."""
    expr = symbolic.pystr_to_symbolic('A[i] + 1')
    full = {**unrelated(5), A: dace.symbol('B')}

    narrowed = replacements_for((expr, ), full)

    assert list(narrowed) == [A]
    assert expr.subs(narrowed) == expr.subs(full)


def test_a_symexpr_counts_the_names_of_both_halves():
    expr = symbolic.SymExpr(N + 1, M + 1)

    assert list(replacements_for((expr, ), {**unrelated(5), M: K})) == [M]


def test_a_mapping_that_cannot_be_narrowed_is_kept_whole():
    full = {**unrelated(5), 'N': 'K'}

    assert replacements_for((N + 1, ), full) is full


def test_a_range_renames_only_through_the_entries_that_apply():
    sdfg = dace.SDFG('range_rename')
    sdfg.add_symbol('N', dace.int64)
    state = sdfg.add_state()
    map_entry, _ = state.add_map('m', dict(k='0:N'))
    names = {f'u{n}': f'w{n}' for n in range(20)}

    sdfg.replace_dict({**names, 'N': 'K'})

    assert [tuple(map(str, rng)) for rng in map_entry.map.range.ranges] == [('0', 'K - 1', '1')]
