# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The shared scoped-symbol/dtype resolver: equivalence with the core call, the dtype it must not guess,
and the invalidation contract that keeps a cached answer honest."""
import pytest

import dace
from dace import subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation.passes.analysis import scopes


def two_deep_map_nest() -> tuple[SDFG, SDFGState, nodes.Tasklet]:
    """``for i in 0:N: for j in 0:8: B[i] += A[i] * j`` -- an inner scope, an outer scope, a top level."""
    n = dace.symbol('N', dace.int32)
    sdfg = SDFG('two_deep_map_nest')
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_array('A', [n], dace.float64)
    sdfg.add_array('B', [n], dace.float32)  # deliberately unlike A, so ladder precedence is observable
    state = sdfg.add_state()
    read = state.add_access('A')
    outer_entry, outer_exit = state.add_map('outer', {'i': '0:N'})
    inner_entry, inner_exit = state.add_map('inner', {'j': '0:8'})
    tasklet = state.add_tasklet('body', {'a'}, {'o'}, 'o = a * j')
    write = state.add_access('B')
    state.add_memlet_path(read, outer_entry, inner_entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet,
                          inner_exit,
                          outer_exit,
                          write,
                          src_conn='o',
                          memlet=dace.Memlet(data='B', subset='i', wcr='lambda x, y: x + y'))
    sdfg.validate()
    return sdfg, state, tasklet


def int32_bounded_map() -> tuple[SDFG, SDFGState, nodes.Tasklet]:
    """``for i in M:N: A[i] = A[i]`` with M and N both int32, so the parameter is int32 and not int64.

    The bounds are handed in as a :class:`~dace.subsets.Range` rather than the ``'M:N'`` string form on
    purpose: the string form stores the end as ``N - 1``, and that integer literal infers as int64 and
    widens the parameter. That is exactly why guessing int64 usually looks right.
    """
    m = dace.symbol('M', dace.int32)
    n = dace.symbol('N', dace.int32)
    sdfg = SDFG('int32_bounded_map')
    sdfg.add_symbol('M', dace.int32)
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_array('A', [n + 1], dace.float64)
    state = sdfg.add_state()
    read = state.add_access('A')
    entry, exit_node = state.add_map('m', {'i': subsets.Range([(m, n, 1)])})
    tasklet = state.add_tasklet('body', {'a'}, {'o'}, 'o = a')
    write = state.add_access('A')
    state.add_memlet_path(read, entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, exit_node, write, src_conn='o', memlet=dace.Memlet('A[i]'))
    sdfg.validate()
    return sdfg, state, tasklet


def add_second_map_scope(state: SDFGState) -> nodes.Tasklet:
    """A second, independent ``0:4`` map in ``state`` -- the scoped symbol a pass would add."""
    read = state.add_access('A')
    entry, exit_node = state.add_map('added', {'k': '0:4'})
    tasklet = state.add_tasklet('added_body', {'a'}, {'o'}, 'o = a + k')
    write = state.add_access('B')
    state.add_memlet_path(read, entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[k]'))
    state.add_memlet_path(tasklet, exit_node, write, src_conn='o', memlet=dace.Memlet('B[k]'))
    return tasklet


def test_resolver_answers_every_node_of_a_map_nest_exactly_as_the_state_does():
    sdfg, state, _ = two_deep_map_nest()
    sut = scopes.ScopedSymbolResolver()

    answers = [(node, dict(sut.defined_at(state, node)), dict(state.symbols_defined_at(node)))
               for node in state.nodes()]

    for node, cached, authority in answers:
        assert cached == authority, node
        assert list(cached) == list(authority), node  # ordered tables: the key order is part of the answer
    # Structure: the nest really did contribute two scoped parameters, so the equivalence above is not
    # an equivalence between two empty tables.
    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert list(sut.defined_at(state, tasklet)) == ['N', 'i', 'j']
    assert list(sut.defined_at(state, next(n for n in state.nodes() if isinstance(n, nodes.AccessNode)))) == ['N']


def test_tabulating_one_state_leaves_the_other_states_untouched():
    sdfg, state, tasklet = two_deep_map_nest()
    second = sdfg.add_state_after(state)
    sut = scopes.ScopedSymbolResolver()

    sut.defined_at(state, tasklet)

    assert list(sut.state_tables) == [state]
    assert second not in sut.state_tables
    assert list(sut.sdfg_bases) == [sdfg]


def test_a_map_parameter_bounded_by_int32_symbols_resolves_to_int32():
    sdfg, state, tasklet = int32_bounded_map()
    sut = scopes.ScopedSymbolResolver()

    resolved = sut.resolve_dtype('i', sdfg, state, tasklet)

    assert resolved == dace.int32
    assert resolved != dace.int64
    assert 'i' not in sdfg.symbols  # a plain symbol-table lookup would have missed it entirely


def test_one_name_at_two_dtypes_is_two_symbols_that_refuse_to_cancel():
    narrow = symbolic.symbol('i', dace.int32)
    wide = symbolic.symbol('i', dace.int64)

    assert narrow != wide
    assert hash(narrow) != hash(wide)
    # sympy reached through dace.symbolic; Min is the construction the injectivity test builds.
    assert str(symbolic.sympy.Min(narrow, wide)) == 'Min(i, i)'
    assert symbolic.simplify(narrow - wide) != 0
    assert str(symbolic.simplify(narrow - wide)) == 'i - i'
    # The asymmetry that makes the trap hard to see: ``_eval_subs`` matches by NAME, so substitution
    # crosses the dtype boundary that equality, hashing and cancellation all refuse to cross.
    assert str(wide.subs({narrow: 4})) == '4'


def test_a_map_scope_added_after_a_query_is_refused_until_the_state_is_invalidated():
    sdfg, state, tasklet = two_deep_map_nest()
    sut = scopes.ScopedSymbolResolver()
    assert list(sut.defined_at(state, tasklet)) == ['N', 'i', 'j']

    added = add_second_map_scope(state)

    with pytest.raises(scopes.StaleScopeCache):
        sut.defined_at(state, added)

    sut.invalidate_state(state)
    assert list(sut.defined_at(state, added)) == ['N', 'k']
    assert sut.resolve_dtype('k', sdfg, state, added) == dace.int64


def test_a_symbol_declared_after_a_query_stays_invisible_until_the_sdfg_is_invalidated():
    sdfg, state, tasklet = two_deep_map_nest()
    sut = scopes.ScopedSymbolResolver()
    assert list(sut.defined_at(state, tasklet)) == ['N', 'i', 'j']

    sdfg.add_symbol('EXTRA', dace.float32)

    # The documented contract: stale until invalidated. The resolver does not detect the mutation and
    # does not quietly re-derive; invalidating a DIFFERENT state does not help either, since the
    # per-SDFG base is what changed.
    assert 'EXTRA' not in sut.defined_at(state, tasklet)
    sut.invalidate_state(state)
    assert 'EXTRA' not in sut.defined_at(state, tasklet)

    sut.invalidate_sdfg(sdfg)
    assert list(sut.defined_at(state, tasklet)) == ['N', 'EXTRA', 'i', 'j']
    assert sut.resolve_dtype('EXTRA', sdfg, state, tasklet) == dace.float32


def test_an_undeterminable_dtype_is_reported_and_cannot_be_defaulted():
    sdfg, state, tasklet = two_deep_map_nest()
    sut = scopes.ScopedSymbolResolver()

    with pytest.raises(scopes.UndeterminedSymbolDType) as raised:
        sut.resolve_dtype('nowhere', sdfg, state, tasklet)

    assert raised.value.name == 'nowhere'
    reported = sut.resolve_dtype_or_undetermined('nowhere', sdfg, state, tasklet)
    assert reported is scopes.UNDETERMINED
    assert repr(reported) == 'UNDETERMINED'
    # The failure this sentinel exists to stop: `resolved or dace.int64` must not compile a guess.
    with pytest.raises(TypeError):
        bool(reported)


def test_the_ladder_prefers_each_more_specific_source_over_the_symbol_table():
    sdfg, state, tasklet = two_deep_map_nest()
    write_edge = next(e for e in state.out_edges(tasklet))  # carries B, float32
    sut = scopes.ScopedSymbolResolver()

    # Each rung answers on its own ...
    assert sut.resolve_dtype('A', sdfg) == dace.float64
    assert sut.resolve_dtype('unnamed', sdfg, state, tasklet, edge=write_edge) == dace.float32
    assert sut.resolve_dtype('i', sdfg, state, tasklet) == dace.int64
    assert sut.resolve_dtype('N', sdfg) == dace.int32
    # ... and a more specific rung outranks every rung below it.
    assert sut.resolve_dtype('A', sdfg, state, tasklet, edge=write_edge) == dace.float64
    tasklet.in_connectors['a'] = dace.float16
    assert sut.resolve_dtype('unnamed', sdfg, state, tasklet, connector='a', edge=write_edge) == dace.float16
    assert 'i' not in sdfg.symbols and 'unnamed' not in sdfg.symbols


def test_an_untyped_connector_falls_through_instead_of_answering_void():
    sdfg, state, tasklet = two_deep_map_nest()
    sut = scopes.ScopedSymbolResolver()

    assert tasklet.in_connectors['a'] == scopes.UNTYPED_CONNECTOR
    assert sut.resolve_dtype('N', sdfg, state, tasklet, connector='a') == dace.int32

    tasklet.in_connectors['a'] = dace.float32
    assert sut.resolve_dtype('N', sdfg, state, tasklet, connector='a') == dace.float32


def test_an_interstate_assignment_declared_nowhere_resolves_from_its_edge():
    sdfg, state, tasklet = two_deep_map_nest()
    follow = sdfg.add_state('follow')
    sdfg.add_edge(state, follow, dace.InterstateEdge(assignments={'zlcrit': 'N + 1'}))
    iedge = next(e.data for e in sdfg.edges())
    sut = scopes.ScopedSymbolResolver()

    assert 'zlcrit' not in sdfg.symbols
    assert sut.resolve_dtype_or_undetermined('zlcrit', sdfg, state, tasklet) is scopes.UNDETERMINED
    assert sut.resolve_dtype('zlcrit', sdfg, state, tasklet, interstate_edge=iedge) == dace.int64


if __name__ == '__main__':
    test_resolver_answers_every_node_of_a_map_nest_exactly_as_the_state_does()
    test_tabulating_one_state_leaves_the_other_states_untouched()
    test_a_map_parameter_bounded_by_int32_symbols_resolves_to_int32()
    test_one_name_at_two_dtypes_is_two_symbols_that_refuse_to_cancel()
    test_a_map_scope_added_after_a_query_is_refused_until_the_state_is_invalidated()
    test_a_symbol_declared_after_a_query_stays_invisible_until_the_sdfg_is_invalidated()
    test_an_undeterminable_dtype_is_reported_and_cannot_be_defaulted()
    test_the_ladder_prefers_each_more_specific_source_over_the_symbol_table()
    test_an_untyped_connector_falls_through_instead_of_answering_void()
    test_an_interstate_assignment_declared_nowhere_resolves_from_its_edge()
