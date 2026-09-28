# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SymbolScopes`` must answer ``symbols_defined_at`` for every node, tabulating each state once."""
import collections

import pytest

import dace
from dace import dtypes
from dace.sdfg.state import LoopRegion, SDFGState, SymbolResolver
from dace.transformation.passes.analysis.scopes import SymbolScopes

N = dace.symbol('N')
M = dace.symbol('M')


def assert_matches(sdfg: dace.SDFG) -> SymbolResolver:
    """Same keys, same order, same types as ``symbols_defined_at``, for every node of every state."""
    result = SymbolScopes().apply_pass(sdfg, {})
    checked = 0
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            for node in state.nodes():
                expected = state.symbols_defined_at(node)
                actual = result.defined_at(state, node)
                assert list(actual.items()) == list(expected.items()), f'{state.label}/{node}'
                checked += 1
    assert checked > 0, 'no nodes compared'
    return result


def nested_maps() -> dace.SDFG:

    @dace.program
    def program(A: dace.float64[N, M]):
        for i in dace.map[0:N]:
            for j in dace.map[0:M]:
                A[i, j] = A[i, j] * 2.0

    return program.to_sdfg(simplify=False)


def nested_sdfg() -> dace.SDFG:

    @dace.program
    def inner(A: dace.float64[N]):
        for i in dace.map[0:N]:
            A[i] = A[i] + 1.0

    @dace.program
    def outer(A: dace.float64[N]):
        inner(A)

    return outer.to_sdfg(simplify=False)


def gemm(simplify: bool) -> dace.SDFG:

    @dace.program
    def program(A: dace.float64[N, M], B: dace.float64[M, N], C: dace.float64[N, N]):
        for i, j in dace.map[0:N, 0:N]:
            acc = dace.float64(0)
            for k in range(M):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc

    return program.to_sdfg(simplify=simplify)


def loop_around_a_map() -> tuple[dace.SDFG, SDFGState, dace.nodes.Tasklet]:
    sdfg = dace.SDFG('loop_around_a_map')
    sdfg.add_array('A', [N, M], dace.float64)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    entry, exit_node = body.add_map('m', {'j': '0:M'})
    tasklet = body.add_tasklet('w', {}, {'o': None}, 'o = i + j')
    body.add_nedge(entry, tasklet, dace.Memlet())
    body.add_memlet_path(tasklet, exit_node, body.add_access('A'), src_conn='o', memlet=dace.Memlet('A[i, j]'))
    return sdfg, body, tasklet


def dynamic_map_range() -> tuple[dace.SDFG, SDFGState, dace.nodes.MapEntry, dace.nodes.Tasklet]:
    sdfg = dace.SDFG('dynamic_range')
    sdfg.add_array('A', [N * M], dace.float64)
    sdfg.add_array('lim', [1], dace.int32)
    state = sdfg.add_state('s', is_start_block=True)
    entry, exit_node = state.add_map('m', {'i': '0:bound'})
    entry.add_in_connector('bound')
    state.add_edge(state.add_access('lim'), None, entry, 'bound', dace.Memlet('lim[0]'))
    tasklet = state.add_tasklet('w', {}, {'o': None}, 'o = 1.0')
    state.add_nedge(entry, tasklet, dace.Memlet())
    state.add_memlet_path(tasklet, exit_node, state.add_access('A'), src_conn='o', memlet=dace.Memlet('A[i]'))
    return sdfg, state, entry, tasklet


def map_inside_a_consume() -> dace.SDFG:
    sdfg = dace.SDFG('map_inside_a_consume')
    sdfg.add_stream('S', dace.int32, transient=True)
    sdfg.add_array('out', [M], dace.int32)
    state = sdfg.add_state('s', is_start_block=True)
    centry, cexit = state.add_consume('cons', ('p', '4'))
    mentry, mexit = state.add_map('m', {'j': '0:M'})
    tasklet = state.add_tasklet('w', {'s': None}, {'o': None}, 'o = s + p + j')
    state.add_edge(state.add_access('S'), None, centry, 'IN_stream', dace.Memlet('S[0]'))
    state.add_memlet_path(centry, mentry, tasklet, src_conn='OUT_stream', dst_conn='s', memlet=dace.Memlet('S[0]'))
    state.add_memlet_path(tasklet, mexit, cexit, state.add_access('out'), src_conn='o', memlet=dace.Memlet('out[j]'))
    return sdfg


def map_around_a_nested_sdfg() -> tuple[dace.SDFG, dace.SDFG]:
    """Outer map ``i`` (int64) calls a body binding its own ``n`` (declared int32) to ``i``; the body never names ``i``."""
    inner = dace.SDFG('body')
    inner.add_symbol('n', dace.int32)
    inner.add_array('b', [M], dace.float64)
    state = inner.add_state('s', is_start_block=True)
    entry, exit_node = state.add_map('inner', {'k': '0:n'})
    tasklet = state.add_tasklet('w', {}, {'o': None}, 'o = k')
    state.add_nedge(entry, tasklet, dace.Memlet())
    state.add_memlet_path(tasklet, exit_node, state.add_access('b'), src_conn='o', memlet=dace.Memlet('b[k]'))

    sdfg = dace.SDFG('outer')
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_array('A', [M, M], dace.float64)
    outer = sdfg.add_state('s', is_start_block=True)
    entry, exit_node = outer.add_map('outer', {'i': '0:M'})
    entry.map.range = dace.subsets.Range([(0, dace.symbol('M', dace.int64) - 1, 1)])
    node = outer.add_nested_sdfg(inner, {}, {'b': None}, symbol_mapping={'n': 'i', 'M': 'M'})
    outer.add_nedge(entry, node, dace.Memlet())
    outer.add_memlet_path(node, exit_node, outer.add_access('A'), src_conn='b', memlet=dace.Memlet('A[i, 0:M]'))
    sdfg.validate()
    return sdfg, inner


def test_a_nested_sdfg_sees_only_its_own_symbols():
    """The outer map parameter reaches the body only through ``symbol_mapping``, as the body's own ``n``."""
    sdfg, inner = map_around_a_nested_sdfg()
    result = assert_matches(sdfg)

    for state in inner.states():
        for node in state.nodes():
            visible = result.defined_at(state, node)
            assert 'i' not in visible, f'{node}: the outer map parameter leaked into the nested SDFG'
            assert visible['n'] == dace.int32, 'the body types its own symbol as it declares it'


@pytest.mark.parametrize('build', [
    nested_maps, nested_sdfg, map_inside_a_consume, lambda: loop_around_a_map()[0], lambda: dynamic_map_range()[0],
    lambda: gemm(False), lambda: gemm(True)
])
def test_every_node_sees_what_symbols_defined_at_sees(build):
    assert_matches(build())


def test_a_loop_body_sees_the_iterator_and_the_map_parameter():
    sdfg, body, tasklet = loop_around_a_map()
    visible = SymbolScopes().apply_pass(sdfg, {}).defined_at(body, tasklet)
    assert {'i', 'j', 'N', 'M'} <= visible.keys(), sorted(visible)


def test_a_dynamic_range_binds_its_connector_inside_the_map_only():
    sdfg, state, entry, tasklet = dynamic_map_range()
    result = SymbolScopes().apply_pass(sdfg, {})
    assert {'i', 'bound', 'N', 'M'} <= result.defined_at(state, tasklet).keys()
    assert not {'i', 'bound'} & result.defined_at(state, entry).keys(), 'an entry sees only its outer scope'
    assert isinstance(result.defined_at(state, tasklet)['N'], dtypes.typeclass)


def test_every_state_and_scope_is_tabulated_once(monkeypatch):
    """One ``symbols_defined_at_state`` per state and one ``new_symbols`` per scope, however many nodes ask."""
    sdfg = nested_sdfg()
    calls = collections.Counter()
    for cls, name in [(SDFGState, 'sdfg_symbols'), (SDFGState, 'symbols_defined_at_state'),
                      (dace.nodes.MapEntry, 'new_symbols')]:
        original = vars(cls)[name]

        def counting(*args, original=original, name=name, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(cls, name, counting)

    result = SymbolScopes().apply_pass(sdfg, {})
    tabulated = dict(calls)
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            for node in state.nodes():
                result.defined_at(state, node)

    states = [state for nested in sdfg.all_sdfgs_recursive() for state in nested.states()]
    entries = [n for state in states for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]
    assert tabulated == {
        'sdfg_symbols': len(list(sdfg.all_sdfgs_recursive())),
        'symbols_defined_at_state': len(states),
        'new_symbols': len(entries),
    }, tabulated
    assert dict(calls) == tabulated, 'a query recomputed a table'


def test_a_resolver_tabulates_only_the_states_it_is_asked_about():
    sdfg, body, tasklet = loop_around_a_map()
    other = sdfg.add_state_after(sdfg.start_block, 'after')
    sut = SymbolResolver()

    sut.defined_at(body, tasklet)

    assert list(sut.scope_tables) == [body] and other not in sut.scope_tables


def test_a_scope_added_after_tabulation_is_still_answered():
    state = dynamic_map_range()[1]
    sut = SymbolResolver()
    sut.scopes(state)
    entry, exit_node = state.add_map('late', {'k': '0:N'})
    tasklet = state.add_tasklet('late_w', {}, {'o': None}, 'o = k')
    state.add_nedge(entry, tasklet, dace.Memlet())
    state.add_memlet_path(tasklet, exit_node, state.add_access('A'), src_conn='o', memlet=dace.Memlet('A[k]'))

    assert list(sut.defined_at(state, tasklet).items()) == list(state.symbols_defined_at(tasklet).items())


def test_the_answer_is_a_copy_the_caller_may_extend():
    state, tasklet = dynamic_map_range()[1::2]
    sut = SymbolResolver()
    sut.defined_at(state, tasklet)['extra'] = dace.int32
    assert 'extra' not in sut.defined_at(state, tasklet)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
