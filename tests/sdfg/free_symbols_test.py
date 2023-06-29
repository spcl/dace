# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math

N, M, K, L, unused = (dace.symbol(s) for s in 'NMKLU')


@dace.program
def fsymtest(A: dace.float32[20, N]):
    for i, j in dace.map[0:20, 0:N]:
        with dace.tasklet:
            a << A[i, i + M]
            b >> A[i, j]
            b = a + math.exp(K)


@dace.program
def fsymtest_multistate(A: dace.float32[20, N]):
    for k in range(K):
        for i, j in dace.map[0:20, 0:N]:
            with dace.tasklet:
                a << A[i, k + M]
                b >> A[i, j]
                b = L + a


def test_single_state():
    sdfg: dace.SDFG = fsymtest.to_sdfg()
    sdfg.simplify()
    assert len(sdfg.nodes()) == 1
    state = sdfg.node(0)
    assert state.free_symbols == {'M', 'N', 'K'}


def test_state_subgraph():
    sdfg = dace.SDFG('fsymtest2')
    state = sdfg.add_state()

    # Add a nested SDFG
    nsdfg = dace.SDFG('nsdfg')
    nstate = nsdfg.add_state()
    me, mx = state.add_map('map', dict(i='0:N'))
    nsdfg = state.add_nested_sdfg(nsdfg, None, {}, {}, symbol_mapping=dict(l=L / 2, i='i'))
    state.add_nedge(me, nsdfg, dace.Memlet())
    state.add_nedge(nsdfg, mx, dace.Memlet())

    # Entire graph
    assert state.free_symbols == {'L', 'N'}

    # Try a subgraph containing only the map contents
    assert state.scope_subgraph(me, include_entry=False, include_exit=False).free_symbols == {'L', 'i'}


def test_sdfg():
    sdfg: dace.SDFG = fsymtest_multistate.to_sdfg()
    sdfg.simplify()
    # Test each state separately
    for state in sdfg.nodes():
        assert (state.free_symbols == {'k', 'N', 'M', 'L'} or state.free_symbols == set())
    # The SDFG itself should have another free symbol
    assert sdfg.free_symbols == {'K', 'M', 'N', 'L'}


def test_constants():
    sdfg: dace.SDFG = fsymtest_multistate.to_sdfg()
    sdfg.simplify()
    sdfg.add_constant('K', 5)
    sdfg.add_constant('L', 20)

    for state in sdfg.nodes():
        assert (state.free_symbols == {'k', 'N', 'M'} or state.free_symbols == set())
    assert sdfg.free_symbols == {'M', 'N'}


def test_interstate_edge_symbols():
    i, j, k = (dace.symbol(s) for s in 'ijk')

    edge = dace.InterstateEdge(assignments={'i': 'j + k'})
    assert 'j' in edge.free_symbols
    assert 'k' in edge.free_symbols
    assert 'i' not in edge.free_symbols

    edge = dace.InterstateEdge(assignments={'i': 'i+1'})
    assert 'i' in edge.free_symbols

    edge = dace.InterstateEdge(condition='i < j', assignments={'i': '3'})
    assert 'i' in edge.free_symbols
    assert 'j' in edge.free_symbols

    edge = dace.InterstateEdge(assignments={'j': 'i + 1', 'i': '3'})
    assert 'i' in edge.free_symbols
    assert 'j' not in edge.free_symbols


def test_nested_sdfg_free_symbols():
    i, j, k = (dace.symbol(s) for s in 'ijk')

    outer_sdfg = dace.SDFG('outer')
    outer_init_state = outer_sdfg.add_state('outer_init')
    outer_guard_state = outer_sdfg.add_state('outer_guard')
    outer_body_state_1 = outer_sdfg.add_state('outer_body_1')
    outer_body_state_2 = outer_sdfg.add_state('outer_body_2')
    outer_exit_state = outer_sdfg.add_state('outer_exit')
    outer_sdfg.add_edge(outer_init_state, outer_guard_state, dace.InterstateEdge(assignments={'i': '0'}))
    outer_sdfg.add_edge(outer_guard_state, outer_body_state_1,
                        dace.InterstateEdge(condition='i < 10', assignments={'j': 'i + 1'}))
    outer_sdfg.add_edge(outer_guard_state, outer_exit_state, dace.InterstateEdge(condition='i >= 10'))
    outer_sdfg.add_edge(outer_body_state_1, outer_guard_state,
                        dace.InterstateEdge(condition='j >= 10', assignments={'i': 'i + 1'}))
    outer_sdfg.add_edge(outer_body_state_1, outer_body_state_2, dace.InterstateEdge(condition='j < 10'))
    outer_sdfg.add_edge(outer_body_state_2, outer_body_state_1, dace.InterstateEdge(assignments={'j': 'j + 1'}))

    inner_sdfg = dace.SDFG('inner')
    inner_init_state = inner_sdfg.add_state('inner_init')
    inner_guard_state = inner_sdfg.add_state('inner_guard')
    inner_body_state = inner_sdfg.add_state('inner_body')
    inner_exit_state = inner_sdfg.add_state('inner_exit')
    inner_sdfg.add_edge(inner_init_state, inner_guard_state, dace.InterstateEdge(assignments={'k': 'j + 1'}))
    inner_sdfg.add_edge(inner_guard_state, inner_body_state, dace.InterstateEdge(condition='k < 10'))
    inner_sdfg.add_edge(inner_guard_state, inner_exit_state,
                        dace.InterstateEdge(condition='k >= 10', assignments={'j': 'j + 1'}))
    inner_sdfg.add_edge(inner_body_state, inner_guard_state, dace.InterstateEdge(assignments={'k': 'k + 1'}))

    outer_body_state_2.add_nested_sdfg(inner_sdfg, None, {}, {}, symbol_mapping={'j': 'j'})

    assert not outer_sdfg.free_symbols
    assert 'i' not in inner_sdfg.free_symbols
    assert 'j' in inner_sdfg.free_symbols
    assert 'k' not in inner_sdfg.free_symbols


if __name__ == '__main__':
    test_single_state()
    test_state_subgraph()
    test_sdfg()
    test_constants()
    test_interstate_edge_symbols()
    test_nested_sdfg_free_symbols()
