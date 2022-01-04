# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
    sdfg.coarsen_dataflow()
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
    sdfg.coarsen_dataflow()
    # Test each state separately
    for state in sdfg.nodes():
        assert (state.free_symbols == {'k', 'N', 'M', 'L'} or state.free_symbols == set())
    # The SDFG itself should have another free symbol
    assert sdfg.free_symbols == {'K', 'M', 'N', 'L'}


def test_constants():
    sdfg: dace.SDFG = fsymtest_multistate.to_sdfg()
    sdfg.coarsen_dataflow()
    sdfg.add_constant('K', 5)
    sdfg.add_constant('L', 20)

    for state in sdfg.nodes():
        assert (state.free_symbols == {'k', 'N', 'M'} or state.free_symbols == set())
    assert sdfg.free_symbols == {'M', 'N'}


if __name__ == '__main__':
    test_single_state()
    test_state_subgraph()
    test_sdfg()
    test_constants()
