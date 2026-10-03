# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.transformation.interstate import EndStateElimination, StartStateElimination, StateAssignElimination, StateFusion


def test_eliminate_end_state():
    sdfg = dace.SDFG('state_elimination_test')
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    state3 = sdfg.add_state()
    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=dict(k=1)))
    sdfg.add_edge(state2, state3, dace.InterstateEdge(assignments=dict(k='k + 1')))
    sdfg.apply_transformations(EndStateElimination)
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1


def test_eliminate_end_state_noassign():
    outer_sdfg = dace.SDFG('state_elimination_test_outer')
    outer_state = outer_sdfg.add_state()

    sdfg = dace.SDFG('state_elimination_test')
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    state3 = sdfg.add_state()
    sdfg.add_edge(state1, state2, dace.InterstateEdge())
    sdfg.add_edge(state2, state3, dace.InterstateEdge(assignments=dict(k='k + 1')))

    nsdfg = outer_state.add_nested_sdfg(sdfg, {}, {}, symbol_mapping={'k': 3})

    nsdfg.sdfg.simplify()
    nsdfg.sdfg.simplify()
    assert nsdfg.sdfg.number_of_nodes() == 2
    nsdfg.sdfg.apply_transformations(EndStateElimination)
    nsdfg.sdfg.simplify()
    assert nsdfg.sdfg.number_of_nodes() == 1


def test_state_assign_elimination():
    sdfg = dace.SDFG('state_assign_elimination_test')
    sdfg.add_array('A', [10], dace.float32)
    sdfg.add_array('B', [10], dace.float32)
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    state3 = sdfg.add_state()
    state3.add_nedge(state3.add_read('A'), state3.add_write('B'), dace.Memlet.simple('A', 'k'))

    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=dict(k=1)))
    sdfg.add_edge(state2, state3, dace.InterstateEdge(assignments=dict(k='k + 1')))

    # Assertions before/after transformations
    sdfg.apply_transformations_repeated(StateFusion)
    assert sdfg.number_of_nodes() == 3
    assert sdfg.apply_transformations_repeated(StateAssignElimination) == 1
    assert str(sdfg.nodes()[-1].edges()[0].data.subset) == 'k + 1'
    sdfg.apply_transformations_repeated(StateFusion)
    assert sdfg.number_of_nodes() == 2

    # Applying transformations again should yield one state
    assert sdfg.apply_transformations_repeated(StateAssignElimination) == 1
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1
    assert str(sdfg.nodes()[-1].edges()[0].data.subset) == '2'


def test_start_state_elimination_substitutes_moved_assignment_through_symbol_mapping():
    N = dace.symbol('N', dtype=dace.int64)

    inner = dace.SDFG('inner_start_state_elim')
    inner.add_symbol('M', dace.int64)
    inner.add_symbol('k', dace.int64)
    inner.add_array('a', [N], dace.float64)
    inner.add_array('b', [N], dace.float64)
    start = inner.add_state('start', is_start_block=True)
    body = inner.add_state('body')
    inner.add_edge(start, body, dace.InterstateEdge(assignments={'k': 'M + 1'}))
    body.add_mapped_tasklet('use', {'i': '0:N'}, {'__in': dace.Memlet('a[i]')},
                            '__out = __in + k', {'__out': dace.Memlet('b[i]')},
                            external_edges=True)

    sdfg = dace.SDFG('outer_start_state_elim')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state('call', is_start_block=True)
    # 'M' is bound to a DIFFERENTLY NAMED outer expression, so a naive move of the inner
    # assignment's RHS ('M + 1') into symbol_mapping leaks the inner-only name 'M'.
    nsdfg_node = state.add_nested_sdfg(inner, {'a'}, {'b'}, symbol_mapping={'M': N})
    nsdfg_node.no_inline = True
    state.add_edge(state.add_read('a'), None, nsdfg_node, 'a', dace.Memlet('a[0:N]'))
    state.add_edge(nsdfg_node, 'b', state.add_write('b'), None, dace.Memlet('b[0:N]'))

    assert inner.apply_transformations(StartStateElimination) == 1

    mapped_k = nsdfg_node.symbol_mapping['k']
    outer_names = set(sdfg.symbols.keys()) | set(sdfg.arrays.keys())
    assert {str(s) for s in mapped_k.free_symbols} <= outer_names
    assert str(mapped_k) == 'N + 1'

    sdfg.validate()

    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    sdfg(a=a, b=b, N=8)
    assert np.allclose(b, a + 9.0)  # k = M + 1 = N + 1 = 9 when N = 8


def test_sae_scalar():
    # Construct SDFG
    sdfg = dace.SDFG('state_assign_elimination_test')
    sdfg.add_array('A', [20, 20], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_scalar('scal', dace.int32, transient=True)
    initstate = sdfg.add_state()
    initstate.add_edge(initstate.add_tasklet('do', {}, {'out'}, 'out = 5'), 'out', initstate.add_write('scal'), None,
                       dace.Memlet('scal'))
    state = sdfg.add_state()
    sdfg.add_edge(initstate, state, dace.InterstateEdge(assignments=dict(s2='scal')))
    a = state.add_read('A')
    t = state.add_tasklet('do', {'inp'}, {'out'}, 'out = inp')
    b = state.add_write('B')
    state.add_edge(a, None, t, 'inp', dace.Memlet('A[s2, s2 + 1]'))
    state.add_edge(t, 'out', b, None, dace.Memlet('B[0]'))
    #######################################################

    assert sdfg.apply_transformations(StateAssignElimination) == 0


if __name__ == '__main__':
    test_eliminate_end_state()
    test_eliminate_end_state_noassign()
    test_start_state_elimination_substitutes_moved_assignment_through_symbol_mapping()
    test_state_assign_elimination()
    test_sae_scalar()
