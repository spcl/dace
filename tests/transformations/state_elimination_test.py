# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.interstate import EndStateElimination, StateAssignElimination, StateFusion


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
    sdfg = dace.SDFG('state_elimination_test')
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    state3 = sdfg.add_state()
    sdfg.add_edge(state1, state2, dace.InterstateEdge())
    sdfg.add_edge(state2, state3, dace.InterstateEdge(assignments=dict(k='k + 1')))
    sdfg.simplify()
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 2
    sdfg.apply_transformations(EndStateElimination)
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1


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
    test_state_assign_elimination()
    test_sae_scalar()
