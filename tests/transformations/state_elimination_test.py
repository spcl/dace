import dace
from dace.transformation.interstate import StateAssignElimination


def test_eliminate_end_state():
    sdfg = dace.SDFG('state_elimination_test')
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    state3 = sdfg.add_state()
    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=dict(k=1)))
    sdfg.add_edge(state2, state3,
                  dace.InterstateEdge(assignments=dict(k='k + 1')))
    sdfg.apply_strict_transformations()
    assert sdfg.number_of_nodes() == 1


def test_state_assign_elimination():
    sdfg = dace.SDFG('state_assign_elimination_test')
    sdfg.add_array('A', [10], dace.float32)
    sdfg.add_array('B', [10], dace.float32)
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    state3 = sdfg.add_state()
    state3.add_nedge(state3.add_read('A'), state3.add_write('B'),
                     dace.Memlet.simple('A', 'k'))

    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=dict(k=1)))
    sdfg.add_edge(state2, state3,
                  dace.InterstateEdge(assignments=dict(k='k + 1')))

    # Assertions before/after transformations
    sdfg.apply_strict_transformations()
    assert sdfg.number_of_nodes() == 3
    assert sdfg.apply_transformations_repeated(StateAssignElimination) == 1
    assert str(sdfg.nodes()[-1].edges()[0].data.subset) == 'k + 1'
    sdfg.apply_strict_transformations()
    assert sdfg.number_of_nodes() == 2

    # Applying transformations again should yield one state
    assert sdfg.apply_transformations_repeated(StateAssignElimination) == 1
    sdfg.apply_strict_transformations()
    assert sdfg.number_of_nodes() == 1
    assert str(sdfg.nodes()[-1].edges()[0].data.subset) == '2'

if __name__ == '__main__':
    test_eliminate_end_state()
    test_state_assign_elimination()