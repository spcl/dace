import dace
from dace.transformation.interstate import StateFusion


def test_fuse_assignments():
    sdfg = dace.SDFG('state_fusion_test')
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    state3 = sdfg.add_state()
    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=dict(k=1)))
    sdfg.add_edge(state2, state3,
                  dace.InterstateEdge(assignments=dict(k='k + 1')))
    sdfg.apply_transformations_repeated(StateFusion)
    assert sdfg.number_of_nodes() == 3


if __name__ == '__main__':
    test_fuse_assignments()
