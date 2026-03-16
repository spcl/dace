"""
Unit test for CleanAccessNodeToScalarSliceToTaskletPattern.
"""
import dace
import numpy as np
from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import CleanAccessNodeToScalarSliceToTaskletPattern


def test_scalar_slice_removed():
    """
    Build:  A[5] -> tmp[0] -> tasklet -> B[5]
    where tmp is a transient size-1 array with in/out degree 1.
    After the pass, tmp should be gone and A connects directly to the tasklet.
    """
    sdfg = dace.SDFG('test_clean_scalar')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('tmp', [1], dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64)

    state = sdfg.add_state()

    a = state.add_access('A')
    tmp = state.add_access('tmp')
    b = state.add_access('B')
    tasklet = state.add_tasklet('double', {'inp'}, {'out'}, 'out = inp * 2.0')

    state.add_edge(a, None, tmp, None, dace.Memlet('A[5]'))
    state.add_edge(tmp, None, tasklet, 'inp', dace.Memlet('tmp[0]'))
    state.add_edge(tasklet, 'out', b, None, dace.Memlet('B[5]'))

    assert state.number_of_nodes() == 4
    assert any(n.data == 'tmp' for n in state.data_nodes())

    CleanAccessNodeToScalarSliceToTaskletPattern().apply_pass(sdfg, None)

    # tmp should be removed, A wired directly to tasklet
    assert state.number_of_nodes() == 3
    assert not any(n.data == 'tmp' for n in state.data_nodes())

    tasklet_node = next(n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet))
    in_edges = state.in_edges(tasklet_node)
    assert len(in_edges) == 1
    assert in_edges[0].data.data == 'A'
    assert str(in_edges[0].data.subset) == '5'

    # Numerical check
    sdfg.validate()
    A = np.random.rand(10)
    B = np.zeros(10)
    sdfg(A=A, B=B)
    assert np.isclose(B[5], A[5] * 2.0)


if __name__ == '__main__':
    test_scalar_slice_removed()