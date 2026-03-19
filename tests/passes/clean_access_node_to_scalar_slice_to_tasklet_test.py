"""
Unit test for CleanAccessNodeToScalarSliceToTaskletPattern.
"""
import dace
import numpy as np
from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import CleanAccessNodeToScalarSliceToTaskletPattern
import pytest


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


def _build_two_state_sdfg():
    """
    State 0:  A[5] -> tmp[0] -> tasklet -> B[5]
    State 1:  tmp[0] -> tasklet2 -> C[3]
    """
    sdfg = dace.SDFG('test_no_remove')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('tmp', [1], dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64)
    sdfg.add_array('C', [10], dace.float64)

    s0 = sdfg.add_state('s0')
    a = s0.add_access('A')
    tmp0 = s0.add_access('tmp')
    b = s0.add_access('B')
    t0 = s0.add_tasklet('double', {'inp'}, {'out'}, 'out = inp * 2.0')
    s0.add_edge(a, None, tmp0, None, dace.Memlet('A[5]'))
    s0.add_edge(tmp0, None, t0, 'inp', dace.Memlet('tmp[0]'))
    s0.add_edge(t0, 'out', b, None, dace.Memlet('B[5]'))

    s1 = sdfg.add_state('s1')
    tmp1 = s1.add_access('tmp')
    c = s1.add_access('C')
    t1 = s1.add_tasklet('triple', {'inp'}, {'out'}, 'out = inp * 3.0')
    s1.add_edge(tmp1, None, t1, 'inp', dace.Memlet('tmp[0]'))
    s1.add_edge(t1, 'out', c, None, dace.Memlet('C[3]'))

    sdfg.add_edge(s0, s1, dace.InterstateEdge())
    return sdfg, s0


@pytest.mark.parametrize(
    "permissive, expect_removed",
    [
        (False, False),  # tmp used in other state -> keep
        (True, True),  # permissive -> remove anyway
    ])
def test_scalar_used_in_other_state(permissive, expect_removed):
    sdfg, s0 = _build_two_state_sdfg()
    assert s0.number_of_nodes() == 4

    CleanAccessNodeToScalarSliceToTaskletPattern(permissive=permissive).apply_pass(sdfg, None)

    if expect_removed:
        assert s0.number_of_nodes() == 3, "tmp should have been removed (permissive)"
        assert not any(n.data == 'tmp' for n in s0.data_nodes())
    else:
        assert s0.number_of_nodes() == 4, "tmp should not have been removed"
        assert any(n.data == 'tmp' for n in s0.data_nodes())


if __name__ == '__main__':
    test_scalar_slice_removed()
    test_scalar_not_removed_if_used_in_other_state()
