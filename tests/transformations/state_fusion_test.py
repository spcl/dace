# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation import transformation
from dace.transformation.interstate import StateFusion
import networkx as nx
import numpy as np


# Inter-state condition tests
def test_fuse_assignments():
    """
    Two states in which the interstate assignment depends on an interstate
    value going into the first state. Should fail.
    """
    sdfg = dace.SDFG('state_fusion_test')
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    state3 = sdfg.add_state()
    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=dict(k=1)))
    sdfg.add_edge(state2, state3, dace.InterstateEdge(assignments=dict(k='k + 1')))
    sdfg.apply_transformations_repeated(StateFusion)
    assert sdfg.number_of_nodes() == 3


def test_fuse_assignment_in_use():
    """ 
    Two states with an interstate assignment in between, where the assigned
    value is used in the first state. Should fail.
    """
    sdfg = dace.SDFG('state_fusion_test')
    sdfg.add_array('A', [2], dace.int32)
    state1, state2, state3, state4 = tuple(sdfg.add_state() for _ in range(4))
    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments=dict(k=1)))
    sdfg.add_edge(state2, state3, dace.InterstateEdge())
    sdfg.add_edge(state3, state4, dace.InterstateEdge(assignments=dict(k=2)))

    state3.add_edge(state3.add_tasklet('one', {}, {'a'}, 'a = k'), 'a', state3.add_write('A'), None,
                    dace.Memlet('A[0]'))

    state4.add_edge(state3.add_tasklet('two', {}, {'a'}, 'a = k'), 'a', state3.add_write('A'), None,
                    dace.Memlet('A[1]'))

    try:
        StateFusion.apply_to(sdfg, first_state=state3, second_state=state4)
        raise AssertionError('States fused, test failed')
    except ValueError:
        print('Exception successfully caught')


# Connected components tests
def test_two_to_one_cc_fusion():
    """ Two states, first with two connected components, second with one. """
    sdfg = dace.SDFG('state_fusion_test')
    sdfg.add_array('A', [1], dace.int32)
    sdfg.add_array('B', [1], dace.int32)
    sdfg.add_array('C', [1], dace.int32)
    state1, state2 = tuple(sdfg.add_state() for _ in range(2))
    sdfg.add_edge(state1, state2, dace.InterstateEdge())

    # First state
    state1.add_edge(state1.add_tasklet('one', {}, {'a'}, 'a = 1'), 'a', state1.add_write('A'), None, dace.Memlet('A'))

    t2 = state1.add_tasklet('two', {}, {'b', 'c'}, 'b = 2; c = 3')
    state1.add_edge(t2, 'b', state1.add_write('B'), None, dace.Memlet('B'))
    state1.add_edge(t2, 'c', state1.add_write('C'), None, dace.Memlet('C'))

    # Second state
    t2 = state2.add_tasklet('three', {'a', 'b', 'c'}, {'out'}, 'out = a+b+c')
    state2.add_edge(state2.add_read('A'), None, t2, 'a', dace.Memlet('A'))
    state2.add_edge(state2.add_read('B'), None, t2, 'b', dace.Memlet('B'))
    state2.add_edge(state2.add_read('C'), None, t2, 'c', dace.Memlet('C'))
    state2.add_edge(t2, 'out', state2.add_write('C'), None, dace.Memlet('C'))

    assert sdfg.apply_transformations_repeated(StateFusion) == 1


def test_one_to_two_cc_fusion():
    """ Two states, first with one connected component, second with two. """
    sdfg = dace.SDFG('state_fusion_test')
    sdfg.add_array('A', [1], dace.int32)
    sdfg.add_array('B', [1], dace.int32)
    state1, state2 = tuple(sdfg.add_state() for _ in range(2))
    sdfg.add_edge(state1, state2, dace.InterstateEdge())

    # First state
    t1 = state1.add_tasklet('one', {}, {'a', 'b'}, 'a = 1; b = 2')
    state1.add_edge(t1, 'a', state1.add_write('A'), None, dace.Memlet('A'))
    state1.add_edge(t1, 'b', state1.add_write('B'), None, dace.Memlet('B'))

    # Second state
    state2.add_edge(state2.add_read('A'), None, state2.add_tasklet('one', {'a'}, {}, ''), 'a', dace.Memlet('A'))
    state2.add_edge(state2.add_read('B'), None, state2.add_tasklet('two', {'b'}, {}, ''), 'b', dace.Memlet('B'))

    assert sdfg.apply_transformations_repeated(StateFusion) == 1


def test_two_cc_fusion_separate():
    """ Two states, both with two connected components, fused separately. """
    sdfg = dace.SDFG('state_fusion_test')
    sdfg.add_array('A', [1], dace.int32)
    sdfg.add_array('B', [1], dace.int32)
    sdfg.add_array('C', [1], dace.int32)
    state1, state2 = tuple(sdfg.add_state() for _ in range(2))
    sdfg.add_edge(state1, state2, dace.InterstateEdge())

    # First state
    state1.add_edge(state1.add_tasklet('one', {}, {'a'}, 'a = 1'), 'a', state1.add_write('A'), None, dace.Memlet('A'))

    t2 = state1.add_tasklet('two', {}, {'b', 'c'}, 'b = 2; c = 3')
    state1.add_edge(t2, 'b', state1.add_write('B'), None, dace.Memlet('B'))
    state1.add_edge(t2, 'c', state1.add_write('C'), None, dace.Memlet('C'))

    # Second state
    state2.add_edge(state2.add_read('A'), None, state2.add_tasklet('one', {'a'}, {}, ''), 'a', dace.Memlet('A'))

    t2 = state2.add_tasklet('two', {'b', 'c'}, {}, '')
    state2.add_edge(state2.add_read('B'), None, t2, 'b', dace.Memlet('B'))
    state2.add_edge(state2.add_read('C'), None, t2, 'c', dace.Memlet('C'))

    assert sdfg.apply_transformations_repeated(StateFusion) == 1


def test_two_cc_fusion_together():
    """ Two states, both with two connected components, fused to one CC. """
    sdfg = dace.SDFG('state_fusion_test')
    sdfg.add_array('A', [1], dace.int32)
    sdfg.add_array('B', [1], dace.int32)
    sdfg.add_array('C', [1], dace.int32)
    state1, state2 = tuple(sdfg.add_state() for _ in range(2))
    sdfg.add_edge(state1, state2, dace.InterstateEdge())

    # First state
    state1.add_edge(state1.add_tasklet('one', {}, {'a'}, 'a = 1'), 'a', state1.add_write('A'), None, dace.Memlet('A'))

    t2 = state1.add_tasklet('two', {}, {'b', 'c'}, 'b = 2; c = 3')
    state1.add_edge(t2, 'b', state1.add_write('B'), None, dace.Memlet('B'))
    state1.add_edge(t2, 'c', state1.add_write('C'), None, dace.Memlet('C'))

    # Second state
    state2.add_edge(state2.add_read('B'), None, state2.add_tasklet('one', {'a'}, {}, ''), 'a', dace.Memlet('B'))

    t2 = state2.add_tasklet('two', {'b', 'c'}, {'d', 'e'}, 'd = b + c; e = b')
    state2.add_edge(state2.add_read('A'), None, t2, 'b', dace.Memlet('A'))
    state2.add_edge(state2.add_read('C'), None, t2, 'c', dace.Memlet('C'))
    state2.add_edge(t2, 'd', state2.add_write('A'), None, dace.Memlet('A'))
    state2.add_edge(t2, 'e', state2.add_write('C'), None, dace.Memlet('C'))

    assert sdfg.apply_transformations_repeated(StateFusion) == 1


# Data race avoidance tests
def test_write_write_path():
    """
    Two states where both write to the same range of an array, but there is
    a path between the write and the second write.
    """
    @dace.program
    def state_fusion_test(A: dace.int32[20, 20]):
        A += 1
        tmp = A + 2
        A[:] = tmp + 3

    sdfg = state_fusion_test.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(StateFusion)
    assert len(sdfg.nodes()) == 1


def test_write_write_no_overlap():
    """
    Two states where both write to different ranges of an array.
    """
    N = dace.symbol('N', positive=True)

    @dace.program
    def state_fusion_test(A: dace.int32[N, N]):
        A[0:N - 1, :] = 1
        A[N - 1, :] = 2

    sdfg = state_fusion_test.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(StateFusion)
    assert len(sdfg.nodes()) == 1


def test_read_write_no_overlap():
    """
    Two states where two separate CCs write and read to/from an array, but
    in different ranges.
    """
    N = dace.symbol('N')

    @dace.program
    def state_fusion_test(A: dace.int32[N, N], B: dace.int32[N, N]):
        A[:, 5:N] = 1
        B[:, 3:6] = A[:, 0:3]

    sdfg = state_fusion_test.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(StateFusion)
    assert len(sdfg.nodes()) == 1


def test_array_in_middle_no_overlap():
    """ 
    Two states that write and read from an array without overlap. Should be
    fused to two separate components.
    """
    sdfg = dace.SDFG('state_fusion_test')
    sdfg.add_array('A', [10, 10], dace.int32)
    sdfg.add_array('B', [5, 5], dace.int32)
    sdfg.add_array('C', [5, 5], dace.int32)
    state = sdfg.add_state()
    t1 = state.add_tasklet('init_a1', {}, {'a'}, '')
    rw1 = state.add_access('A')
    t2 = state.add_tasklet('a2b', {'a'}, {'b'}, '')
    wb = state.add_write('B')
    state.add_edge(t1, 'a', rw1, None, dace.Memlet('A[0:5, 0:5]'))
    state.add_edge(rw1, None, t2, 'a', dace.Memlet('A[0:5, 0:5]'))
    state.add_edge(t2, 'b', wb, None, dace.Memlet('B'))

    state2 = sdfg.add_state_after(state)
    t1 = state2.add_tasklet('init_a2', {}, {'a'}, '')
    rw2 = state2.add_access('A')
    t2 = state2.add_tasklet('a2c', {'a'}, {'c'}, '')
    wc = state2.add_write('C')
    state2.add_edge(t1, 'a', rw2, None, dace.Memlet('A[5:10, 5:10]'))
    state2.add_edge(rw2, None, t2, 'a', dace.Memlet('A[5:10, 5:10]'))
    state2.add_edge(t2, 'c', wc, None, dace.Memlet('C'))

    assert sdfg.apply_transformations_repeated(StateFusion) == 1
    assert len(list(nx.weakly_connected_components(sdfg.node(0).nx))) == 2


def test_array_in_middle_overlap():
    """ 
    Two states that write and read from an array with overlap. Should not be
    fused.
    """
    sdfg = dace.SDFG('state_fusion_test')
    sdfg.add_array('A', [10, 10], dace.int32)
    sdfg.add_array('B', [5, 5], dace.int32)
    sdfg.add_array('C', [5, 5], dace.int32)
    state = sdfg.add_state()
    t1 = state.add_tasklet('init_a1', {}, {'a'}, '')
    rw1 = state.add_access('A')
    t2 = state.add_tasklet('a2b', {'a'}, {'b'}, '')
    wb = state.add_write('B')
    state.add_edge(t1, 'a', rw1, None, dace.Memlet('A[0:5, 0:5]'))
    state.add_edge(rw1, None, t2, 'a', dace.Memlet('A[0:5, 0:5]'))
    state.add_edge(t2, 'b', wb, None, dace.Memlet('B'))

    state2 = sdfg.add_state_after(state)
    t1 = state2.add_tasklet('init_a2', {}, {'a'}, '')
    rw2 = state2.add_access('A')
    t2 = state2.add_tasklet('a2c', {'a'}, {'c'}, '')
    wc = state2.add_write('C')
    state2.add_edge(t1, 'a', rw2, None, dace.Memlet('A[0:5, 0:5]'))
    state2.add_edge(rw2, None, t2, 'a', dace.Memlet('A[0:5, 0:5]'))
    state2.add_edge(t2, 'c', wc, None, dace.Memlet('C'))

    assert sdfg.apply_transformations_repeated(StateFusion) == 0


def test_two_outputs_same_name():
    """ 
    First state writes to the same array twice, second state updates one value. 
    Should be fused to the right node in the second state or a data race will
    occur.
    """
    sdfg = dace.SDFG('state_fusion_test')
    sdfg.add_array('A', [2], dace.int32)
    sdfg.add_scalar('scal', dace.int32)
    state = sdfg.add_state()
    r = state.add_read('scal')
    t1 = state.add_tasklet('init_a1', {'s'}, {'a'}, 'a = 1 + s')
    w1 = state.add_write('A')
    t2 = state.add_tasklet('init_a2', {'s'}, {'a'}, 'a = 2 + s')
    w2 = state.add_write('A')
    state.add_edge(r, None, t1, 's', dace.Memlet('scal'))
    state.add_edge(t1, 'a', w1, None, dace.Memlet('A[0]'))
    state.add_edge(r, None, t2, 's', dace.Memlet('scal'))
    state.add_edge(t2, 'a', w2, None, dace.Memlet('A[1]'))

    state2 = sdfg.add_state_after(state)
    r1 = state2.add_read('A')
    t1 = state2.add_tasklet('update_a2', {'a'}, {'b'}, 'b = a + 2')
    w1 = state2.add_write('A')
    state2.add_edge(r1, None, t1, 'a', dace.Memlet('A[1]'))
    state2.add_edge(t1, 'b', w1, None, dace.Memlet('A[1]'))

    assert sdfg.apply_transformations_repeated(StateFusion) == 1

    A = np.zeros([2], dtype=np.int32)
    sdfg(A=A, scal=np.int32(0))
    assert A[0] == 1 and A[1] == 4


def test_inout_read_after_write():
    """ 
    First state ends with a computation that reads an array, while the second
    state both reads and writes to that same array. Fusion will then cause
    a RAW conflict.
    """
    sdfg = dace.SDFG('state_fusion_test')
    sdfg.add_array('A', [1], dace.int32)
    sdfg.add_array('B', [1], dace.int32)
    sdfg.add_array('C', [1], dace.int32)
    state = sdfg.add_state()
    r = state.add_read('A')
    t1 = state.add_tasklet('init_b', {'a'}, {'b'}, 'b = a + 1')
    rw = state.add_access('B')
    t2 = state.add_tasklet('init_c', {'b'}, {'c'}, 'c = 2 + b')
    w = state.add_access('C')
    state.add_edge(r, None, t1, 'a', dace.Memlet('A[0]'))
    state.add_edge(t1, 'b', rw, None, dace.Memlet('B[0]'))
    state.add_edge(rw, None, t2, 'b', dace.Memlet('B[0]'))
    state.add_edge(t2, 'c', w, None, dace.Memlet('C[0]'))

    state2 = sdfg.add_state_after(state)
    r1 = state2.add_read('B')
    t1 = state2.add_tasklet('update_b', {'bin'}, {'bout'}, 'bout = bin + bin')
    w1 = state2.add_write('B')
    state2.add_edge(r1, None, t1, 'bin', dace.Memlet('B[0]'))
    state2.add_edge(t1, 'bout', w1, None, dace.Memlet('B[0]'))

    assert sdfg.apply_transformations_repeated(StateFusion) == 0

    A = np.zeros([1], dtype=np.int32)
    B = np.zeros([1], dtype=np.int32)
    C = np.zeros([1], dtype=np.int32)
    sdfg(A=A, B=B, C=C)
    assert C[0] == 3
    assert B[0] == 2


def test_inout_second_state():
    """ 
    Second state has a computation that reads and writes to the same array, 
    while the first state also reads from that same array. Fusion will then 
    cause a potential data race.
    """
    sdfg = dace.SDFG('state_fusion_test')
    sdfg.add_array('A', [1], dace.int32)
    sdfg.add_array('B', [1], dace.int32)
    state = sdfg.add_state()
    r = state.add_read('A')
    t1 = state.add_tasklet('init_b', {'a'}, {'b'}, 'b = a + 1')
    w = state.add_write('B')
    state.add_edge(r, None, t1, 'a', dace.Memlet('A[0]'))
    state.add_edge(t1, 'b', w, None, dace.Memlet('B[0]'))

    state2 = sdfg.add_state_after(state)
    r1 = state2.add_read('A')
    t1 = state2.add_tasklet('update_a', {'a'}, {'aout'}, 'aout = a + 5')
    w1 = state2.add_write('A')
    state2.add_edge(r1, None, t1, 'a', dace.Memlet('A[0]'))
    state2.add_edge(t1, 'aout', w1, None, dace.Memlet('A[0]'))

    assert sdfg.apply_transformations_repeated(StateFusion) == 0

    A = np.zeros([1], dtype=np.int32)
    B = np.zeros([1], dtype=np.int32)
    sdfg(A=A, B=B)
    assert A[0] == 5
    assert B[0] == 1


def test_inout_second_state_2():
    @dace.program
    def func(A: dace.float64[128, 128], B: dace.float64[128, 128]):
        B << A
        for i, j in dace.map[0:128, 0:128]:
            with dace.tasklet:
                ai << A[i, j]
                ao >> A[i, j]
                ao = 2 * ai

    sdfg = func.to_sdfg(simplify=False)
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 2


if __name__ == '__main__':
    test_fuse_assignments()
    test_fuse_assignment_in_use()
    test_two_to_one_cc_fusion()
    test_one_to_two_cc_fusion()
    test_two_cc_fusion_separate()
    test_two_cc_fusion_together()
    test_write_write_path()
    test_write_write_no_overlap()
    test_read_write_no_overlap()
    test_array_in_middle_no_overlap()
    test_array_in_middle_overlap()
    test_two_outputs_same_name()
    test_inout_read_after_write()
    test_inout_second_state()
    test_inout_second_state_2()
