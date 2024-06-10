# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.sdfg.state import ConditionalRegion


def test_conditional_regular_if_true():
    sdfg = dace.SDFG('regular_if')
    state0 = sdfg.add_state('state0', is_start_block=True)
    if1 = ConditionalRegion(label='loop1', condition_expr='True')
    sdfg.add_array('A', [1], dace.float32)
    sdfg.add_node(if1)
    state1 = if1.add_state('state1', is_start_block=True)
    acc_a = state1.add_access('A')
    t1 = state1.add_tasklet('t1', None, {'a'}, 'a = 20')
    state1.add_edge(t1, 'a', acc_a, None, dace.Memlet('A[0]'))
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, if1, dace.InterstateEdge())
    sdfg.add_edge(if1, state3, dace.InterstateEdge())

    assert sdfg.is_valid()

    a_validation = np.array([20], dtype=np.float32)
    a_test = np.zeros((1,), dtype=np.float32)
    sdfg(A=a_test)
    assert np.allclose(a_validation, a_test)

def test_conditional_regular_if_false():
    sdfg = dace.SDFG('regular_if')
    state0 = sdfg.add_state('state0', is_start_block=True)
    if1 = ConditionalRegion(label='loop1', condition_expr='False')
    sdfg.add_array('A', [1], dace.float32)
    sdfg.add_node(if1)
    state1 = if1.add_state('state1', is_start_block=True)
    acc_a = state1.add_access('A')
    t1 = state1.add_tasklet('t1', None, {'a'}, 'a = 20')
    state1.add_edge(t1, 'a', acc_a, None, dace.Memlet('A[0]'))
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, if1, dace.InterstateEdge())
    sdfg.add_edge(if1, state3, dace.InterstateEdge())

    assert sdfg.is_valid()

    a_validation = np.zeros((1,), dtype=np.float32)
    a_test = np.zeros((1,), dtype=np.float32)
    sdfg(A=a_test)
    assert np.allclose(a_validation, a_test)

if __name__ == '__main__':
    test_conditional_regular_if_true()
    test_conditional_regular_if_false()
