# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.sdfg.region_inline import inline
from dace.sdfg.state import ConditionalRegion

def create_conditional_sdfg():
    sdfg = dace.SDFG('regular_if')
    sdfg.add_symbol('cond', dace.int32)
    state0 = sdfg.add_state('state0', is_start_block=True)
    if1 = ConditionalRegion(label='if1', condition_expr='cond', condition_else_expr='not cond')
    sdfg.add_array('A', [1], dace.float32)
    sdfg.add_node(if1)
    state1 = if1.add_state('state1', is_start_block=True)
    acc_a = state1.add_access('A')
    t1 = state1.add_tasklet('t1', None, {'a'}, 'a = 20')
    state1.add_edge(t1, 'a', acc_a, None, dace.Memlet('A[0]'))

    if1.init_else_branch()
    state2 = if1.else_branch.add_state('state2', is_start_block=True)
    acc_a2 = state2.add_access('A')
    t2 = state2.add_tasklet('t2', None, {'a'}, 'a = 30')
    state2.add_edge(t2, 'a', acc_a2, None, dace.Memlet('A[0]'))

    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, if1, dace.InterstateEdge())
    sdfg.add_edge(if1, state3, dace.InterstateEdge())
    return sdfg

def test_conditional_regular_if_true():
    sdfg = create_conditional_sdfg()

    assert sdfg.is_valid()

    inline(sdfg)
    
    a_validation = np.array([20], dtype=np.float32)
    a_test = np.zeros((1,), dtype=np.float32)
    sdfg(A=a_test, cond=1)
    assert np.allclose(a_validation, a_test)

def test_conditional_regular_if_false():
    sdfg = create_conditional_sdfg()

    assert sdfg.is_valid()

    inline(sdfg)
    
    a_validation = np.array([30], dtype=np.float32)
    a_test = np.zeros((1,), dtype=np.float32)
    sdfg(A=a_test, cond=0)
    assert np.allclose(a_validation, a_test)


if __name__ == '__main__':
    test_conditional_regular_if_true()
    test_conditional_regular_if_false()