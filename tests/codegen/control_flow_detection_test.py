# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from math import exp

import pytest
import dace
import numpy as np


def test_for_loop_detection():
    N = dace.symbol('N')

    @dace.program
    def looptest(A: dace.float64[N]):
        for i in range(N):
            A[i] += 5

    sdfg: dace.SDFG = looptest.to_sdfg()
    if dace.Config.get_bool('optimizer', 'detect_control_flow'):
        assert 'for (' in sdfg.generate_code()[0].code

    A = np.random.rand(20)
    expected = A + 5
    sdfg(A=A, N=20)
    assert np.allclose(A, expected)


def test_invalid_for_loop_detection():
    sdfg = dace.SDFG('looptest')
    sdfg.add_array('A', [20], dace.float64)
    init = sdfg.add_state()
    guard = sdfg.add_state()
    loop = sdfg.add_state()
    end = sdfg.add_state()
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments=dict(i='0')))
    # Invalid: Edge between guard and loop state must not have assignments
    # This edge will be split in code generation
    sdfg.add_edge(guard, loop, dace.InterstateEdge(condition='i < 20', assignments=dict(j='i')))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= 20'))
    sdfg.add_edge(loop, guard, dace.InterstateEdge(assignments=dict(i='i + 1')))

    r = loop.add_read('A')
    t = loop.add_tasklet('add', {'a'}, {'out'}, 'out = a + 5')
    w = loop.add_write('A')
    loop.add_edge(r, None, t, 'a', dace.Memlet('A[j]'))
    loop.add_edge(t, 'out', w, None, dace.Memlet('A[j]'))

    # If edge was split successfully, a for loop will be generated
    if dace.Config.get_bool('optimizer', 'detect_control_flow'):
        assert 'for (' in sdfg.generate_code()[0].code
    A = np.random.rand(20)
    expected = A + 5
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_edge_split_loop_detection():
    @dace.program
    def looptest():
        A = dace.ndarray([10], dtype=dace.int32)
        i = 0
        while (i < 10):
            A[i] = i
            i += 2
        return A

    sdfg: dace.SDFG = looptest.to_sdfg(simplify=True)
    if dace.Config.get_bool('optimizer', 'detect_control_flow'):
        assert 'for (' in sdfg.generate_code()[0].code

    A = looptest()
    A_ref = np.array([0, 0, 2, 0, 4, 0, 6, 0, 8, 0], dtype=np.int32)
    assert (np.array_equal(A[::2], A_ref[::2]))

@pytest.mark.parametrize('mode', ('FalseTrue', 'TrueFalse', 'SwitchCase'))
def test_edge_sympy_function(mode):
    sdfg = dace.SDFG("test")
    sdfg.add_symbol('N', stype=dace.int32)
    sdfg.add_symbol('cnd', stype=dace.int32)
    
    state_start = sdfg.add_state()
    state_condition = sdfg.add_state()
    state_br1 = sdfg.add_state()
    state_br1_1 = sdfg.add_state_after(state_br1)
    state_br2 = sdfg.add_state()
    state_br2_1 = sdfg.add_state_after(state_br2)
    state_merge = sdfg.add_state()

    sdfg.add_edge(state_start, state_condition, dace.InterstateEdge())  #assignments=dict(cnd=1)))
    if mode == 'FalseTrue':
        sdfg.add_edge(state_condition, state_br1, dace.InterstateEdge('Ne(cnd, 0)', dict(N=2)))
        sdfg.add_edge(state_condition, state_br2, dace.InterstateEdge('Eq(cnd, 0)', dict(N=3)))
    elif mode == 'TrueFalse':
        sdfg.add_edge(state_condition, state_br1, dace.InterstateEdge('Eq(cnd, 0)', dict(N=2)))
        sdfg.add_edge(state_condition, state_br2, dace.InterstateEdge('Ne(cnd, 0)', dict(N=3)))
    elif mode == 'SwitchCase':
        sdfg.add_edge(state_condition, state_br1, dace.InterstateEdge('Eq(cnd, 1)', dict(N=2)))
        sdfg.add_edge(state_condition, state_br2, dace.InterstateEdge('Eq(cnd, 0)', dict(N=3)))

    sdfg.add_edge(state_br1_1, state_merge, dace.InterstateEdge())
    sdfg.add_edge(state_br2_1, state_merge, dace.InterstateEdge())

    sdfg.compile()


if __name__ == '__main__':
    test_for_loop_detection()
    test_invalid_for_loop_detection()
    test_edge_split_loop_detection()
    test_edge_sympy_function('FalseTrue')
    test_edge_sympy_function('TrueFalse')
    test_edge_sympy_function('SwitchCase')
