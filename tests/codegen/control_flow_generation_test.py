# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import dace
import numpy as np

from dace.sdfg.state import ConditionalBlock, ReturnBlock
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising


def test_for_loop_generation():
    N = dace.symbol('N')

    @dace.program
    def looptest(A: dace.float64[N]):
        for i in range(N):
            A[i] += 5

    sdfg: dace.SDFG = looptest.to_sdfg()
    assert 'for (' in sdfg.generate_code()[0].code

    A = np.random.rand(20)
    expected = A + 5
    sdfg(A=A, N=20)
    assert np.allclose(A, expected)


def test_edge_split_loop_generation():

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
        assert 'while (' in sdfg.generate_code()[0].code

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

    FixedPointPipeline([ControlFlowRaising()]).apply_pass(sdfg, {})
    if mode != 'SwitchCase':
        assert any(isinstance(node, ConditionalBlock) for node in sdfg.nodes())
    else:
        assert any(isinstance(node, ReturnBlock) for node in sdfg.nodes())

    sdfg.compile()


def test_single_outedge_branch():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('result', [1], dace.float64)
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    state2.add_edge(state2.add_tasklet('save', {}, {'out'}, 'out = 2'), 'out', state2.add_write('result'), None,
                    dace.Memlet('result'))

    sdfg.add_edge(state1, state2, dace.InterstateEdge('1 > 0'))

    FixedPointPipeline([ControlFlowRaising()]).apply_pass(sdfg, {})
    assert any(isinstance(node, ReturnBlock) for node in sdfg.nodes())

    sdfg.compile()
    res = np.random.rand(1)
    sdfg(result=res)
    assert np.allclose(res, 2)


def test_extraneous_goto():

    @dace.program
    def tester(a: dace.float64[20]):
        if a[0] < 0:
            a[1] = 1
        a[2] = 1

    sdfg = tester.to_sdfg(simplify=True)
    assert 'goto' not in sdfg.generate_code()[0].code


def test_extraneous_goto_nested():

    @dace.program
    def tester(a: dace.float64[20]):
        if a[0] < 0:
            if a[0] < 1:
                a[1] = 1
            else:
                a[1] = 2
        a[2] = 1

    sdfg = tester.to_sdfg(simplify=True)
    assert 'goto' not in sdfg.generate_code()[0].code


@pytest.mark.parametrize('detect_control_flow', (False, True))
def test_do_while_if_while(detect_control_flow):
    """
    Test a corner case that generates an infinite loop
    """
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('j', dace.int32)
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_scalar('i', dace.int32)
    sdfg.add_array('a', [1], dace.int32)
    init = sdfg.add_state(is_start_block=True)
    fini = sdfg.add_state()

    # Do-while guard
    do_guard = sdfg.add_state_after(init)
    do_inc = sdfg.add_state()

    # If that guards internal loop
    do_body_1 = sdfg.add_state()
    do_latch = sdfg.add_state()
    sdfg.add_edge(do_guard, do_body_1, dace.InterstateEdge('N > 0'))
    sdfg.add_edge(do_guard, do_latch, dace.InterstateEdge('N <= 0'))

    # While loop
    while_body = sdfg.add_state_after(do_body_1)
    while_increment = sdfg.add_state()
    sdfg.add_edge(while_body, do_latch, dace.InterstateEdge('i >= N'))
    sdfg.add_edge(while_body, while_increment, dace.InterstateEdge('i < N'))
    t = while_increment.add_tasklet('add1', {'inp'}, {'out'}, 'out = inp + 1')
    while_increment.add_edge(while_increment.add_read('i'), None, t, 'inp', dace.Memlet('i'))
    while_increment.add_edge(t, 'out', while_increment.add_write('i'), None, dace.Memlet('i'))
    sdfg.add_edge(while_increment, while_body, dace.InterstateEdge())

    # Contents of internal loop
    t = while_body.add_tasklet('add1', {'inp'}, {'out'}, 'out = inp + 1')
    while_body.add_edge(while_body.add_read('a'), None, t, 'inp', dace.Memlet('a[0]'))
    while_body.add_edge(t, 'out', while_body.add_write('a'), None, dace.Memlet('a[0]'))

    # Loop-back to do-while
    sdfg.add_edge(do_latch, fini, dace.InterstateEdge('j >= N'))
    sdfg.add_edge(do_latch, do_inc, dace.InterstateEdge('j < N', assignments=dict(j='j + 1')))
    sdfg.add_edge(do_inc, do_guard, dace.InterstateEdge())

    # Reset scalar in tasklet
    t = do_inc.add_tasklet('setzero', {}, {'out'}, 'out = 0')
    do_inc.add_edge(t, 'out', do_inc.add_write('i'), None, dace.Memlet('i'))

    # Test code
    a = np.zeros(1, dtype=np.int32)
    with dace.config.set_temporary('optimizer', 'detect_control_flow', value=detect_control_flow):
        sdfg(i=0, j=0, N=5, a=a)
        assert np.allclose(a, 6 * 6)


if __name__ == '__main__':
    test_for_loop_generation()
    test_edge_split_loop_generation()
    test_edge_sympy_function('FalseTrue')
    test_edge_sympy_function('TrueFalse')
    test_edge_sympy_function('SwitchCase')
    test_single_outedge_branch()
    test_extraneous_goto()
    test_extraneous_goto_nested()
    test_do_while_if_while(False)
    test_do_while_if_while(True)
