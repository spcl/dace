# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests loop raising trainsformations. """

import numpy as np
import pytest
import dace
from dace.memlet import Memlet
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_lifting import LoopLifting


def test_lift_regular_for_loop():
    sdfg = SDFG('regular_for')
    N = dace.symbol('N')
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_symbol('j', dace.int32)
    sdfg.add_symbol('k', dace.int32)
    sdfg.add_array('A', (N, ), dace.int32)
    start_state = sdfg.add_state('start', is_start_block=True)
    init_state = sdfg.add_state('init')
    guard_state = sdfg.add_state('guard')
    main_state = sdfg.add_state('loop_state')
    loop_exit = sdfg.add_state('exit')
    final_state = sdfg.add_state('final')
    sdfg.add_edge(start_state, init_state, InterstateEdge(assignments={'j': 0}))
    sdfg.add_edge(init_state, guard_state, InterstateEdge(assignments={'i': 0, 'k': 0}))
    sdfg.add_edge(guard_state, main_state, InterstateEdge(condition='i < N'))
    sdfg.add_edge(main_state, guard_state, InterstateEdge(assignments={'i': 'i + 2', 'j': 'j + 1'}))
    sdfg.add_edge(guard_state, loop_exit, InterstateEdge(condition='i >= N', assignments={'k': 2}))
    sdfg.add_edge(loop_exit, final_state, InterstateEdge())
    a_access = main_state.add_access('A')
    w_tasklet = main_state.add_tasklet('t1', {}, {'out'}, 'out = 1')
    main_state.add_edge(w_tasklet, 'out', a_access, None, Memlet('A[i]'))
    a_access_2 = loop_exit.add_access('A')
    w_tasklet_2 = loop_exit.add_tasklet('t1', {}, {'out'}, 'out = k')
    loop_exit.add_edge(w_tasklet_2, 'out', a_access_2, None, Memlet('A[1]'))
    a_access_3 = final_state.add_access('A')
    w_tasklet_3 = final_state.add_tasklet('t1', {}, {'out'}, 'out = j')
    final_state.add_edge(w_tasklet_3, 'out', a_access_3, None, Memlet('A[3]'))

    N = 30
    A = np.zeros((N, )).astype(np.int32)
    A_valid = np.zeros((N, )).astype(np.int32)
    sdfg(A=A_valid, N=N)
    sdfg.apply_transformations_repeated([LoopLifting])

    assert sdfg.using_explicit_control_flow == True
    assert any(isinstance(x, LoopRegion) for x in sdfg.nodes())

    sdfg(A=A, N=N)

    assert np.allclose(A_valid, A)


@pytest.mark.parametrize('increment_before_condition', (True, False))
def test_lift_loop_llvm_canonical(increment_before_condition):
    addendum = '_incr_before_cond' if increment_before_condition else ''
    sdfg = dace.SDFG('llvm_canonical' + addendum)
    N = dace.symbol('N')
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_symbol('j', dace.int32)
    sdfg.add_symbol('k', dace.int32)
    sdfg.add_array('A', (N, ), dace.int32)

    entry = sdfg.add_state('entry', is_start_block=True)
    guard = sdfg.add_state('guard')
    preheader = sdfg.add_state('preheader')
    body = sdfg.add_state('body')
    latch = sdfg.add_state('latch')
    loopexit = sdfg.add_state('loopexit')
    exitstate = sdfg.add_state('exitstate')

    sdfg.add_edge(entry, guard, InterstateEdge(assignments={'j': 0}))
    sdfg.add_edge(guard, exitstate, InterstateEdge(condition='N <= 0'))
    sdfg.add_edge(guard, preheader, InterstateEdge(condition='N > 0'))
    sdfg.add_edge(preheader, body, InterstateEdge(assignments={'i': 0, 'k': 0}))
    if increment_before_condition:
        sdfg.add_edge(body, latch, InterstateEdge(assignments={'i': 'i + 2', 'j': 'j + 1'}))
        sdfg.add_edge(latch, body, InterstateEdge(condition='i < N'))
        sdfg.add_edge(latch, loopexit, InterstateEdge(condition='i >= N', assignments={'k': 2}))
    else:
        sdfg.add_edge(body, latch, InterstateEdge(assignments={'j': 'j + 1'}))
        sdfg.add_edge(latch, body, InterstateEdge(condition='i < N - 2', assignments={'i': 'i + 2'}))
        sdfg.add_edge(latch, loopexit, InterstateEdge(condition='i >= N - 2', assignments={'k': 2}))
    sdfg.add_edge(loopexit, exitstate, InterstateEdge())

    a_access = body.add_access('A')
    w_tasklet = body.add_tasklet('t1', {}, {'out'}, 'out = 1')
    body.add_edge(w_tasklet, 'out', a_access, None, Memlet('A[i]'))
    a_access_2 = loopexit.add_access('A')
    w_tasklet_2 = loopexit.add_tasklet('t1', {}, {'out'}, 'out = k')
    loopexit.add_edge(w_tasklet_2, 'out', a_access_2, None, Memlet('A[1]'))
    a_access_3 = exitstate.add_access('A')
    w_tasklet_3 = exitstate.add_tasklet('t1', {}, {'out'}, 'out = j')
    exitstate.add_edge(w_tasklet_3, 'out', a_access_3, None, Memlet('A[3]'))

    N = 30
    A = np.zeros((N, )).astype(np.int32)
    A_valid = np.zeros((N, )).astype(np.int32)
    sdfg(A=A_valid, N=N)
    sdfg.apply_transformations_repeated([LoopLifting])

    assert sdfg.using_explicit_control_flow == True
    assert any(isinstance(x, LoopRegion) for x in sdfg.nodes())

    sdfg(A=A, N=N)

    assert np.allclose(A_valid, A)


def test_lift_loop_llvm_canonical_while():
    sdfg = dace.SDFG('llvm_canonical_while')
    N = dace.symbol('N')
    sdfg.add_symbol('j', dace.int32)
    sdfg.add_symbol('k', dace.int32)
    sdfg.add_array('A', (N, ), dace.int32)
    sdfg.add_scalar('i', dace.int32, transient=True)

    entry = sdfg.add_state('entry', is_start_block=True)
    guard = sdfg.add_state('guard')
    preheader = sdfg.add_state('preheader')
    body_1 = sdfg.add_state('body_1')
    body_2 = sdfg.add_state('body_2')
    latch = sdfg.add_state('latch')
    loopexit = sdfg.add_state('loopexit')
    exitstate = sdfg.add_state('exitstate')

    sdfg.add_edge(entry, guard, InterstateEdge(assignments={'j': 0}))
    sdfg.add_edge(guard, exitstate, InterstateEdge(condition='N <= 0'))
    sdfg.add_edge(guard, preheader, InterstateEdge(condition='N > 0'))
    sdfg.add_edge(preheader, body_1, InterstateEdge(assignments={'k': 0}))
    sdfg.add_edge(body_1, body_2, InterstateEdge())
    sdfg.add_edge(body_2, latch, InterstateEdge(assignments={'j': 'j + 1'}))
    sdfg.add_edge(latch, body_1, InterstateEdge(condition='i < N - 2'))
    sdfg.add_edge(latch, loopexit, InterstateEdge(condition='i >= N - 2', assignments={'k': 2}))
    sdfg.add_edge(loopexit, exitstate, InterstateEdge())

    i_init_write = entry.add_access('i')
    iw_init_tasklet = entry.add_tasklet('ti', {}, {'out'}, 'out = 0')
    entry.add_edge(iw_init_tasklet, 'out', i_init_write, None, Memlet('i[0]'))
    a_access = body_1.add_access('A')
    w_tasklet = body_1.add_tasklet('t1', {}, {'out'}, 'out = 1')
    body_1.add_edge(w_tasklet, 'out', a_access, None, Memlet('A[i]'))
    i_read = body_2.add_access('i')
    i_write = body_2.add_access('i')
    iw_tasklet = body_2.add_tasklet('t2', {'in1'}, {'out'}, 'out = in1 + 2')
    body_2.add_edge(i_read, None, iw_tasklet, 'in1', Memlet('i[0]'))
    body_2.add_edge(iw_tasklet, 'out', i_write, None, Memlet('i[0]'))
    a_access_2 = loopexit.add_access('A')
    w_tasklet_2 = loopexit.add_tasklet('t1', {}, {'out'}, 'out = k')
    loopexit.add_edge(w_tasklet_2, 'out', a_access_2, None, Memlet('A[1]'))
    a_access_3 = exitstate.add_access('A')
    w_tasklet_3 = exitstate.add_tasklet('t1', {}, {'out'}, 'out = j')
    exitstate.add_edge(w_tasklet_3, 'out', a_access_3, None, Memlet('A[3]'))

    N = 30
    A = np.zeros((N, )).astype(np.int32)
    A_valid = np.zeros((N, )).astype(np.int32)
    sdfg(A=A_valid, N=N)
    sdfg.apply_transformations_repeated([LoopLifting])

    assert sdfg.using_explicit_control_flow == True
    assert any(isinstance(x, LoopRegion) for x in sdfg.nodes())

    sdfg(A=A, N=N)

    assert np.allclose(A_valid, A)


def test_do_while():
    sdfg = SDFG('do_while')
    N = dace.symbol('N')
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_symbol('j', dace.int32)
    sdfg.add_symbol('k', dace.int32)
    sdfg.add_array('A', (N, ), dace.int32)
    start_state = sdfg.add_state('start', is_start_block=True)
    init_state = sdfg.add_state('init')
    guard_state = sdfg.add_state('guard')
    main_state = sdfg.add_state('loop_state')
    loop_exit = sdfg.add_state('exit')
    final_state = sdfg.add_state('final')
    sdfg.add_edge(start_state, init_state, InterstateEdge(assignments={'j': 0}))
    sdfg.add_edge(init_state, main_state, InterstateEdge(assignments={'i': 0, 'k': 0}))
    sdfg.add_edge(main_state, guard_state, InterstateEdge(assignments={'i': 'i + 2', 'j': 'j + 1'}))
    sdfg.add_edge(guard_state, main_state, InterstateEdge(condition='i < N'))
    sdfg.add_edge(guard_state, loop_exit, InterstateEdge(condition='i >= N', assignments={'k': 2}))
    sdfg.add_edge(loop_exit, final_state, InterstateEdge())
    a_access = main_state.add_access('A')
    w_tasklet = main_state.add_tasklet('t1', {}, {'out'}, 'out = 1')
    main_state.add_edge(w_tasklet, 'out', a_access, None, Memlet('A[i]'))
    a_access_2 = loop_exit.add_access('A')
    w_tasklet_2 = loop_exit.add_tasklet('t1', {}, {'out'}, 'out = k')
    loop_exit.add_edge(w_tasklet_2, 'out', a_access_2, None, Memlet('A[1]'))
    a_access_3 = final_state.add_access('A')
    w_tasklet_3 = final_state.add_tasklet('t1', {}, {'out'}, 'out = j')
    final_state.add_edge(w_tasklet_3, 'out', a_access_3, None, Memlet('A[3]'))

    N = 30
    A = np.zeros((N, )).astype(np.int32)
    A_valid = np.zeros((N, )).astype(np.int32)
    sdfg(A=A_valid, N=N)
    sdfg.apply_transformations_repeated([LoopLifting])

    assert sdfg.using_explicit_control_flow == True
    assert any(isinstance(x, LoopRegion) for x in sdfg.nodes())

    sdfg(A=A, N=N)

    assert np.allclose(A_valid, A)


def test_inverted_loop_with_additional_increment_assignment():
    sdfg = SDFG('inverted_loop_with_additional_increment_assignment')
    N = dace.symbol('N')
    sdfg.add_scalar('i', dace.int32, transient=True)
    sdfg.add_symbol('k', dace.int32)
    sdfg.add_array('A', (N, ), dace.int32)
    a_state = sdfg.add_state('a_state', is_start_block=True)
    b_state = sdfg.add_state('b_state')
    c_state = sdfg.add_state('c_state')
    d_state = sdfg.add_state('d_state')
    sdfg.add_edge(a_state, b_state, InterstateEdge(assignments={'k': 0}))
    sdfg.add_edge(b_state, c_state, InterstateEdge())
    sdfg.add_edge(c_state, b_state, InterstateEdge(condition='i < N', assignments={'k': 'k + 1'}))
    sdfg.add_edge(c_state, d_state, InterstateEdge(condition='i >= N'))
    a_access = b_state.add_access('A')
    w_tasklet = b_state.add_tasklet('t1', {}, {'out'}, 'out = 1')
    b_state.add_edge(w_tasklet, 'out', a_access, None, Memlet('A[i]'))
    i_read = c_state.add_access('i')
    i_write = c_state.add_access('i')
    iw_tasklet = c_state.add_tasklet('t2', {'in1'}, {'out'}, 'out = in1 + 2')
    c_state.add_edge(i_read, None, iw_tasklet, 'in1', Memlet('i[0]'))
    c_state.add_edge(iw_tasklet, 'out', i_write, None, Memlet('i[0]'))
    a_access_2 = d_state.add_access('A')
    w_tasklet_2 = d_state.add_tasklet('t1', {}, {'out'}, 'out = k')
    d_state.add_edge(w_tasklet_2, 'out', a_access_2, None, Memlet('A[1]'))

    N = 30
    A = np.zeros((N, )).astype(np.int32)
    A_valid = np.zeros((N, )).astype(np.int32)
    sdfg(A=A_valid, N=N)

    sdfg.apply_transformations_repeated([LoopLifting])

    assert sdfg.using_explicit_control_flow == True
    assert any(isinstance(x, LoopRegion) for x in sdfg.nodes())

    sdfg(A=A, N=N)

    assert np.allclose(A_valid, A)


def test_lift_previously_illegal_for_loop():
    sdfg = dace.SDFG('looptest')
    sdfg.add_array('A', [20], dace.float64)
    init = sdfg.add_state()
    guard = sdfg.add_state()
    loop = sdfg.add_state()
    end = sdfg.add_state()
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments=dict(i='0')))
    sdfg.add_edge(guard, loop, dace.InterstateEdge(condition='i < 20', assignments=dict(j='i')))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= 20'))
    sdfg.add_edge(loop, guard, dace.InterstateEdge(assignments=dict(i='i + 1')))

    r = loop.add_read('A')
    t = loop.add_tasklet('add', {'a'}, {'out'}, 'out = a + 5')
    w = loop.add_write('A')
    loop.add_edge(r, None, t, 'a', dace.Memlet('A[j]'))
    loop.add_edge(t, 'out', w, None, dace.Memlet('A[j]'))

    sdfg.apply_transformations_repeated([LoopLifting])
    assert any(isinstance(x, LoopRegion) for x in sdfg.nodes())

    A = np.random.rand(20)
    expected = A + 5
    sdfg(A=A)
    assert np.allclose(A, expected)


if __name__ == '__main__':
    test_lift_regular_for_loop()
    test_lift_loop_llvm_canonical(True)
    test_lift_loop_llvm_canonical(False)
    test_lift_loop_llvm_canonical_while()
    test_do_while()
    test_inverted_loop_with_additional_increment_assignment()
    test_lift_previously_illegal_for_loop()
