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


def test_lift_for_loop_with_dataflow_guard():
    """A guard block with dataflow executes ONCE MORE than the loop body (it
    also runs on the final, failing condition check). The lift must preserve
    that extra execution; the former guard-conditional construction dropped it
    (and its "loop not executed" branch condition was accidentally constant
    false)."""
    sdfg = SDFG('for_with_dataflow_guard')
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_array('A', (11, ), dace.int32)
    sdfg.add_array('C', (10, ), dace.int32)
    init_state = sdfg.add_state('init', is_start_block=True)
    guard_state = sdfg.add_state('guard')
    main_state = sdfg.add_state('loop_state')
    end_state = sdfg.add_state('end')
    sdfg.add_edge(init_state, guard_state, InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_state, main_state, InterstateEdge(condition='i < 10'))
    sdfg.add_edge(main_state, guard_state, InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_state, end_state, InterstateEdge(condition='i >= 10'))
    # Dataflow in the GUARD: A[i] = i - runs for i = 0..10 (11 times).
    ga = guard_state.add_access('A')
    gt = guard_state.add_tasklet('t_guard', {}, {'out'}, 'out = i')
    guard_state.add_edge(gt, 'out', ga, None, Memlet('A[i]'))
    # Dataflow in the body: C[i] = 1 - runs for i = 0..9 (10 times).
    ba = main_state.add_access('C')
    bt = main_state.add_tasklet('t_body', {}, {'out'}, 'out = 1')
    main_state.add_edge(bt, 'out', ba, None, Memlet('C[i]'))

    sdfg.apply_transformations_repeated([LoopLifting])
    assert sdfg.using_explicit_control_flow == True
    assert any(isinstance(x, LoopRegion) for x in sdfg.all_control_flow_regions())

    A = np.full((11, ), -1, dtype=np.int32)
    C = np.zeros((10, ), dtype=np.int32)
    sdfg(A=A, C=C)
    assert np.allclose(A, np.arange(11, dtype=np.int32))
    assert np.allclose(C, 1)


@pytest.mark.parametrize('extra_state', (False, True))
def test_lift_while_loop_with_data_dependent_condition(extra_state):
    """A while-style loop whose condition reads a transient scalar that the
    guard block's dataflow computes: no lifted condition check may execute
    before the dataflow that defines it (the former construction emitted a
    pre-checked loop that read the scalar uninitialized - undefined behavior
    that surfaced as a wrongly-skipped loop depending on stack garbage).
    Mirrors ``tests/passes/constant_propagation_test.py::
    test_dependency_change_same_edge``, where the miscompilation was found."""
    from dace.sdfg.state import ConditionalBlock

    sdfg = SDFG('while_data_dependent_cond' + ('_extra' if extra_state else ''))
    sdfg.add_array('a', [1], dace.int64)
    sdfg.add_scalar('cont', dace.int64, transient=True)
    init = sdfg.add_state('init', is_start_block=True)
    entry = sdfg.add_state('entry')
    body = sdfg.add_state('body')
    latch = sdfg.add_state('latch')
    final = sdfg.add_state('final')
    sdfg.add_edge(init, entry, InterstateEdge(assignments=dict(i60='0')))
    sdfg.add_edge(entry, body, InterstateEdge(assignments=dict(i61='i60 + 1', i17='i60 * 12')))
    sdfg.add_edge(body, final, InterstateEdge('cont'))
    sdfg.add_edge(body, latch, InterstateEdge('not cont', dict(i60='i61')))
    if not extra_state:
        sdfg.add_edge(latch, body, InterstateEdge(assignments=dict(i61='i60 + 1', i17='i60 * 12')))
    else:
        extra = sdfg.add_state('extra')
        sdfg.add_edge(latch, extra, InterstateEdge(assignments=dict(i61='i60 + 1', i17='i60 * 12')))
        sdfg.add_edge(extra, body, InterstateEdge(assignments=dict(i18='i60 + i61')))
    t = body.add_tasklet('add', {'inp'}, {'out', 'c'}, 'out = inp + i17; c = i61 == 10')
    body.add_edge(body.add_read('a'), None, t, 'inp', Memlet('a[0]'))
    body.add_edge(t, 'out', body.add_write('a'), None, Memlet('a[0]'))
    body.add_edge(t, 'c', body.add_write('cont'), None, Memlet('cont[0]'))

    sdfg.apply_transformations_repeated([LoopLifting])
    # Only the extra_state variant matches a liftable pattern today; the other
    # stays a state machine (and must still execute correctly below).
    if extra_state:
        assert any(isinstance(x, LoopRegion) for x in sdfg.all_control_flow_regions())

    # No conditional branch may have a constant-false condition (the dead
    # "loop not executed" branch was guarded by `(not 1)`).
    for region in sdfg.all_control_flow_regions():
        if isinstance(region, ConditionalBlock):
            for cond, _ in region.branches:
                if cond is not None:
                    assert cond.as_string.strip() not in ('(not 1)', 'not 1', '0', 'false', 'False')

    # Reference: python equivalent (the loop body must run for i60 = 0..9).
    ref = sum(i60 * 12 for i60 in range(10))

    # Force non-zero initialization of automatic variables so an uninitialized
    # condition read fails deterministically instead of depending on stack
    # garbage (the bug escaped detection for a long time exactly because fresh
    # stack memory is usually zero).
    from dace.codegen.exceptions import CompilationError
    from dace.config import Config, set_temporary
    hardened_args = Config.get('compiler', 'cpu', 'args') + ' -ftrivial-auto-var-init=pattern'
    with set_temporary('compiler', 'cpu', 'args', value=hardened_args):
        a = np.zeros([1], np.int64)
        try:
            sdfg(a=a)
        except CompilationError:
            pytest.skip('CPU compiler does not support -ftrivial-auto-var-init')
    assert a[0] == ref


if __name__ == '__main__':
    test_lift_regular_for_loop()
    test_lift_loop_llvm_canonical(True)
    test_lift_loop_llvm_canonical(False)
    test_lift_loop_llvm_canonical_while()
    test_do_while()
    test_inverted_loop_with_additional_increment_assignment()
    test_lift_previously_illegal_for_loop()
    test_lift_for_loop_with_dataflow_guard()
    test_lift_while_loop_with_data_dependent_condition(False)
    test_lift_while_loop_with_data_dependent_condition(True)
