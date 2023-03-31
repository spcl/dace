# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the scalar fission pass. """

import pytest

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.scalar_fission import ScalarFission


def test_scalar_fission():
    """
    Test the scalar fission pass.
    This heavily relies on the scalar write shadow scopes pass, which is tested separately.

    :see: ``tests.passes.scalar_write_shadow_scopes_test``
    """
    # Construct the SDFG.
    sdfg = dace.SDFG('scalar_isedge')

    N = dace.symbol('N')
    sdfg.add_array('A', [N], dace.int32)
    sdfg.add_array('B', [N], dace.int32)
    sdfg.add_array('tmp', [1], dace.int32, transient=True)

    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1_1 = sdfg.add_state('loop_1_1')
    loop_1_2 = sdfg.add_state('loop_1_2')
    intermediate = sdfg.add_state('intermediate')
    guard_2 = sdfg.add_state('guard_2')
    loop_2_1 = sdfg.add_state('loop_2_1')
    loop_2_2 = sdfg.add_state('loop_2_2')
    end_state = sdfg.add_state('end')

    init_tasklet = init_state.add_tasklet('init', {}, {'out'}, 'out = 0')
    init_write = init_state.add_write('tmp')
    init_state.add_edge(init_tasklet, 'out', init_write, None, dace.Memlet('tmp[0]'))

    tmp1_tasklet = loop_1_1.add_tasklet('tmp1', {'a', 'b'}, {'out'}, 'out = a * b')
    tmp1_write = loop_1_1.add_write('tmp')
    a1_read = loop_1_1.add_read('A')
    b1_read = loop_1_1.add_read('B')
    loop_1_1.add_edge(a1_read, None, tmp1_tasklet, 'a', dace.Memlet('A[i]'))
    loop_1_1.add_edge(b1_read, None, tmp1_tasklet, 'b', dace.Memlet('B[i]'))
    loop_1_1.add_edge(tmp1_tasklet, 'out', tmp1_write, None, dace.Memlet('tmp[0]'))

    loop1_tasklet_1 = loop_1_2.add_tasklet('loop1_1', {'ap', 't'}, {'a'}, 'a = ap + 2 * t')
    loop1_tasklet_2 = loop_1_2.add_tasklet('loop1_2', {'bp', 't'}, {'b'}, 'b = bp - 2 * t')
    loop1_read_tmp = loop_1_2.add_read('tmp')
    loop1_read_a = loop_1_2.add_read('A')
    loop1_read_b = loop_1_2.add_read('B')
    loop1_write_a = loop_1_2.add_write('A')
    loop1_write_b = loop_1_2.add_write('B')
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_1, 't', dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_read_tmp, None, loop1_tasklet_2, 't', dace.Memlet('tmp[0]'))
    loop_1_2.add_edge(loop1_read_a, None, loop1_tasklet_1, 'ap', dace.Memlet('A[i + 1]'))
    loop_1_2.add_edge(loop1_read_b, None, loop1_tasklet_2, 'bp', dace.Memlet('B[i + 1]'))
    loop_1_2.add_edge(loop1_tasklet_1, 'a', loop1_write_a, None, dace.Memlet('A[i]'))
    loop_1_2.add_edge(loop1_tasklet_2, 'b', loop1_write_b, None, dace.Memlet('B[i]'))

    tmp2_tasklet = loop_2_1.add_tasklet('tmp2', {'a', 'b'}, {'out'}, 'out = a / b')
    tmp2_write = loop_2_1.add_write('tmp')
    a2_read = loop_2_1.add_read('A')
    b2_read = loop_2_1.add_read('B')
    loop_2_1.add_edge(a2_read, None, tmp2_tasklet, 'a', dace.Memlet('A[i + 1]'))
    loop_2_1.add_edge(b2_read, None, tmp2_tasklet, 'b', dace.Memlet('B[i + 1]'))
    loop_2_1.add_edge(tmp2_tasklet, 'out', tmp2_write, None, dace.Memlet('tmp[0]'))

    loop2_tasklet_1 = loop_2_2.add_tasklet('loop2_1', {'ap', 't'}, {'a'}, 'a = ap + t * t')
    loop2_tasklet_2 = loop_2_2.add_tasklet('loop2_2', {'bp', 't'}, {'b'}, 'b = bp - t * t')
    loop2_read_tmp = loop_2_2.add_read('tmp')
    loop2_read_a = loop_2_2.add_read('A')
    loop2_read_b = loop_2_2.add_read('B')
    loop2_write_a = loop_2_2.add_write('A')
    loop2_write_b = loop_2_2.add_write('B')
    loop_2_2.add_edge(loop2_read_tmp, None, loop2_tasklet_1, 't', dace.Memlet('tmp[0]'))
    loop_2_2.add_edge(loop2_read_tmp, None, loop2_tasklet_2, 't', dace.Memlet('tmp[0]'))
    loop_2_2.add_edge(loop2_read_a, None, loop2_tasklet_1, 'ap', dace.Memlet('A[i]'))
    loop_2_2.add_edge(loop2_read_b, None, loop2_tasklet_2, 'bp', dace.Memlet('B[i]'))
    loop_2_2.add_edge(loop2_tasklet_1, 'a', loop2_write_a, None, dace.Memlet('A[i + 1]'))
    loop_2_2.add_edge(loop2_tasklet_2, 'b', loop2_write_b, None, dace.Memlet('B[i + 1]'))

    sdfg.add_edge(init_state, guard_1, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, loop_1_1, dace.InterstateEdge(condition='i < (N - 1)'))
    tmp1_edge = dace.InterstateEdge(assignments={'j': 'tmp'})
    sdfg.add_edge(loop_1_1, loop_1_2, tmp1_edge)
    sdfg.add_edge(loop_1_2, guard_1, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_1, intermediate, dace.InterstateEdge(condition='i >= (N - 1)'))

    sdfg.add_edge(intermediate, guard_2, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_2, loop_2_1, dace.InterstateEdge(condition='i < (N - 1)'))
    tmp2_edge = dace.InterstateEdge(assignments={'j': 'tmp'})
    sdfg.add_edge(loop_2_1, loop_2_2, tmp2_edge)
    sdfg.add_edge(loop_2_2, guard_2, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard_2, end_state, dace.InterstateEdge(condition='i >= (N - 1)'))

    # Test the pass.
    pipeline = Pipeline([ScalarFission()])
    pipeline.apply_pass(sdfg, {})

    # Both interstate edges should be different now.
    assert tmp1_edge.assignments != tmp2_edge.assignments
    # There should now be 5 arrays in the SDFG, i.e. 2 more than before since two isolated scopes of tmp exist.
    assert len(sdfg.arrays.keys()) == 5
    # Assert all accesses per scope are identical.
    assert all([n.data == list(tmp1_edge.assignments.values())[0] for n in [tmp1_write, loop1_read_tmp]])
    assert all([n.data == list(tmp2_edge.assignments.values())[0] for n in [tmp2_write, loop2_read_tmp]])

def test_branch_subscopes_nofission():
    sdfg = dace.SDFG('branch_subscope_fission')
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_array('A', [2], dace.int32)
    sdfg.add_array('B', [1], dace.int32, transient=True)
    sdfg.add_array('C', [1], dace.int32)
    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    guard_2 = sdfg.add_state('guard_2')
    right1_state = sdfg.add_state('right1')
    right2_state = sdfg.add_state('right2')
    left2_state = sdfg.add_state('left2')
    merge_1 = sdfg.add_state('merge_1')
    merge_2 = sdfg.add_state('merge_2')
    guard_after = sdfg.add_state('guard_after')
    left_after = sdfg.add_state('left_after')
    right_after = sdfg.add_state('right_after')
    merge_after = sdfg.add_state('merge_after')
    first_assign = dace.InterstateEdge(assignments={'i': 'A[0]'})
    sdfg.add_edge(init_state, guard_1, first_assign)
    combined_assign_cond = dace.InterstateEdge(assignments={'i': 'A[1]'}, condition='i > 0')
    sdfg.add_edge(guard_1, guard_2, combined_assign_cond)
    right_cond = dace.InterstateEdge(condition='i <= 0')
    left_2_cond = dace.InterstateEdge(condition='i <= 0')
    right_2_cond = dace.InterstateEdge(condition='i > 0')
    sdfg.add_edge(guard_1, right1_state, right_cond)
    sdfg.add_edge(guard_2, right2_state, right_2_cond)
    sdfg.add_edge(guard_2, left2_state, left_2_cond)
    sdfg.add_edge(right1_state, merge_1, dace.InterstateEdge())
    sdfg.add_edge(right2_state, merge_2, dace.InterstateEdge())
    sdfg.add_edge(left2_state, merge_2, dace.InterstateEdge())
    sdfg.add_edge(merge_2, merge_1, dace.InterstateEdge())
    sdfg.add_edge(merge_1, guard_after, dace.InterstateEdge())
    after_cond_left = dace.InterstateEdge(condition='i <= 0')
    after_cond_right = dace.InterstateEdge(condition='i > 0')
    sdfg.add_edge(guard_after, left_after, after_cond_left)
    sdfg.add_edge(guard_after, right_after, after_cond_right)
    sdfg.add_edge(left_after, merge_after, dace.InterstateEdge())
    sdfg.add_edge(right_after, merge_after, dace.InterstateEdge())

    t1 = guard_1.add_tasklet('t1', {}, {'b'}, 'b = 1')
    a1 = guard_1.add_access('B')
    guard_1.add_edge(t1, 'b', a1, None, dace.Memlet('B[0]'))

    t0 = guard_2.add_tasklet('t1', {}, {'b'}, 'b = 0')
    a0 = guard_2.add_access('B')
    guard_2.add_edge(t0, 'b', a0, None, dace.Memlet('B[0]'))

    a2 = left2_state.add_access('B')
    t2 = left2_state.add_tasklet('t2', {'b'}, {'c'}, 'c = b')
    a3 = left2_state.add_access('C')
    left2_state.add_edge(a2, None, t2, 'b', dace.Memlet('B[0]'))
    left2_state.add_edge(t2, 'c', a3, None, dace.Memlet('C[0]'))

    a4 = right2_state.add_access('B')
    t3 = right2_state.add_tasklet('t3', {'b'}, {'c'}, 'c = b + 1')
    a5 = right2_state.add_access('C')
    right2_state.add_edge(a4, None, t3, 'b', dace.Memlet('B[0]'))
    right2_state.add_edge(t3, 'c', a5, None, dace.Memlet('C[0]'))

    a6 = right1_state.add_access('B')
    t4 = right1_state.add_tasklet('t4', {'b'}, {'c'}, 'c = b - 1')
    a7 = right1_state.add_access('C')
    right1_state.add_edge(a6, None, t4, 'b', dace.Memlet('B[0]'))
    right1_state.add_edge(t4, 'c', a7, None, dace.Memlet('C[0]'))

    a8 = left_after.add_access('B')
    t5 = left_after.add_tasklet('t5', {'b'}, {'c'}, 'c = b * 2')
    a9 = left_after.add_access('C')
    left_after.add_edge(a8, None, t5, 'b', dace.Memlet('B[0]'))
    left_after.add_edge(t5, 'c', a9, None, dace.Memlet('C[0]'))

    a10 = right_after.add_access('B')
    t6 = right_after.add_tasklet('t6', {'b'}, {'c'}, 'c = b * 3')
    a11 = right_after.add_access('C')
    right_after.add_edge(a10, None, t6, 'b', dace.Memlet('B[0]'))
    right_after.add_edge(t6, 'c', a11, None, dace.Memlet('C[0]'))

    Pipeline([ScalarFission()]).apply_pass(sdfg, {})

    assert set(sdfg.arrays.keys()) == {'A', 'B', 'C'}

def test_branch_subscopes_fission():
    sdfg = dace.SDFG('branch_subscope_fission')
    sdfg.add_symbol('i', dace.int32)
    sdfg.add_array('A', [2], dace.int32)
    sdfg.add_array('B', [1], dace.int32, transient=True)
    sdfg.add_array('C', [1], dace.int32)
    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    guard_2 = sdfg.add_state('guard_2')
    right1_state = sdfg.add_state('right1')
    right2_state = sdfg.add_state('right2')
    left2_state = sdfg.add_state('left2')
    merge_1 = sdfg.add_state('merge_1')
    merge_2 = sdfg.add_state('merge_2')
    guard_after = sdfg.add_state('guard_after')
    left_after = sdfg.add_state('left_after')
    right_after = sdfg.add_state('right_after')
    merge_after = sdfg.add_state('merge_after')
    first_assign = dace.InterstateEdge(assignments={'i': 'A[0]'})
    sdfg.add_edge(init_state, guard_1, first_assign)
    combined_assign_cond = dace.InterstateEdge(assignments={'i': 'A[1]'}, condition='i > 0')
    sdfg.add_edge(guard_1, guard_2, combined_assign_cond)
    right_cond = dace.InterstateEdge(condition='i <= 0')
    left_2_cond = dace.InterstateEdge(condition='i <= 0')
    right_2_cond = dace.InterstateEdge(condition='i > 0')
    sdfg.add_edge(guard_1, right1_state, right_cond)
    sdfg.add_edge(guard_2, right2_state, right_2_cond)
    sdfg.add_edge(guard_2, left2_state, left_2_cond)
    sdfg.add_edge(right1_state, merge_1, dace.InterstateEdge())
    sdfg.add_edge(right2_state, merge_2, dace.InterstateEdge())
    sdfg.add_edge(left2_state, merge_2, dace.InterstateEdge())
    sdfg.add_edge(merge_2, merge_1, dace.InterstateEdge())
    sdfg.add_edge(merge_1, guard_after, dace.InterstateEdge())
    after_cond_left = dace.InterstateEdge(condition='i <= 0')
    after_cond_right = dace.InterstateEdge(condition='i > 0')
    sdfg.add_edge(guard_after, left_after, after_cond_left)
    sdfg.add_edge(guard_after, right_after, after_cond_right)
    sdfg.add_edge(left_after, merge_after, dace.InterstateEdge())
    sdfg.add_edge(right_after, merge_after, dace.InterstateEdge())

    t1 = guard_1.add_tasklet('t1', {}, {'b'}, 'b = 1')
    a1 = guard_1.add_access('B')
    guard_1.add_edge(t1, 'b', a1, None, dace.Memlet('B[0]'))

    t0 = guard_2.add_tasklet('t1', {}, {'b'}, 'b = 0')
    a0 = guard_2.add_access('B')
    guard_2.add_edge(t0, 'b', a0, None, dace.Memlet('B[0]'))

    a2 = left2_state.add_access('B')
    t2 = left2_state.add_tasklet('t2', {'b'}, {'c'}, 'c = b')
    a3 = left2_state.add_access('C')
    left2_state.add_edge(a2, None, t2, 'b', dace.Memlet('B[0]'))
    left2_state.add_edge(t2, 'c', a3, None, dace.Memlet('C[0]'))

    a4 = right2_state.add_access('B')
    t3 = right2_state.add_tasklet('t3', {'b'}, {'c'}, 'c = b + 1')
    a5 = right2_state.add_access('C')
    right2_state.add_edge(a4, None, t3, 'b', dace.Memlet('B[0]'))
    right2_state.add_edge(t3, 'c', a5, None, dace.Memlet('C[0]'))

    a6 = right1_state.add_access('B')
    t4 = right1_state.add_tasklet('t4', {'b'}, {'c'}, 'c = b - 1')
    a7 = right1_state.add_access('C')
    right1_state.add_edge(a6, None, t4, 'b', dace.Memlet('B[0]'))
    right1_state.add_edge(t4, 'c', a7, None, dace.Memlet('C[0]'))

    t7 = guard_after.add_tasklet('t7', {}, {'b'}, 'b = 5')
    a12 = guard_after.add_access('B')
    guard_after.add_edge(t7, 'b', a12, None, dace.Memlet('B[0]'))

    a8 = left_after.add_access('B')
    t5 = left_after.add_tasklet('t5', {'b'}, {'c'}, 'c = b * 2')
    a9 = left_after.add_access('C')
    left_after.add_edge(a8, None, t5, 'b', dace.Memlet('B[0]'))
    left_after.add_edge(t5, 'c', a9, None, dace.Memlet('C[0]'))

    a10 = right_after.add_access('B')
    t6 = right_after.add_tasklet('t6', {'b'}, {'c'}, 'c = b * 3')
    a11 = right_after.add_access('C')
    right_after.add_edge(a10, None, t6, 'b', dace.Memlet('B[0]'))
    right_after.add_edge(t6, 'c', a11, None, dace.Memlet('C[0]'))

    a13 = merge_1.add_access('B')
    t8 = merge_1.add_tasklet('t8', {'b'}, {'c'}, 'c = b + 1')
    a14 = merge_1.add_access('C')
    merge_1.add_edge(a13, None, t8, 'b', dace.Memlet('B[0]'))
    merge_1.add_edge(t8, 'c', a14, None, dace.Memlet('C[0]'))

    Pipeline([ScalarFission()]).apply_pass(sdfg, {})

    assert set(sdfg.arrays.keys()) == {'A', 'B', 'C', 'B_0', 'B_1'}

if __name__ == '__main__':
    test_scalar_fission()
    test_branch_subscopes_nofission()
    test_branch_subscopes_fission()
