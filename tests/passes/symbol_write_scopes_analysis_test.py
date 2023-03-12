# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the symbol write scopes analysis pass. """

import pytest

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.analysis import SymbolWriteScopes, SymbolScopeDict

def test_loop_iter_symbol_reused_split():
    """
    Test the symbol write scopes pass with reused loop iteration variables where the exit condition of the first loop
    and the init assignment of the second loop are on different interstate edges.
    """
    # Construct the SDFG.
    sdfg = dace.SDFG('symbol')

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

    loop_1_init_edge = dace.InterstateEdge(assignments={'i': 0})
    sdfg.add_edge(init_state, guard_1, loop_1_init_edge)
    loop_1_in_condition_edge = dace.InterstateEdge(condition='i < (N - 1)')
    sdfg.add_edge(guard_1, loop_1_1, loop_1_in_condition_edge)
    sdfg.add_edge(loop_1_1, loop_1_2, dace.InterstateEdge())
    loop_1_inc_edge = dace.InterstateEdge(assignments={'i': 'i + 1'})
    sdfg.add_edge(loop_1_2, guard_1, loop_1_inc_edge)
    loop_1_out_condition_edge = dace.InterstateEdge(condition='i >= (N - 1)')
    sdfg.add_edge(guard_1, intermediate, loop_1_out_condition_edge)

    loop_2_init_edge = dace.InterstateEdge(assignments={'i': 0})
    sdfg.add_edge(intermediate, guard_2, loop_2_init_edge)
    loop_2_in_condition_edge = dace.InterstateEdge(condition='i < (N - 1)')
    sdfg.add_edge(guard_2, loop_2_1, loop_2_in_condition_edge)
    sdfg.add_edge(loop_2_1, loop_2_2, dace.InterstateEdge())
    loop_2_inc_edge = dace.InterstateEdge(assignments={'i': 'i + 1'})
    sdfg.add_edge(loop_2_2, guard_2, loop_2_inc_edge)
    loop_2_out_condition_edge = dace.InterstateEdge(condition='i >= (N - 1)')
    sdfg.add_edge(guard_2, end_state, loop_2_out_condition_edge)

    # Test the pass.
    pipeline = Pipeline([SymbolWriteScopes()])
    results = pipeline.apply_pass(sdfg, {})[SymbolWriteScopes.__name__]

    sdfg_results: SymbolScopeDict = results[0]
    assert set(sdfg_results.keys()) == {'i', 'N'}
    assert set(sdfg_results['N'].keys()) == {None}
    assert set(sdfg_results['i'].keys()) == {loop_1_init_edge, loop_2_init_edge}
    assert sdfg_results['i'][loop_1_init_edge] == {
        loop_1_in_condition_edge, loop_1_inc_edge, loop_1_out_condition_edge, loop_1_2, loop_1_1
    }
    assert sdfg_results['i'][loop_2_init_edge] == {
        loop_2_in_condition_edge, loop_2_inc_edge, loop_2_out_condition_edge, loop_2_2, loop_2_1
    }


def test_loop_iter_symbol_reused_fused():
    """
    Test the symbol write scopes pass with reused loop iteration variables where the exit condition of the first loop
    and the init assignment of the second loop are on the same interstate edge.
    """
    # Construct the SDFG.
    sdfg = dace.SDFG('symbol')

    N = dace.symbol('N')
    sdfg.add_array('A', [N], dace.int32)
    sdfg.add_array('B', [N], dace.int32)
    sdfg.add_array('tmp', [1], dace.int32, transient=True)

    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1_1 = sdfg.add_state('loop_1_1')
    loop_1_2 = sdfg.add_state('loop_1_2')
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

    loop_1_init_edge = dace.InterstateEdge(assignments={'i': 0})
    sdfg.add_edge(init_state, guard_1, loop_1_init_edge)
    loop_1_in_condition_edge = dace.InterstateEdge(condition='i < (N - 1)')
    sdfg.add_edge(guard_1, loop_1_1, loop_1_in_condition_edge)
    sdfg.add_edge(loop_1_1, loop_1_2, dace.InterstateEdge())
    loop_1_inc_edge = dace.InterstateEdge(assignments={'i': 'i + 1'})
    sdfg.add_edge(loop_1_2, guard_1, loop_1_inc_edge)
    shared_cond_init_edge = dace.InterstateEdge(condition='i >= (N - 1)', assignments={'i': 0})
    sdfg.add_edge(guard_1, guard_2, shared_cond_init_edge)

    loop_2_in_condition_edge = dace.InterstateEdge(condition='i < (N - 1)')
    sdfg.add_edge(guard_2, loop_2_1, loop_2_in_condition_edge)
    sdfg.add_edge(loop_2_1, loop_2_2, dace.InterstateEdge())
    loop_2_inc_edge = dace.InterstateEdge(assignments={'i': 'i + 1'})
    sdfg.add_edge(loop_2_2, guard_2, loop_2_inc_edge)
    loop_2_out_condition_edge = dace.InterstateEdge(condition='i >= (N - 1)')
    sdfg.add_edge(guard_2, end_state, loop_2_out_condition_edge)

    # Test the pass.
    pipeline = Pipeline([SymbolWriteScopes()])
    results = pipeline.apply_pass(sdfg, {})[SymbolWriteScopes.__name__]

    sdfg_results: SymbolScopeDict = results[0]
    assert set(sdfg_results.keys()) == {'i', 'N'}
    assert set(sdfg_results['N'].keys()) == {None}
    assert set(sdfg_results['i'].keys()) == {loop_1_init_edge, shared_cond_init_edge}
    assert sdfg_results['i'][loop_1_init_edge] == {
        loop_1_in_condition_edge, loop_1_inc_edge, shared_cond_init_edge, loop_1_2, loop_1_1
    }
    assert sdfg_results['i'][shared_cond_init_edge] == {
        loop_2_in_condition_edge, loop_2_inc_edge, loop_2_out_condition_edge, loop_2_2, loop_2_1
    }


def test_branch_subscope():
    sdfg = dace.SDFG('branch_subscope')
    sdfg.add_array('A', [2], dace.int32)
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

    # Test the pass.
    pipeline = Pipeline([SymbolWriteScopes()])
    results = pipeline.apply_pass(sdfg, {})[SymbolWriteScopes.__name__]

    sdfg_results: SymbolScopeDict = results[0]
    assert set(sdfg_results.keys()) == {'i'}
    assert set(sdfg_results['i'].keys()) == {first_assign, combined_assign_cond}
    assert sdfg_results['i'][first_assign] == {right_cond, combined_assign_cond, after_cond_right, after_cond_left}
    assert sdfg_results['i'][combined_assign_cond] == {right_2_cond, left_2_cond}


if __name__ == '__main__':
    test_loop_iter_symbol_reused_split()
    test_loop_iter_symbol_reused_fused()
    test_branch_subscope()
