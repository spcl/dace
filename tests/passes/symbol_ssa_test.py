# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the symbol write scopes analysis pass. """

import dace
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes.symbol_ssa import StrictSymbolSSA, SymbolSSA


def test_loop_iter_symbol_reused_split():
    """
    Test the symbol write scopes pass with reused loop iteration variables where the exit condition of the first loop
    and the init assignment of the second loop are on different interstate edges.
    """
    # Construct the SDFG.
    sdfg = dace.SDFG('symbol')

    N = dace.symbol('N')
    sdfg.add_symbol('i', dace.int32)
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
    pipeline = FixedPointPipeline([StrictSymbolSSA()])
    pipeline.apply_pass(sdfg, {})

    assert set(sdfg.symbols.keys()) == {'N', 'i', 'i_0', 'i_1'}

    assert set(loop_1_init_edge.assignments.keys()) == {'i_0'}
    assert loop_1_in_condition_edge.free_symbols == {'i_0', 'N'}
    assert loop_1_out_condition_edge.free_symbols == {'i_0', 'N'}
    assert 'i_0' in loop_1_inc_edge.assignments
    assert 'i_0' in loop_1_inc_edge.assignments['i_0']
    assert set(loop_1_inc_edge.assignments.keys()) == {'i_0'}

    assert set(loop_2_init_edge.assignments.keys()) == {'i_1'}
    assert loop_2_in_condition_edge.free_symbols == {'i_1', 'N'}
    assert loop_2_out_condition_edge.free_symbols == {'i_1', 'N'}
    assert 'i_1' in loop_2_inc_edge.assignments
    assert 'i_1' in loop_2_inc_edge.assignments['i_1']
    assert set(loop_2_inc_edge.assignments.keys()) == {'i_1'}

    assert loop_1_1.free_symbols == {'i_0', 'N'}
    assert loop_1_2.free_symbols == {'i_0', 'N'}
    assert loop_2_1.free_symbols == {'i_1', 'N'}
    assert loop_2_2.free_symbols == {'i_1', 'N'}


def test_loop_iter_symbol_reused_fused():
    """
    Test the symbol write scopes pass with reused loop iteration variables where the exit condition of the first loop
    and the init assignment of the second loop are on the same interstate edge.
    """
    # Construct the SDFG.
    sdfg = dace.SDFG('symbol')

    N = dace.symbol('N')
    sdfg.add_symbol('i', dace.int32)
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
    pipeline = FixedPointPipeline([StrictSymbolSSA()])
    pipeline.apply_pass(sdfg, {})

    assert set(sdfg.symbols.keys()) == {'N', 'i', 'i_0', 'i_1'}

    assert set(loop_1_init_edge.assignments.keys()) == {'i_0'}
    assert loop_1_in_condition_edge.free_symbols == {'i_0', 'N'}
    assert 'i_0' in loop_1_inc_edge.assignments
    assert 'i_0' in loop_1_inc_edge.assignments['i_0']
    assert set(loop_1_inc_edge.assignments.keys()) == {'i_0'}
    assert 'i_0' in shared_cond_init_edge.free_symbols

    assert set(shared_cond_init_edge.assignments.keys()) == {'i_1'}
    assert loop_2_in_condition_edge.free_symbols == {'i_1', 'N'}
    assert loop_2_out_condition_edge.free_symbols == {'i_1', 'N'}
    assert 'i_1' in loop_2_inc_edge.assignments
    assert 'i_1' in loop_2_inc_edge.assignments['i_1']
    assert set(loop_2_inc_edge.assignments.keys()) == {'i_1'}

    assert loop_1_1.free_symbols == {'i_0', 'N'}
    assert loop_1_2.free_symbols == {'i_0', 'N'}
    assert loop_2_1.free_symbols == {'i_1', 'N'}
    assert loop_2_2.free_symbols == {'i_1', 'N'}


def test_branch_subscope_nofission():
    sdfg = dace.SDFG('branch_subscope_nofission')
    sdfg.add_symbol('i', dace.int32)
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
    pipeline = FixedPointPipeline([StrictSymbolSSA()])
    pipeline.apply_pass(sdfg, {})

    assert set(sdfg.symbols.keys()) == {'i'}


def test_branch_subscope_fission():
    sdfg = dace.SDFG('branch_subscope_fission')
    sdfg.add_symbol('i', dace.int32)
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
    after_assign = dace.InterstateEdge(assignments={'i': 'i + 1'})
    sdfg.add_edge(merge_1, guard_after, after_assign)
    after_cond_left = dace.InterstateEdge(condition='i <= 0')
    after_cond_right = dace.InterstateEdge(condition='i > 0')
    sdfg.add_edge(guard_after, left_after, after_cond_left)
    sdfg.add_edge(guard_after, right_after, after_cond_right)
    sdfg.add_edge(left_after, merge_after, dace.InterstateEdge())
    sdfg.add_edge(right_after, merge_after, dace.InterstateEdge())

    # Test the pass.
    pipeline = FixedPointPipeline([StrictSymbolSSA()])
    pipeline.apply_pass(sdfg, {})

    assert set(sdfg.symbols.keys()) == {'i', 'i_0', 'i_1'}


if __name__ == '__main__':
    test_loop_iter_symbol_reused_split()
    test_loop_iter_symbol_reused_fused()
    test_branch_subscope_nofission()
    test_branch_subscope_fission()


def unrolled_index_chain_sdfg() -> dace.SDFG:
    """The shape short-loop unrolling leaves behind: one index symbol reassigned per replay.

    ``s0 -[idx=arr[0]]-> s1 -[idx=arr[1]]-> s2 -[idx=arr[2]]-> s3``, each state reading ``idx``.
    """
    sdfg = dace.SDFG('unrolled_index_chain')
    sdfg.add_array('arr', [3], dace.int64)
    sdfg.add_array('out', [3], dace.float64)
    sdfg.add_symbol('idx', dace.int64)

    states = [sdfg.add_state(f's{i}', is_start_block=(i == 0)) for i in range(4)]
    for i in range(3):
        sdfg.add_edge(states[i], states[i + 1], dace.InterstateEdge(assignments={'idx': f'arr[{i}]'}))
    # Every state after the first reads the symbol, so each replay has a consumer.
    for i in range(1, 4):
        tasklet = states[i].add_tasklet(f't{i}', {}, {'o'}, 'o = 1.0')
        access = states[i].add_access('out')
        states[i].add_edge(tasklet, 'o', access, None, dace.Memlet(data='out', subset='idx'))
    return sdfg


def test_symbol_ssa_versions_an_unrolled_index_chain():
    """Each replay's definition gets its own symbol, and each consumer reads its own version."""
    sdfg = unrolled_index_chain_sdfg()
    result = SymbolSSA().apply_pass(sdfg, {})

    assert result is not None and 'idx' in result
    # Three definitions; the last is live at the region exit and must keep the original name.
    assert len(result['idx']) == 2, result
    defined = [set(e.data.assignments.keys()) for e in sdfg.edges()]
    assert all(len(names) == 1 for names in defined), defined
    assert len({next(iter(names)) for names in defined}) == 3, defined

    # Each state reads exactly the version the edge above it defines.
    for edge in sdfg.edges():
        version = next(iter(edge.data.assignments))
        reads = {str(s) for s in edge.dst.free_symbols}
        assert version in reads, (version, reads)
    sdfg.validate()


def test_symbol_ssa_leaves_an_ambiguous_definition_alone():
    """A symbol whose versions merge at a join would need a phi, so nothing is renamed."""
    sdfg = dace.SDFG('ambiguous_join')
    sdfg.add_array('out', [4], dace.float64)
    sdfg.add_symbol('idx', dace.int64)
    entry = sdfg.add_state('entry', is_start_block=True)
    left, right, join = sdfg.add_state('left'), sdfg.add_state('right'), sdfg.add_state('join')
    sdfg.add_edge(entry, left, dace.InterstateEdge(condition='idx < 1', assignments={'idx': '1'}))
    sdfg.add_edge(entry, right, dace.InterstateEdge(condition='idx >= 1', assignments={'idx': '2'}))
    # Both definitions reach the join, so neither may be renamed to it.
    sdfg.add_edge(left, join, dace.InterstateEdge())
    sdfg.add_edge(right, join, dace.InterstateEdge())
    tasklet = join.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    join.add_edge(tasklet, 'o', join.add_access('out'), None, dace.Memlet(data='out', subset='idx'))

    assert SymbolSSA().apply_pass(sdfg, {}) is None
    assert all(set(e.data.assignments) <= {'idx'} for e in sdfg.edges())


def test_symbol_ssa_leaves_a_loop_carried_increment_alone():
    """``k = k + 1`` around a back edge must keep its name: it is the induction variable.

    The body is reached both by the initializing edge and by the back edge, so the read inside it
    has two reaching definitions -- versioning either one would break the recurrence, and would
    also destroy the self-referential shape ``InductionVariableSubstitution`` matches on.
    """
    sdfg = dace.SDFG('loop_carried_increment')
    sdfg.add_array('out', [16], dace.float64)
    sdfg.add_symbol('k', dace.int64)
    init, body, done = sdfg.add_state('init', is_start_block=True), sdfg.add_state('body'), sdfg.add_state('done')
    sdfg.add_edge(init, body, dace.InterstateEdge(assignments={'k': '0'}))
    tasklet = body.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    body.add_edge(tasklet, 'o', body.add_access('out'), None, dace.Memlet(data='out', subset='k'))
    sdfg.add_edge(body, body, dace.InterstateEdge(condition='k < 15', assignments={'k': 'k + 1'}))
    sdfg.add_edge(body, done, dace.InterstateEdge(condition='k >= 15'))

    assert SymbolSSA().apply_pass(sdfg, {}) is None
    assert {k for e in sdfg.edges() for k in e.data.assignments} == {'k'}


def test_symbol_ssa_versions_an_unrolled_accumulator_chain():
    """The same increment, once unrolling has laid it out straight, does version.

    Each definition then has exactly one reaching predecessor, so the chain becomes
    ``k_1 = k + 1; k_2 = k_1 + 1; ...`` -- with the exit-live definition keeping the original name
    so whatever reads ``k`` after the region still finds it.
    """
    sdfg = dace.SDFG('unrolled_accumulator')
    sdfg.add_array('out', [8], dace.float64)
    sdfg.add_symbol('k', dace.int64)
    states = [sdfg.add_state(f's{i}', is_start_block=(i == 0)) for i in range(4)]
    for i in range(3):
        sdfg.add_edge(states[i], states[i + 1], dace.InterstateEdge(assignments={'k': 'k + 1'}))
    for state in states[1:]:
        tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
        state.add_edge(tasklet, 'o', state.add_access('out'), None, dace.Memlet(data='out', subset='k'))

    result = SymbolSSA().apply_pass(sdfg, {})
    assert result is not None and len(result['k']) == 2, result
    # Every definition is distinct, and the last one still writes the escaping name.
    defined = [next(iter(e.data.assignments)) for e in sdfg.edges()]
    assert len(set(defined)) == 3, defined
    assert 'k' in defined, defined
    # Each increment reads exactly the version defined immediately above it, never a later one.
    # Compare symbols, not substrings: 'k' occurs inside 'k_0'.
    chain = [next(iter(e.data.assignments)) for e in sdfg.edges()]
    for position, edge in enumerate(sdfg.edges()):
        target = next(iter(edge.data.assignments))
        rhs = edge.data.assignments[target]
        read = {str(sym) for sym in dace.symbolic.pystr_to_symbolic(rhs).free_symbols}
        expected = 'k' if position == 0 else chain[position - 1]
        assert read == {expected}, (target, rhs, expected)
    sdfg.validate()
