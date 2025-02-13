# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
from dace.dtypes import Language
from dace.properties import CodeProperty, CodeBlock
from dace.sdfg.sdfg import InterstateEdge
import dace
from dace.sdfg.propagation import propagate_states
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising


def state_check_executions(state, expected, expected_dynamic=False):
    if state.executions != expected:
        raise RuntimeError('Expected {} execution, got {}'.format(expected, state.executions))
    elif expected_dynamic and not state.dynamic_executions:
        raise RuntimeError('Expected dynamic executions, got static')
    elif state.dynamic_executions and not expected_dynamic:
        raise RuntimeError('Expected static executions, got dynamic')


@pytest.mark.parametrize('with_regions', [False, True])
def test_conditional_fake_merge(with_regions):
    sdfg = dace.SDFG('fake_merge')

    state_init = sdfg.add_state('init')
    state_a = sdfg.add_state('A')
    state_b = sdfg.add_state('B')
    state_c = sdfg.add_state('C')
    state_d = sdfg.add_state('D')
    state_e = sdfg.add_state('E')

    sdfg.add_edge(state_init, state_a, InterstateEdge(assignments={
        'i': '0',
        'j': '0',
    }))
    sdfg.add_edge(state_a, state_b,
                  InterstateEdge(condition=CodeProperty.from_string('i < 10', language=Language.Python)))
    sdfg.add_edge(state_a, state_c,
                  InterstateEdge(condition=CodeProperty.from_string('not (i < 10)', language=Language.Python)))
    sdfg.add_edge(state_b, state_d, InterstateEdge())
    sdfg.add_edge(state_c, state_d,
                  InterstateEdge(condition=CodeProperty.from_string('j < 10', language=Language.Python)))
    sdfg.add_edge(state_c, state_e,
                  InterstateEdge(condition=CodeProperty.from_string('not (j < 10)', language=Language.Python)))

    if with_regions:
        ControlFlowRaising().apply_pass(sdfg, {})

    propagate_states(sdfg)

    state_check_executions(state_d, 1, True)
    state_check_executions(state_e, 1, True)


@pytest.mark.parametrize('with_regions', [False, True])
def test_conditional_full_merge(with_regions):
    sdfg = dace.SDFG('conditional_full_merge')

    sdfg.add_scalar('a', dace.int32)
    sdfg.add_scalar('b', dace.int32)

    init_state = sdfg.add_state('init_state')
    if_guard_1 = sdfg.add_state('if_guard_1')
    l_branch_1 = sdfg.add_state('l_branch_1')
    if_guard_2 = sdfg.add_state('if_guard_2')
    l_branch = sdfg.add_state('l_branch')
    r_branch = sdfg.add_state('r_branch')
    if_merge_1 = sdfg.add_state('if_merge_1')
    if_merge_2 = sdfg.add_state('if_merge_2')

    sdfg.add_edge(init_state, if_guard_1, dace.InterstateEdge())
    sdfg.add_edge(if_guard_1, l_branch_1, dace.InterstateEdge(condition=CodeBlock('a < 10')))
    sdfg.add_edge(l_branch_1, if_guard_2, dace.InterstateEdge())
    sdfg.add_edge(if_guard_1, if_merge_1, dace.InterstateEdge(condition=CodeBlock('not (a < 10)')))
    sdfg.add_edge(if_guard_2, l_branch, dace.InterstateEdge(condition=CodeBlock('b < 10')))
    sdfg.add_edge(if_guard_2, r_branch, dace.InterstateEdge(condition=CodeBlock('not (b < 10)')))
    sdfg.add_edge(l_branch, if_merge_2, dace.InterstateEdge())
    sdfg.add_edge(r_branch, if_merge_2, dace.InterstateEdge())
    sdfg.add_edge(if_merge_2, if_merge_1, dace.InterstateEdge())

    if with_regions:
        ControlFlowRaising().apply_pass(sdfg, {})

    propagate_states(sdfg)

    # Check start state.
    state_check_executions(init_state, 1)

    # Check the first if guard, `a < 10`.
    state_check_executions(if_guard_1, 1)
    # Check the true branch.
    state_check_executions(l_branch_1, 1, expected_dynamic=True)
    # Check the next if guard, `b < 20`
    state_check_executions(if_guard_2, 1, expected_dynamic=True)
    # Check the true branch.
    state_check_executions(l_branch_1, 1, expected_dynamic=True)
    # Check the false branch.
    state_check_executions(r_branch, 1, expected_dynamic=True)
    # Check the first branch merge state.
    state_check_executions(if_merge_2, 1, expected_dynamic=True)
    # Check the second branch merge state.
    state_check_executions(if_merge_1, 1)


@pytest.mark.parametrize('with_regions', [False, True])
def test_while_inside_for(with_regions):
    sdfg = dace.SDFG('while_inside_for')

    sdfg.add_symbol('i', dace.int32)
    sdfg.add_symbol('j', dace.int32)

    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1 = sdfg.add_state('loop_1')
    end_1 = sdfg.add_state('end_1')
    guard_2 = sdfg.add_state('guard_2')
    loop_2 = sdfg.add_state('loop_2')
    end_2 = sdfg.add_state('end_2')

    sdfg.add_edge(init_state, guard_1, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, end_1, dace.InterstateEdge(condition=CodeBlock('not (i < 20)')))
    sdfg.add_edge(guard_1, loop_1, dace.InterstateEdge(condition=CodeBlock('i < 20')))
    sdfg.add_edge(loop_1, guard_2, dace.InterstateEdge())
    sdfg.add_edge(end_2, guard_1, dace.InterstateEdge(assignments={'i': 'i + 1'}))

    sdfg.add_edge(guard_2, end_2, dace.InterstateEdge(condition=CodeBlock('not (j < 20)')))
    sdfg.add_edge(guard_2, loop_2, dace.InterstateEdge(condition=CodeBlock('j < 20')))
    sdfg.add_edge(loop_2, guard_2, dace.InterstateEdge())

    if with_regions:
        ControlFlowRaising().apply_pass(sdfg, {})

    propagate_states(sdfg)

    # Check start state.
    state_check_executions(init_state, 1)

    # Check the for loop guard, `i in range(20)`.
    if with_regions:
        state_check_executions(guard_1, 20)
    else:
        state_check_executions(guard_1, 21)
    # Check loop-end branch.
    state_check_executions(end_1, 1)
    # Check inside the loop.
    state_check_executions(loop_1, 20)

    # Check the while guard, `j < 20`.
    state_check_executions(guard_2, 0, expected_dynamic=True)
    # Check loop-end branch.
    state_check_executions(end_2, 20)
    # Check inside the loop.
    state_check_executions(loop_2, 0, expected_dynamic=True)


@pytest.mark.parametrize('with_regions', [False, True])
def test_for_with_nested_full_merge_branch(with_regions):
    sdfg = dace.SDFG('for_full_merge')

    sdfg.add_symbol('i', dace.int32)
    sdfg.add_scalar('a', dace.int32)

    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    if_guard = sdfg.add_state('if_guard')
    l_branch = sdfg.add_state('l_branch')
    r_branch = sdfg.add_state('r_branch')
    if_merge = sdfg.add_state('if_merge')
    end_1 = sdfg.add_state('end_1')

    lra = l_branch.add_access('a')
    lt = l_branch.add_tasklet('t1', {'i1'}, {'o1'}, 'o1 = i1 + 5')
    lwa = l_branch.add_access('a')
    l_branch.add_edge(lra, None, lt, 'i1', dace.Memlet('a[0]'))
    l_branch.add_edge(lt, 'o1', lwa, None, dace.Memlet('a[0]'))

    rra = r_branch.add_access('a')
    rt = r_branch.add_tasklet('t2', {'i1'}, {'o1'}, 'o1 = i1 + 10')
    rwa = r_branch.add_access('a')
    r_branch.add_edge(rra, None, rt, 'i1', dace.Memlet('a[0]'))
    r_branch.add_edge(rt, 'o1', rwa, None, dace.Memlet('a[0]'))

    sdfg.add_edge(init_state, guard_1, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, end_1, dace.InterstateEdge(condition=CodeBlock('not (i < 20)')))
    sdfg.add_edge(guard_1, if_guard, dace.InterstateEdge(condition=CodeBlock('i < 20')))
    sdfg.add_edge(if_guard, l_branch, dace.InterstateEdge(condition=CodeBlock('not (a < 10)')))
    sdfg.add_edge(if_guard, r_branch, dace.InterstateEdge(condition=CodeBlock('a < 10')))
    sdfg.add_edge(l_branch, if_merge, dace.InterstateEdge())
    sdfg.add_edge(r_branch, if_merge, dace.InterstateEdge())
    sdfg.add_edge(if_merge, guard_1, dace.InterstateEdge(assignments={'i': 'i + 1'}))

    if with_regions:
        ControlFlowRaising().apply_pass(sdfg, {})

    propagate_states(sdfg)

    # Check start state.
    state_check_executions(init_state, 1)

    # For loop, check loop guard, `for i in range(20)`.
    if with_regions:
        state_check_executions(guard_1, 20)
    else:
        state_check_executions(guard_1, 21)
    # Check loop-end branch.
    state_check_executions(end_1, 1)
    # Check inside the loop.
    state_check_executions(if_guard, 20)
    # Check the 'true' branch.
    state_check_executions(r_branch, 20, expected_dynamic=True)
    # Check the 'false' branch.
    state_check_executions(l_branch, 20, expected_dynamic=True)
    # Check where the branches meet again.
    state_check_executions(if_merge, 20)


@pytest.mark.parametrize('with_regions', [False, True])
def test_for_inside_branch(with_regions):
    sdfg = dace.SDFG('for_in_branch')

    state_init = sdfg.add_state('init')
    branch_guard = sdfg.add_state('branch_guard')
    loop_guard = sdfg.add_state('loop_guard')
    loop_state = sdfg.add_state('loop_state')
    branch_merge = sdfg.add_state('branch_merge')

    sdfg.add_edge(state_init, branch_guard, InterstateEdge(assignments={
        'i': '0',
    }))
    sdfg.add_edge(branch_guard, branch_merge,
                  InterstateEdge(condition=CodeProperty.from_string('i < 10', language=Language.Python)))
    sdfg.add_edge(
        branch_guard, loop_guard,
        InterstateEdge(condition=CodeProperty.from_string('not (i < 10)', language=Language.Python),
                       assignments={
                           'j': '0',
                       }))
    sdfg.add_edge(loop_guard, loop_state,
                  InterstateEdge(condition=CodeProperty.from_string('j < 10', language=Language.Python)))
    sdfg.add_edge(loop_guard, branch_merge,
                  InterstateEdge(condition=CodeProperty.from_string('not (j < 10)', language=Language.Python)))
    sdfg.add_edge(loop_state, loop_guard, InterstateEdge(assignments={
        'j': 'j + 1',
    }))

    if with_regions:
        ControlFlowRaising().apply_pass(sdfg, {})

    propagate_states(sdfg)

    state_check_executions(branch_guard, 1, False)
    if with_regions:
        state_check_executions(loop_guard, 10, True)
    else:
        state_check_executions(loop_guard, 11, True)
    state_check_executions(loop_state, 10, True)
    state_check_executions(branch_merge, 1, False)


@pytest.mark.parametrize('with_regions', [False, True])
def test_full_merge_inside_loop(with_regions):
    sdfg = dace.SDFG('full_merge_inside_loop')

    state_init = sdfg.add_state('init')
    intermittent = sdfg.add_state('intermittent')
    loop_guard = sdfg.add_state('loop_guard')
    branch_guard = sdfg.add_state('branch_guard')
    branch_state = sdfg.add_state('branch_state')
    branch_merge = sdfg.add_state('branch_merge')
    loop_end = sdfg.add_state('loop_end')

    sdfg.add_edge(state_init, intermittent, InterstateEdge(assignments={
        'j': '0',
    }))
    sdfg.add_edge(intermittent, loop_guard, InterstateEdge(assignments={
        'i': '0',
    }))
    sdfg.add_edge(loop_guard, branch_guard,
                  InterstateEdge(condition=CodeProperty.from_string('i < 10', language=Language.Python)))
    sdfg.add_edge(loop_guard, loop_end,
                  InterstateEdge(condition=CodeProperty.from_string('not (i < 10)', language=Language.Python)))
    sdfg.add_edge(branch_guard, branch_state,
                  InterstateEdge(condition=CodeProperty.from_string('j < 10', language=Language.Python)))
    sdfg.add_edge(branch_guard, branch_merge,
                  InterstateEdge(condition=CodeProperty.from_string('not (j < 10)', language=Language.Python)))
    sdfg.add_edge(branch_state, branch_merge, InterstateEdge())
    sdfg.add_edge(branch_merge, loop_guard, InterstateEdge(assignments={
        'i': 'i + 1',
    }))

    if with_regions:
        ControlFlowRaising().apply_pass(sdfg, {})

    propagate_states(sdfg)

    if with_regions:
        state_check_executions(loop_guard, 10, False)
    else:
        state_check_executions(loop_guard, 11, False)
    state_check_executions(branch_guard, 10, False)
    state_check_executions(branch_state, 10, True)
    state_check_executions(branch_merge, 10, False)
    state_check_executions(loop_end, 1, False)


@pytest.mark.parametrize('with_regions', [False, True])
def test_while_with_nested_full_merge_branch(with_regions):
    sdfg = dace.SDFG('while_full_merge')

    sdfg.add_scalar('a', dace.int32)

    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    if_guard = sdfg.add_state('if_guard')
    l_branch = sdfg.add_state('l_branch')
    r_branch = sdfg.add_state('r_branch')
    if_merge = sdfg.add_state('if_merge')
    end_1 = sdfg.add_state('end_1')

    lra = l_branch.add_access('a')
    lt = l_branch.add_tasklet('t1', {'i1'}, {'o1'}, 'o1 = i1 + 5')
    lwa = l_branch.add_access('a')
    l_branch.add_edge(lra, None, lt, 'i1', dace.Memlet('a[0]'))
    l_branch.add_edge(lt, 'o1', lwa, None, dace.Memlet('a[0]'))

    rra = r_branch.add_access('a')
    rt = r_branch.add_tasklet('t2', {'i1'}, {'o1'}, 'o1 = i1 + 10')
    rwa = r_branch.add_access('a')
    r_branch.add_edge(rra, None, rt, 'i1', dace.Memlet('a[0]'))
    r_branch.add_edge(rt, 'o1', rwa, None, dace.Memlet('a[0]'))

    sdfg.add_edge(init_state, guard_1, dace.InterstateEdge())
    sdfg.add_edge(guard_1, end_1, dace.InterstateEdge(condition=CodeBlock('not (a < 20)')))
    sdfg.add_edge(guard_1, if_guard, dace.InterstateEdge(condition=CodeBlock('a < 20')))
    sdfg.add_edge(if_guard, l_branch, dace.InterstateEdge(condition=CodeBlock('not (a < 10)')))
    sdfg.add_edge(if_guard, r_branch, dace.InterstateEdge(condition=CodeBlock('a < 10')))
    sdfg.add_edge(l_branch, if_merge, dace.InterstateEdge())
    sdfg.add_edge(r_branch, if_merge, dace.InterstateEdge())
    sdfg.add_edge(if_merge, guard_1, dace.InterstateEdge())

    if with_regions:
        ControlFlowRaising().apply_pass(sdfg, {})

    propagate_states(sdfg)

    # Check start state.
    state_check_executions(init_state, 1)

    # While loop, check loop guard, `while a < N`. Must be dynamic unbounded.
    state_check_executions(guard_1, 0, expected_dynamic=True)
    # Check loop-end branch.
    state_check_executions(end_1, 1)
    # Check inside the loop.
    state_check_executions(if_guard, 0, expected_dynamic=True)
    # Check the 'true' branch.
    state_check_executions(r_branch, 0, expected_dynamic=True)
    # Check the 'false' branch.
    state_check_executions(l_branch, 0, expected_dynamic=True)
    # Check where the branches meet again.
    state_check_executions(if_merge, 0, expected_dynamic=True)


@pytest.mark.parametrize('with_regions', [False, True])
def test_3_fold_nested_loop_with_symbolic_bounds(with_regions):
    N = dace.symbol('N')
    M = dace.symbol('M')
    K = dace.symbol('K')

    sdfg = dace.SDFG('nest_3_symbolic')

    sdfg.add_symbol('i', dace.int32)
    sdfg.add_symbol('j', dace.int32)
    sdfg.add_symbol('k', dace.int32)

    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1 = sdfg.add_state('loop_1')
    end_1 = sdfg.add_state('end_1')
    guard_2 = sdfg.add_state('guard_2')
    loop_2 = sdfg.add_state('loop_2')
    end_2 = sdfg.add_state('end_2')
    guard_3 = sdfg.add_state('guard_3')
    end_3 = sdfg.add_state('end_3')
    loop_3 = sdfg.add_state('loop_3')

    sdfg.add_edge(init_state, guard_1, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, end_1, dace.InterstateEdge(condition=CodeBlock('not (i < N)')))
    sdfg.add_edge(guard_1, loop_1, dace.InterstateEdge(condition=CodeBlock('i < N')))
    sdfg.add_edge(loop_1, guard_2, dace.InterstateEdge(assignments={'j': 0}))
    sdfg.add_edge(end_2, guard_1, dace.InterstateEdge(assignments={'i': 'i + 1'}))

    sdfg.add_edge(guard_2, end_2, dace.InterstateEdge(condition=CodeBlock('not (j < M)')))
    sdfg.add_edge(guard_2, loop_2, dace.InterstateEdge(condition=CodeBlock('j < M')))
    sdfg.add_edge(loop_2, guard_3, dace.InterstateEdge(assignments={'k': 0}))
    sdfg.add_edge(end_3, guard_2, dace.InterstateEdge(assignments={'j': 'j + 1'}))

    sdfg.add_edge(guard_3, end_3, dace.InterstateEdge(condition=CodeBlock('not (k < K)')))
    sdfg.add_edge(guard_3, loop_3, dace.InterstateEdge(condition=CodeBlock('k < K')))
    sdfg.add_edge(loop_3, guard_3, dace.InterstateEdge(assignments={'k': 'k + 1'}))

    if with_regions:
        ControlFlowRaising().apply_pass(sdfg, {})

    propagate_states(sdfg)

    # Check start state.
    state_check_executions(init_state, 1)

    # 1st level loop, check loop guard, `for i in range(N)`.
    if with_regions:
        state_check_executions(guard_1, N)
    else:
        state_check_executions(guard_1, N + 1)
    # Check loop-end branch.
    state_check_executions(end_1, 1)
    # Check inside the loop.
    state_check_executions(loop_1, N)

    # 2nd level nested loop, check loog guard, `for j in range(M)`.
    if with_regions:
        state_check_executions(guard_2, M * N)
    else:
        state_check_executions(guard_2, M * N + N)
    # Check loop-end branch.
    state_check_executions(end_2, N)
    # Check inside the loop.
    state_check_executions(loop_2, M * N)

    # 3rd level nested loop, check loop guard, `for k in range(K)`.
    if with_regions:
        state_check_executions(guard_3, M * N * K)
    else:
        state_check_executions(guard_3, M * N * K + M * N)
    # Check loop-end branch.
    state_check_executions(end_3, M * N)
    # Check inside the loop.
    state_check_executions(loop_3, M * N * K)


@pytest.mark.parametrize('with_regions', [False, True])
def test_3_fold_nested_loop(with_regions):
    sdfg = dace.SDFG('nest_3')

    sdfg.add_symbol('i', dace.int32)
    sdfg.add_symbol('j', dace.int32)
    sdfg.add_symbol('k', dace.int32)

    init_state = sdfg.add_state('init')
    guard_1 = sdfg.add_state('guard_1')
    loop_1 = sdfg.add_state('loop_1')
    end_1 = sdfg.add_state('end_1')
    guard_2 = sdfg.add_state('guard_2')
    loop_2 = sdfg.add_state('loop_2')
    end_2 = sdfg.add_state('end_2')
    guard_3 = sdfg.add_state('guard_3')
    end_3 = sdfg.add_state('end_3')
    loop_3 = sdfg.add_state('loop_3')

    sdfg.add_edge(init_state, guard_1, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard_1, end_1, dace.InterstateEdge(condition=CodeBlock('not (i < 20)')))
    sdfg.add_edge(guard_1, loop_1, dace.InterstateEdge(condition=CodeBlock('i < 20')))
    sdfg.add_edge(loop_1, guard_2, dace.InterstateEdge(assignments={'j': 'i'}))
    sdfg.add_edge(end_2, guard_1, dace.InterstateEdge(assignments={'i': 'i + 1'}))

    sdfg.add_edge(guard_2, end_2, dace.InterstateEdge(condition=CodeBlock('not (j < 20)')))
    sdfg.add_edge(guard_2, loop_2, dace.InterstateEdge(condition=CodeBlock('j < 20')))
    sdfg.add_edge(loop_2, guard_3, dace.InterstateEdge(assignments={'k': 'i'}))
    sdfg.add_edge(end_3, guard_2, dace.InterstateEdge(assignments={'j': 'j + 1'}))

    sdfg.add_edge(guard_3, end_3, dace.InterstateEdge(condition=CodeBlock('not (k < j)')))
    sdfg.add_edge(guard_3, loop_3, dace.InterstateEdge(condition=CodeBlock('k < j')))
    sdfg.add_edge(loop_3, guard_3, dace.InterstateEdge(assignments={'k': 'k + 1'}))

    if with_regions:
        ControlFlowRaising().apply_pass(sdfg, {})

    propagate_states(sdfg)

    # Check start state.
    state_check_executions(init_state, 1)

    # 1st level loop, check loop guard, `for i in range(20)`.
    if with_regions:
        state_check_executions(guard_1, 20)
    else:
        # When using a state-machine-style loop, the guard is executed N+1 times for N loop iterations.
        state_check_executions(guard_1, 21)
    # Check loop-end branch.
    state_check_executions(end_1, 1)
    # Check inside the loop.
    state_check_executions(loop_1, 20)

    # 2nd level nested loop, check loog guard, `for j in range(i, 20)`.
    if with_regions:
        state_check_executions(guard_2, 210)
    else:
        state_check_executions(guard_2, 230)
    # Check loop-end branch.
    state_check_executions(end_2, 20)
    # Check inside the loop.
    state_check_executions(loop_2, 210)

    # 3rd level nested loop, check loop guard, `for k in range(i, j)`.
    if with_regions:
        state_check_executions(guard_3, 1330)
    else:
        state_check_executions(guard_3, 1540)
    # Check loop-end branch.
    state_check_executions(end_3, 210)
    # Check inside the loop.
    state_check_executions(loop_3, 1330)


if __name__ == "__main__":
    test_3_fold_nested_loop(False)
    test_3_fold_nested_loop_with_symbolic_bounds(False)
    test_while_with_nested_full_merge_branch(False)
    test_for_with_nested_full_merge_branch(False)
    test_for_inside_branch(False)
    test_while_inside_for(False)
    test_conditional_full_merge(False)
    test_conditional_fake_merge(False)
    test_full_merge_inside_loop(False)
    test_3_fold_nested_loop(True)
    test_3_fold_nested_loop_with_symbolic_bounds(True)
    test_while_with_nested_full_merge_branch(True)
    test_for_with_nested_full_merge_branch(True)
    test_for_inside_branch(True)
    test_while_inside_for(True)
    test_conditional_full_merge(True)
    test_conditional_fake_merge(True)
    test_full_merge_inside_loop(True)
