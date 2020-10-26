# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.sdfg.propagation import propagate_states


def state_check_executions(state, expected, expected_dynamic=False):
    if state.executions != expected:
        raise RuntimeError(
            'Expected {} execution, got {}'.format(expected, state.executions)
        )
    elif expected_dynamic and not state.dynamic_executions:
        raise RuntimeError(
            'Expected dynamic executions, got static'
        )
    elif state.dynamic_executions and not expected_dynamic:
        raise RuntimeError(
            'Expected static executions, got dynamic'
        )


def test_conditional_full_merge():
    @dace.program(dace.int32, dace.int32, dace.int32)
    def conditional_full_merge(a, b, c):
        if a < 10:
            if b < 10:
                c = 0
            else:
                c = 1
        c += 1

    sdfg = conditional_full_merge.to_sdfg(strict=False)
    propagate_states(sdfg)

    # Check start state.
    state = sdfg.start_state
    state_check_executions(state, 1)

    # Check the first if guard, `a < 10`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 1)
    # Get edges to the true and fals branches.
    oedges = sdfg.out_edges(state)
    true_branch_edge = None
    false_branch_edge = None
    for edge in oedges:
        if edge.data.label == '(a < 10)':
            true_branch_edge = edge
        elif edge.data.label == '(not (a < 10))':
            false_branch_edge = edge
    if false_branch_edge is None or true_branch_edge is None:
        raise RuntimeError(
            'Couldn\'t identify guard edges'
        )
    # Check the true branch.
    state = true_branch_edge.dst
    state_check_executions(state, 1, expected_dynamic=True)
    # Check the next if guard, `b < 20`
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 1, expected_dynamic=True)
    # Get edges to the true and fals branches.
    oedges = sdfg.out_edges(state)
    true_branch_edge = None
    false_branch_edge = None
    for edge in oedges:
        if edge.data.label == '(b < 10)':
            true_branch_edge = edge
        elif edge.data.label == '(not (b < 10))':
            false_branch_edge = edge
    if false_branch_edge is None or true_branch_edge is None:
        raise RuntimeError(
            'Couldn\'t identify guard edges'
        )
    # Check the true branch.
    state = true_branch_edge.dst
    state_check_executions(state, 1, expected_dynamic=True)
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 1, expected_dynamic=True)
    # Check the false branch.
    state = false_branch_edge.dst
    state_check_executions(state, 1, expected_dynamic=True)
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 1, expected_dynamic=True)

    # Check the first branch merge state.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 1, expected_dynamic=True)

    # Check the second branch merge state.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 1)

    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 1)


def test_while_inside_for():
    @dace.program(dace.int32)
    def while_inside_for(a):
        for i in range(20):
            j = 0
            while j < 20:
                a += 5

    sdfg = while_inside_for.to_sdfg(strict=False)
    propagate_states(sdfg)

    # Check start state.
    state = sdfg.start_state
    state_check_executions(state, 1)

    # Check the for loop guard, `i in range(20)`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 21)
    # Get edges to inside and outside the loop.
    oedges = sdfg.out_edges(state)
    end_branch_edge = None
    for_branch_edge = None
    for edge in oedges:
        if edge.data.label == '(i < 20)':
            for_branch_edge = edge
        elif edge.data.label == '(not (i < 20))':
            end_branch_edge = edge
    if end_branch_edge is None or for_branch_edge is None:
        raise RuntimeError(
            'Couldn\'t identify guard edges'
        )
    # Check loop-end branch.
    state = end_branch_edge.dst
    state_check_executions(state, 1)
    # Check inside the loop.
    state = for_branch_edge.dst
    state_check_executions(state, 20)
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 20)

    # Check the while guard, `j < 20`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 0, expected_dynamic=True)
    # Get edges to inside and outside the loop.
    oedges = sdfg.out_edges(state)
    end_branch_edge = None
    for_branch_edge = None
    for edge in oedges:
        if edge.data.label == '(j < 20)':
            for_branch_edge = edge
        elif edge.data.label == '(not (j < 20))':
            end_branch_edge = edge
    if end_branch_edge is None or for_branch_edge is None:
        raise RuntimeError(
            'Couldn\'t identify guard edges'
        )
    # Check loop-end branch.
    state = end_branch_edge.dst
    state_check_executions(state, 20)
    # Check inside the loop.
    state = for_branch_edge.dst
    state_check_executions(state, 0, expected_dynamic=True)
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 0, expected_dynamic=True)


def test_for_with_nested_full_merge_branch():
    @dace.program(dace.int32)
    def for_with_nested_full_merge_branch(a):
        for i in range(20):
            if i < 10:
                a += 2
            else:
                a += 1

    sdfg = for_with_nested_full_merge_branch.to_sdfg(strict=False)
    propagate_states(sdfg)

    # Check start state.
    state = sdfg.start_state
    state_check_executions(state, 1)

    # For loop, check loop guard, `for i in range(20)`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 21)
    # Get edges to inside and outside the loop.
    oedges = sdfg.out_edges(state)
    end_branch_edge = None
    for_branch_edge = None
    for edge in oedges:
        if edge.data.label == '(i < 20)':
            for_branch_edge = edge
        elif edge.data.label == '(not (i < 20))':
            end_branch_edge = edge
    if end_branch_edge is None or for_branch_edge is None:
        raise RuntimeError(
            'Couldn\'t identify guard edges'
        )
    # Check loop-end branch.
    state = end_branch_edge.dst
    state_check_executions(state, 1)
    # Check inside the loop.
    state = for_branch_edge.dst
    state_check_executions(state, 20)

    # Check the branch guard, `if i < 10`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 20)
    # Get edges to both sides of the conditional split.
    oedges = sdfg.out_edges(state)
    condition_met_edge = None
    condition_broken_edge = None
    for edge in oedges:
        if edge.data.label == '(i < 10)':
            condition_met_edge = edge
        elif edge.data.label == '(not (i < 10))':
            condition_broken_edge = edge
    if condition_met_edge is None or condition_broken_edge is None:
        raise RuntimeError(
            'Couldn\'t identify conditional guard edges'
        )
    # Check the 'true' branch.
    state = condition_met_edge.dst
    state_check_executions(state, 20, expected_dynamic=True)
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 20, expected_dynamic=True)
    # Check the 'false' branch.
    state = condition_broken_edge.dst
    state_check_executions(state, 20, expected_dynamic=True)
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 20, expected_dynamic=True)

    # Check where the branches meet again.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 20)


def test_while_with_nested_full_merge_branch():
    @dace.program(dace.int32)
    def while_with_nested_full_merge_branch(a):
        while a < 20:
            if a < 10:
                a += 2
            else:
                a += 1

    sdfg = while_with_nested_full_merge_branch.to_sdfg(strict=False)
    propagate_states(sdfg)

    # Check start state.
    state = sdfg.start_state
    state_check_executions(state, 1)

    # While loop, check loop guard, `while a < N`. Must be dynamic unbounded.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 0, expected_dynamic=True)
    # Get edges to inside and outside the loop.
    oedges = sdfg.out_edges(state)
    end_branch_edge = None
    for_branch_edge = None
    for edge in oedges:
        if edge.data.label == '(a < 20)':
            for_branch_edge = edge
        elif edge.data.label == '(not (a < 20))':
            end_branch_edge = edge
    if end_branch_edge is None or for_branch_edge is None:
        raise RuntimeError(
            'Couldn\'t identify guard edges'
        )
    # Check loop-end branch.
    state = end_branch_edge.dst
    state_check_executions(state, 1)
    # Check inside the loop.
    state = for_branch_edge.dst
    state_check_executions(state, 0, expected_dynamic=True)

    # Check the branch guard, `if a < 10`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 0, expected_dynamic=True)
    # Get edges to both sides of the conditional split.
    oedges = sdfg.out_edges(state)
    condition_met_edge = None
    condition_broken_edge = None
    for edge in oedges:
        if edge.data.label == '(a < 10)':
            condition_met_edge = edge
        elif edge.data.label == '(not (a < 10))':
            condition_broken_edge = edge
    if condition_met_edge is None or condition_broken_edge is None:
        raise RuntimeError(
            'Couldn\'t identify conditional guard edges'
        )
    # Check the 'true' branch.
    state = condition_met_edge.dst
    state_check_executions(state, 0, expected_dynamic=True)
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 0, expected_dynamic=True)
    # Check the 'false' branch.
    state = condition_broken_edge.dst
    state_check_executions(state, 0, expected_dynamic=True)
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 0, expected_dynamic=True)

    # Check where the branches meet again.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 0, expected_dynamic=True)


def test_3_fold_nested_loop_with_symbolic_bounds():
    N = dace.symbol('N')
    M = dace.symbol('M')
    K = dace.symbol('K')

    @dace.program(dace.int32)
    def nested_3_symbolic(a):
        for i in range(N):
            for j in range(M):
                for k in range(K):
                    a += 5

    sdfg = nested_3_symbolic.to_sdfg(strict=False)
    propagate_states(sdfg)

    # Check start state.
    state = sdfg.start_state
    state_check_executions(state, 1)

    # 1st level loop, check loop guard, `for i in range(20)`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, N + 1)
    # Get edges to inside and outside the loop.
    oedges = sdfg.out_edges(state)
    end_branch_edge = None
    for_branch_edge = None
    for edge in oedges:
        if edge.data.label == '(i < N)':
            for_branch_edge = edge
        elif edge.data.label == '(not (i < N))':
            end_branch_edge = edge
    if end_branch_edge is None or for_branch_edge is None:
        raise RuntimeError(
            'Couldn\'t identify guard edges'
        )
    # Check loop-end branch.
    state = end_branch_edge.dst
    state_check_executions(state, 1)
    # Check inside the loop.
    state = for_branch_edge.dst
    state_check_executions(state, N)

    # 2nd level nested loop, check loog guard, `for j in range(i, 20)`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, M*N + N)
    # Get edges to inside and outside the loop.
    oedges = sdfg.out_edges(state)
    end_branch_edge = None
    for_branch_edge = None
    for edge in oedges:
        if edge.data.label == '(j < M)':
            for_branch_edge = edge
        elif edge.data.label == '(not (j < M))':
            end_branch_edge = edge
    if end_branch_edge is None or for_branch_edge is None:
        raise RuntimeError(
            'Couldn\'t identify guard edges'
        )
    # Check loop-end branch.
    state = end_branch_edge.dst
    state_check_executions(state, N)
    # Check inside the loop.
    state = for_branch_edge.dst
    state_check_executions(state, M*N)

    # 3rd level nested loop, check loog guard, `for k in range(i, j)`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, M*N*K + M*N)
    # Get edges to inside and outside the loop.
    oedges = sdfg.out_edges(state)
    end_branch_edge = None
    for_branch_edge = None
    for edge in oedges:
        if edge.data.label == '(k < K)':
            for_branch_edge = edge
        elif edge.data.label == '(not (k < K))':
            end_branch_edge = edge
    if end_branch_edge is None or for_branch_edge is None:
        raise RuntimeError(
            'Couldn\'t identify guard edges'
        )
    # Check loop-end branch.
    state = end_branch_edge.dst
    state_check_executions(state, M*N)
    # Check inside the loop.
    state = for_branch_edge.dst
    state_check_executions(state, M*N*K)
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, M*N*K)


def test_3_fold_nested_loop():
    @dace.program(dace.int32[20, 20])
    def nested_3(A):
        for i in range(20):
            for j in range(i, 20):
                for k in range(i, j):
                    A[k, j] += 5

    sdfg = nested_3.to_sdfg(strict=False)
    propagate_states(sdfg)

    # Check start state.
    state = sdfg.start_state
    state_check_executions(state, 1)

    # 1st level loop, check loop guard, `for i in range(20)`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 21)
    # Get edges to inside and outside the loop.
    oedges = sdfg.out_edges(state)
    end_branch_edge = None
    for_branch_edge = None
    for edge in oedges:
        if edge.data.label == '(i < 20)':
            for_branch_edge = edge
        elif edge.data.label == '(not (i < 20))':
            end_branch_edge = edge
    if end_branch_edge is None or for_branch_edge is None:
        raise RuntimeError(
            'Couldn\'t identify guard edges'
        )
    # Check loop-end branch.
    state = end_branch_edge.dst
    state_check_executions(state, 1)
    # Check inside the loop.
    state = for_branch_edge.dst
    state_check_executions(state, 20)

    # 2nd level nested loop, check loog guard, `for j in range(i, 20)`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 230)
    # Get edges to inside and outside the loop.
    oedges = sdfg.out_edges(state)
    end_branch_edge = None
    for_branch_edge = None
    for edge in oedges:
        if edge.data.label == '(j < 20)':
            for_branch_edge = edge
        elif edge.data.label == '(not (j < 20))':
            end_branch_edge = edge
    if end_branch_edge is None or for_branch_edge is None:
        raise RuntimeError(
            'Couldn\'t identify guard edges'
        )
    # Check loop-end branch.
    state = end_branch_edge.dst
    state_check_executions(state, 20)
    # Check inside the loop.
    state = for_branch_edge.dst
    state_check_executions(state, 210)

    # 3rd level nested loop, check loog guard, `for k in range(i, j)`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 1540)
    # Get edges to inside and outside the loop.
    oedges = sdfg.out_edges(state)
    end_branch_edge = None
    for_branch_edge = None
    for edge in oedges:
        if edge.data.label == '(k < j)':
            for_branch_edge = edge
        elif edge.data.label == '(not (k < j))':
            end_branch_edge = edge
    if end_branch_edge is None or for_branch_edge is None:
        raise RuntimeError(
            'Couldn\'t identify guard edges'
        )
    # Check loop-end branch.
    state = end_branch_edge.dst
    state_check_executions(state, 210)
    # Check inside the loop.
    state = for_branch_edge.dst
    state_check_executions(state, 1330)
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, 1330)


if __name__ == "__main__":
    test_3_fold_nested_loop()
    test_3_fold_nested_loop_with_symbolic_bounds()
    test_while_with_nested_full_merge_branch()
    test_for_with_nested_full_merge_branch()
    test_while_inside_for()
    test_conditional_full_merge()
