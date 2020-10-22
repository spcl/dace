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
    state_check_executions(state, N + 1, True)
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
    state_check_executions(state, N, True)

    # 2nd level nested loop, check loog guard, `for j in range(i, 20)`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, M*N + N, True)
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
    state_check_executions(state, N, True)
    # Check inside the loop.
    state = for_branch_edge.dst
    state_check_executions(state, M*N, True)

    # 3rd level nested loop, check loog guard, `for k in range(i, j)`.
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, M*N*K + M*N, True)
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
    state_check_executions(state, M*N, True)
    # Check inside the loop.
    state = for_branch_edge.dst
    state_check_executions(state, M*N*K, True)
    state = sdfg.out_edges(state)[0].dst
    state_check_executions(state, M*N*K, True)


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
