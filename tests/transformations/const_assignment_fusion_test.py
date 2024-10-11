import os
from copy import deepcopy

import numpy as np

import dace
from dace.transformation.dataflow.const_assignment_fusion import ConstAssignmentMapFusion, ConstAssignmentStateFusion
from dace.transformation.interstate import StateFusionExtended

K = dace.symbol('K')
M = dace.symbol('M')
N = dace.symbol('N')


@dace.program
def assign_top_row(A: dace.float32[M, N]):
    for t in dace.map[0:N]:
        A[0, t] = 1


@dace.program
def assign_top_row_branched(A: dace.float32[M, N]):
    for t, in dace.map[0:N]:
        if t % 2 == 0:
            A[0, t] = 1


@dace.program
def assign_bottom_row(A: dace.float32[M, N]):
    for b in dace.map[0:N]:
        A[M - 1, b] = 1


@dace.program
def assign_left_col(A: dace.float32[M, N]):
    for l in dace.map[0:M]:
        A[l, 0] = 1


@dace.program
def assign_right_col(A: dace.float32[M, N]):
    for r in dace.map[0:M]:
        A[r, N - 1] = 1


def assign_bounary_sdfg():
    st0 = assign_top_row.to_sdfg(simplify=True, validate=True, use_cache=False)
    st0.start_block.label = 'st0'

    st1 = assign_bottom_row.to_sdfg(simplify=True, validate=True, use_cache=False)
    st1.start_block.label = 'st1'
    st0.add_edge(st0.start_state, st1.start_state, dace.InterstateEdge())

    st2 = assign_left_col.to_sdfg(simplify=True, validate=True, use_cache=False)
    st2.start_block.label = 'st2'
    st0.add_edge(st1.start_state, st2.start_state, dace.InterstateEdge())

    st3 = assign_right_col.to_sdfg(simplify=True, validate=True, use_cache=False)
    st3.start_block.label = 'st3'
    st0.add_edge(st2.start_state, st3.start_state, dace.InterstateEdge())

    return st0


def test_within_state_fusion():
    A = np.random.uniform(size=(4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_bounary_sdfg()
    g.save(os.path.join('_dacegraphs', 'simple-0.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    g(A=actual_A, M=4, N=5)

    # Fuse the two states so that the const-assignment-fusion is applicable.
    g.apply_transformations_repeated(StateFusionExtended, validate_all=True)
    g.save(os.path.join('_dacegraphs', 'simple-1.sdfg'))
    g.validate()

    g.apply_transformations(ConstAssignmentMapFusion)
    g.save(os.path.join('_dacegraphs', 'simple-2.sdfg'))
    g.validate()

    g.apply_transformations(ConstAssignmentMapFusion)
    g.save(os.path.join('_dacegraphs', 'simple-3.sdfg'))
    g.validate()

    g.apply_transformations(ConstAssignmentMapFusion)
    g.save(os.path.join('_dacegraphs', 'simple-4.sdfg'))
    g.validate()
    our_A = deepcopy(A)
    g(A=our_A, M=4, N=5)

    # print(our_A)
    assert np.allclose(our_A, actual_A)


def test_interstate_fusion():
    A = np.random.uniform(size=(4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_bounary_sdfg()
    g.save(os.path.join('_dacegraphs', 'interstate-0.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    g(A=actual_A, M=4, N=5)

    g.apply_transformations(ConstAssignmentStateFusion)
    g.save(os.path.join('_dacegraphs', 'interstate-1.sdfg'))
    g.validate()

    g.apply_transformations(ConstAssignmentStateFusion)
    g.save(os.path.join('_dacegraphs', 'interstate-2.sdfg'))
    g.validate()

    g.apply_transformations(ConstAssignmentStateFusion)
    g.save(os.path.join('_dacegraphs', 'interstate-3.sdfg'))
    g.validate()
    our_A = deepcopy(A)
    g(A=our_A, M=4, N=5)

    # print(our_A)
    assert np.allclose(our_A, actual_A)


@dace.program
def assign_bounary_free_floating(A: dace.float32[M, N], B: dace.float32[M, N]):
    assign_top_row(A)
    assign_bottom_row(B)


def test_free_floating_fusion():
    A = np.random.uniform(size=(4, 5)).astype(np.float32)
    B = np.random.uniform(size=(4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_bounary_free_floating.to_sdfg(simplify=True, validate=True, use_cache=False)
    g.save(os.path.join('_dacegraphs', 'floating-0.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    actual_B = deepcopy(B)
    g(A=actual_A, B=actual_B, M=4, N=5)

    g.apply_transformations(ConstAssignmentMapFusion)
    g.save(os.path.join('_dacegraphs', 'floating-1.sdfg'))
    g.validate()
    our_A = deepcopy(A)
    our_B = deepcopy(B)
    g(A=our_A, B=our_B, M=4, N=5)

    # print(our_A)
    assert np.allclose(our_A, actual_A)


@dace.program
def assign_top_face(A: dace.float32[K, M, N]):
    for t1, t2 in dace.map[0:M, 0:N]:
        A[0, t1, t2] = 1


@dace.program
def assign_bottom_face(A: dace.float32[K, M, N]):
    for t1, t2 in dace.map[0:M, 0:N]:
        A[K - 1, t1, t2] = 1


@dace.program
def assign_bounary_3d(A: dace.float32[K, M, N], B: dace.float32[K, M, N]):
    assign_top_face(A)
    assign_bottom_face(B)


def test_fusion_with_multiple_indices():
    A = np.random.uniform(size=(3, 4, 5)).astype(np.float32)
    B = np.random.uniform(size=(3, 4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_bounary_3d.to_sdfg(simplify=True, validate=True, use_cache=False)
    g.save(os.path.join('_dacegraphs', '3d-0.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    actual_B = deepcopy(B)
    g(A=actual_A, B=actual_B, K=3, M=4, N=5)

    g.apply_transformations(ConstAssignmentMapFusion)
    g.save(os.path.join('_dacegraphs', '3d-1.sdfg'))
    g.validate()
    our_A = deepcopy(A)
    our_B = deepcopy(B)
    g(A=our_A, B=our_B, K=3, M=4, N=5)

    # print(our_A)
    assert np.allclose(our_A, actual_A)


@dace.program
def assign_bounary_with_branch(A: dace.float32[M, N], B: dace.float32[M, N]):
    assign_top_row_branched(A)
    assign_bottom_row(B)


def test_fusion_with_branch():
    A = np.random.uniform(size=(4, 5)).astype(np.float32)
    B = np.random.uniform(size=(4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_bounary_with_branch.to_sdfg(simplify=True, validate=True, use_cache=False)
    g.save(os.path.join('_dacegraphs', 'branched-0.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    actual_B = deepcopy(B)
    g(A=actual_A, B=actual_B, M=4, N=5)

    g.apply_transformations(ConstAssignmentMapFusion)
    g.save(os.path.join('_dacegraphs', 'branched-1.sdfg'))
    g.validate()
    our_A = deepcopy(A)
    our_B = deepcopy(B)
    g(A=our_A, B=our_B, M=4, N=5)

    # print(our_A)
    assert np.allclose(our_A, actual_A)


if __name__ == '__main__':
    test_within_state_fusion()
    test_interstate_fusion()
    test_free_floating_fusion()
    test_fusion_with_branch()
    test_fusion_with_multiple_indices()
