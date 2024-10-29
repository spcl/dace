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

    assert g.apply_transformations(ConstAssignmentMapFusion,
                                   options={'use_grid_strided_loops': True}) == 1
    g.save(os.path.join('_dacegraphs', 'simple-2.sdfg'))
    g.validate()

    assert g.apply_transformations(ConstAssignmentMapFusion,
                                   options={'use_grid_strided_loops': True}) == 1
    g.save(os.path.join('_dacegraphs', 'simple-3.sdfg'))
    g.validate()

    assert g.apply_transformations(ConstAssignmentMapFusion) == 0
    assert g.apply_transformations(ConstAssignmentMapFusion,
                                   options={'use_grid_strided_loops': True}) == 1
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

    assert g.apply_transformations(ConstAssignmentStateFusion,
                                   options={'use_grid_strided_loops': True}) == 1
    g.save(os.path.join('_dacegraphs', 'interstate-1.sdfg'))
    g.validate()

    assert g.apply_transformations(ConstAssignmentStateFusion,
                                   options={'use_grid_strided_loops': True}) == 1
    g.save(os.path.join('_dacegraphs', 'interstate-2.sdfg'))
    g.validate()

    assert g.apply_transformations(ConstAssignmentStateFusion) == 0
    assert g.apply_transformations(ConstAssignmentStateFusion,
                                   options={'use_grid_strided_loops': True}) == 1
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

    assert g.apply_transformations(ConstAssignmentMapFusion) == 1
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
def assign_front_face(A: dace.float32[K, M, N]):
    for t1, t2 in dace.map[0:K, 0:N]:
        A[t1, 0, t2] = 1


@dace.program
def assign_back_face(A: dace.float32[K, M, N]):
    for t1, t2 in dace.map[0:K, 0:N]:
        A[t1, M - 1, t2] = 1


@dace.program
def assign_left_face(A: dace.float32[K, M, N]):
    for t1, t2 in dace.map[0:K, 0:M]:
        A[t1, t2, 0] = 1


@dace.program
def assign_right_face(A: dace.float32[K, M, N]):
    for t1, t2 in dace.map[0:K, 0:M]:
        A[t1, t2, N - 1] = 1


@dace.program
def assign_bounary_3d(A: dace.float32[K, M, N], B: dace.float32[K, M, N]):
    assign_top_face(A)
    assign_bottom_face(B)
    assign_front_face(A)
    assign_back_face(B)
    assign_left_face(A)
    assign_right_face(B)


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

    assert g.apply_transformations_repeated(ConstAssignmentMapFusion, options={'use_grid_strided_loops': True}) == 3
    g.save(os.path.join('_dacegraphs', '3d-1.sdfg'))
    g.validate()
    our_A = deepcopy(A)
    our_B = deepcopy(B)
    g(A=our_A, B=our_B, K=3, M=4, N=5)

    # Here, the state fusion can apply only with GSLs.
    assert g.apply_transformations_repeated(ConstAssignmentStateFusion) == 0
    assert g.apply_transformations_repeated(ConstAssignmentStateFusion, options={'use_grid_strided_loops': True}) == 2
    g.save(os.path.join('_dacegraphs', '3d-2.sdfg'))
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

    assert g.apply_transformations(ConstAssignmentMapFusion) == 1
    g.save(os.path.join('_dacegraphs', 'branched-1.sdfg'))
    g.validate()
    our_A = deepcopy(A)
    our_B = deepcopy(B)
    g(A=our_A, B=our_B, M=4, N=5)

    # print(our_A)
    assert np.allclose(our_A, actual_A)


@dace.program
def assign_bottom_face_flipped(A: dace.float32[K, M, N]):
    for t2, t1 in dace.map[0:N, 0:M]:
        A[K - 1, t1, t2] = 1


@dace.program
def assign_bounary_3d_with_flip(A: dace.float32[K, M, N], B: dace.float32[K, M, N]):
    assign_top_face(A)
    assign_bottom_face_flipped(B)


def test_does_not_permute_to_fuse():
    """ Negative test """
    A = np.random.uniform(size=(3, 4, 5)).astype(np.float32)
    B = np.random.uniform(size=(3, 4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_bounary_3d_with_flip.to_sdfg(simplify=True, validate=True, use_cache=False)
    g.apply_transformations_repeated(StateFusionExtended, validate_all=True)
    g.save(os.path.join('_dacegraphs', '3d-flip-0.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    actual_B = deepcopy(B)
    g(A=actual_A, B=actual_B, K=3, M=4, N=5)

    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 0


@dace.program
def assign_mixed_dims(A: dace.float32[K, M, N], B: dace.float32[K, M, N]):
    assign_top_face(A)
    assign_left_col(B[0, :, :])


def test_does_not_extend_to_fuse():
    """ Negative test """
    A = np.random.uniform(size=(3, 4, 5)).astype(np.float32)
    B = np.random.uniform(size=(3, 4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_mixed_dims.to_sdfg(simplify=True, validate=True, use_cache=False)
    g.apply_transformations_repeated(StateFusionExtended, validate_all=True)
    g.save(os.path.join('_dacegraphs', '3d-mixed-0.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    actual_B = deepcopy(B)
    g(A=actual_A, B=actual_B, K=3, M=4, N=5)

    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 0


@dace.program
def assign_bottom_face_42(A: dace.float32[K, M, N]):
    for t1, t2 in dace.map[0:M, 0:N]:
        A[K - 1, t1, t2] = 42


@dace.program
def assign_bottom_face_index_sum(A: dace.float32[K, M, N]):
    for t1, t2 in dace.map[0:M, 0:N]:
        A[K - 1, t1, t2] = t1 + t2


@dace.program
def assign_inconsistent_values_1(A: dace.float32[K, M, N]):
    assign_top_face(A)
    assign_bottom_face_42(A)


@dace.program
def assign_inconsistent_values_2(A: dace.float32[K, M, N]):
    assign_top_face(A)
    assign_bottom_face_index_sum(A)


def test_does_not_fuse_with_inconsistent_assignments():
    """ Negative test """
    A = np.random.uniform(size=(3, 4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_inconsistent_values_1.to_sdfg(simplify=True, validate=True, use_cache=False)
    g.apply_transformations_repeated(StateFusionExtended, validate_all=True)
    g.save(os.path.join('_dacegraphs', '3d-inconsistent-0.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    g(A=actual_A, K=3, M=4, N=5)

    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 0

    # Try another case: Construct SDFG with the maps on separate states.
    g = assign_inconsistent_values_2.to_sdfg(simplify=True, validate=True, use_cache=False)
    g.apply_transformations_repeated(StateFusionExtended, validate_all=True)
    g.save(os.path.join('_dacegraphs', '3d-inconsistent-1.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    g(A=actual_A, K=3, M=4, N=5)

    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 0


@dace.program
def tasklet_between_maps(A: dace.float32[K, M, N]):
    assign_top_face(A)
    A[0, 0, 0] = 1
    assign_bottom_face(A)


def test_does_not_fuse_with_unsuitable_dependencies():
    """ Negative test """
    A = np.random.uniform(size=(3, 4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = tasklet_between_maps.to_sdfg(simplify=True, validate=True, use_cache=False)
    g.apply_transformations_repeated(StateFusionExtended, validate_all=True)
    g.save(os.path.join('_dacegraphs', '3d-baddeps-0.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    g(A=actual_A, K=3, M=4, N=5)

    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 0


@dace.program
def assign_top_face_self_copy(A: dace.float32[K, M, N]):
    for t1, t2 in dace.map[0:M, 0:N]:
        A[0, t1, t2] = A[0, t1, t2]


@dace.program
def first_map_reads_data(A: dace.float32[K, M, N]):
    assign_top_face_self_copy(A)
    assign_bottom_face(A)


def test_does_not_fuse_when_the_first_map_reads_anything_at_all():
    """ Negative test """
    A = np.random.uniform(size=(3, 4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = first_map_reads_data.to_sdfg(simplify=True, validate=True, use_cache=False)
    g.save(os.path.join('_dacegraphs', '3d-map1-reads-0.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    g(A=actual_A, K=3, M=4, N=5)

    # The state fusion won't work.
    assert g.apply_transformations_repeated(ConstAssignmentStateFusion) == 0

    # Fuse the states explicitly anyway.
    g.apply_transformations_repeated(StateFusionExtended, validate_all=True)
    g.save(os.path.join('_dacegraphs', '3d-map1-reads-1.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    g(A=actual_A, K=3, M=4, N=5)

    # The map fusion won't work.
    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 0


if __name__ == '__main__':
    test_within_state_fusion()
    test_interstate_fusion()
    test_free_floating_fusion()
    test_fusion_with_branch()
    test_fusion_with_multiple_indices()
