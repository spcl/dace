import os
from collections.abc import Collection
from copy import deepcopy
from itertools import chain
from typing import Tuple, Sequence

import numpy as np

import dace
from dace import SDFG, Memlet
from dace.properties import CodeBlock
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import SDFGState
from dace.subsets import Range
from dace.transformation.dataflow.const_assignment_fusion import ConstAssignmentMapFusion, ConstAssignmentStateFusion
from dace.transformation.interstate import StateFusionExtended

K, M, N = dace.symbol('K'), dace.symbol('M'), dace.symbol('N')


def _add_face_assignment_map(g: SDFGState, name: str, lims: Sequence[Tuple[str, dace.symbol]],
                             fixed_dims: Collection[Tuple[int, int]], assigned_val: int, array: str):
    idx = [k for k, v in lims]
    for fd, at in fixed_dims:
        idx.insert(fd, str(at))
    t, en, ex = g.add_mapped_tasklet(name, [(k, Range([(0, v - 1, 1)])) for k, v in lims],
                                     {}, f"__out = {assigned_val}", {'__out': Memlet(expr=f"{array}[{','.join(idx)}]")},
                                     external_edges=True)
    return en, ex, t


def _simple_if_block(name: str, cond: str, val: int):
    subg = SDFG(name)
    subg.add_array('tmp', (1,), dace.float32)
    # Outer structure.
    head = subg.add_state('if_head')
    branch = subg.add_state('if_b1')
    tail = subg.add_state('if_tail')
    subg.add_edge(head, branch, InterstateEdge(condition=f"({cond})"))
    subg.add_edge(head, tail, InterstateEdge(condition=f"(not ({cond}))"))
    subg.add_edge(branch, tail, InterstateEdge())
    # Inner structure.
    t = branch.add_tasklet('top', inputs={}, outputs={'__out'}, code=f"__out = {val}")
    tmp = branch.add_access('tmp')
    branch.add_edge(t, '__out', tmp, None, Memlet(expr='tmp[0]'))
    return subg


def assign_bounary_sdfg():
    g = SDFG('prog')
    g.add_array('A', (M, N), dace.float32)

    st0 = g.add_state('top')
    _add_face_assignment_map(st0, 'top', [('j', N)], [(0, 0)], 1, 'A')
    st1 = g.add_state('bottom')
    _add_face_assignment_map(st1, 'bottom', [('j', N)], [(0, M - 1)], 1, 'A')
    st2 = g.add_state('left')
    _add_face_assignment_map(st2, 'left', [('i', M)], [(1, 0)], 1, 'A')
    st3 = g.add_state('right')
    _add_face_assignment_map(st3, 'right', [('i', M)], [(1, N - 1)], 1, 'A')

    g.add_edge(st0, st1, dace.InterstateEdge())
    g.add_edge(st1, st2, dace.InterstateEdge())
    g.add_edge(st2, st3, dace.InterstateEdge())

    return g


def test_within_state_fusion():
    A = np.random.uniform(size=(4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_bounary_sdfg()
    # Fuse the two states so that the const-assignment-fusion is applicable.
    g.apply_transformations_repeated(StateFusionExtended, validate_all=True)
    g.save(os.path.join('_dacegraphs', 'simple-0.sdfg'))
    g.validate()
    g.compile()

    # Get the reference data.
    actual_A = deepcopy(A)
    g(A=actual_A, M=4, N=5)

    assert g.apply_transformations_repeated(ConstAssignmentMapFusion, options={'use_grid_strided_loops': True}) == 3
    g.save(os.path.join('_dacegraphs', 'simple-1.sdfg'))
    g.validate()
    g.compile()

    # Get our data.
    our_A = deepcopy(A)
    g(A=our_A, M=4, N=5)

    # Verify numerically.
    assert np.allclose(our_A, actual_A)


def test_interstate_fusion():
    A = np.random.uniform(size=(4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_bounary_sdfg()
    g.save(os.path.join('_dacegraphs', 'interstate-0.sdfg'))
    g.validate()
    g.compile()

    # Get the reference data.
    actual_A = deepcopy(A)
    g(A=actual_A, M=4, N=5)

    assert g.apply_transformations_repeated(ConstAssignmentStateFusion, options={'use_grid_strided_loops': True}) == 3
    g.save(os.path.join('_dacegraphs', 'interstate-1.sdfg'))
    g.validate()
    g.compile()

    # Get our data.
    our_A = deepcopy(A)
    g(A=our_A, M=4, N=5)

    # Verify numerically.
    assert np.allclose(our_A, actual_A)


def assign_bounary_free_floating_sdfg():
    g = SDFG('prog')
    g.add_array('A', (M, N), dace.float32)
    g.add_array('B', (M, N), dace.float32)

    st0 = g.add_state('st0')
    _add_face_assignment_map(st0, 'top', [('j', N)], [(0, 0)], 1, 'A')
    _add_face_assignment_map(st0, 'bottom', [('j', N)], [(0, M - 1)], 2, 'B')

    return g


def test_free_floating_fusion():
    A = np.random.uniform(size=(4, 5)).astype(np.float32)
    B = np.random.uniform(size=(4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_bounary_free_floating_sdfg()
    # g = assign_bounary_free_floating.to_sdfg(simplify=True, validate=True, use_cache=False)
    g.save(os.path.join('_dacegraphs', 'floating-0.sdfg'))
    g.validate()
    g.compile()

    # Get the reference data.
    actual_A = deepcopy(A)
    actual_B = deepcopy(B)
    g(A=actual_A, B=actual_B, M=4, N=5)

    assert g.apply_transformations(ConstAssignmentMapFusion) == 1
    g.save(os.path.join('_dacegraphs', 'floating-1.sdfg'))
    g.validate()

    # Get our data.
    our_A = deepcopy(A)
    our_B = deepcopy(B)
    g(A=our_A, B=our_B, M=4, N=5)

    # Verify numerically.
    assert np.allclose(our_A, actual_A)


def assign_boundary_3d_sdfg():
    g = SDFG('prog')
    g.add_array('A', (K, M, N), dace.float32)
    g.add_array('B', (K, M, N), dace.float32)

    st0 = g.add_state('top')
    _add_face_assignment_map(st0, 'top', [('m', M), ('n', N)], [(0, 0)], 1, 'A')
    _add_face_assignment_map(st0, 'bottom', [('m', M), ('n', N)], [(0, K - 1)], 2, 'B')
    _add_face_assignment_map(st0, 'front', [('k', K), ('n', N)], [(1, 0)], 1, 'A')
    _add_face_assignment_map(st0, 'back', [('k', K), ('n', N)], [(1, M - 1)], 2, 'B')
    _add_face_assignment_map(st0, 'left', [('k', K), ('m', M)], [(2, 0)], 1, 'A')
    _add_face_assignment_map(st0, 'right', [('k', K), ('m', M)], [(2, N - 1)], 2, 'B')

    return g


def test_fusion_with_multiple_indices():
    A = np.random.uniform(size=(3, 4, 5)).astype(np.float32)
    B = np.random.uniform(size=(3, 4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_boundary_3d_sdfg()
    # g = assign_bounary_3d.to_sdfg(simplify=True, validate=True, use_cache=False)
    g.save(os.path.join('_dacegraphs', '3d-0.sdfg'))
    g.validate()
    g.compile()

    # Get the reference data.
    actual_A = deepcopy(A)
    actual_B = deepcopy(B)
    g(A=actual_A, B=actual_B, K=3, M=4, N=5)

    assert g.apply_transformations_repeated(ConstAssignmentMapFusion, options={'use_grid_strided_loops': False}) == 3
    g.save(os.path.join('_dacegraphs', '3d-1.sdfg'))
    g.validate()
    g.compile()

    # Get our data.
    our_A = deepcopy(A)
    our_B = deepcopy(B)
    g(A=our_A, B=our_B, K=3, M=4, N=5)

    # Verify numerically.
    assert np.allclose(our_A, actual_A)

    # Here, the map fusion can apply only with GSLs.
    assert g.apply_transformations_repeated(ConstAssignmentMapFusion, options={'use_grid_strided_loops': False}) == 0
    assert g.apply_transformations_repeated(ConstAssignmentMapFusion, options={'use_grid_strided_loops': True}) == 2
    g.save(os.path.join('_dacegraphs', '3d-2.sdfg'))
    g.validate()
    g.compile()

    # Get our data.
    our_A = deepcopy(A)
    our_B = deepcopy(B)
    g(A=our_A, B=our_B, K=3, M=4, N=5)

    # Verify numerically.
    assert np.allclose(our_A, actual_A)


def assign_bounary_with_branch_sdfg():
    g = SDFG('prog')
    g.add_array('A', (M, N), dace.float32)
    g.add_array('B', (M, N), dace.float32)

    st0 = g.add_state('st0')
    en, ex, t = _add_face_assignment_map(st0, 'top', [('j', N)], [(0, 0)], 1, 'A')
    new_t = _simple_if_block('if_block', 'j == 0', 1)
    new_t = st0.add_nested_sdfg(new_t, None, {}, {'tmp'}, symbol_mapping={'j': 'j'})
    for e in list(chain(st0.in_edges(t), st0.out_edges(t))):
        st0.remove_edge(e)
    st0.add_nedge(en, new_t, Memlet())
    st0.add_edge(new_t, 'tmp', ex, 'IN_A', Memlet(expr='A[0, j]'))
    st0.remove_node(t)

    _add_face_assignment_map(st0, 'bottom', [('j', N)], [(0, M - 1)], 1, 'A')

    return g


def test_fusion_with_branch():
    A = np.random.uniform(size=(4, 5)).astype(np.float32)
    B = np.random.uniform(size=(4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_bounary_with_branch_sdfg()
    # g = assign_bounary_with_branch.to_sdfg(simplify=True, validate=True, use_cache=False)
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


def assign_bounary_3d_with_flip_sdfg():
    g = SDFG('prog')
    g.add_array('A', (K, M, N), dace.float32)

    st0 = g.add_state('st0')
    _add_face_assignment_map(st0, 'face', [('j', M), ('k', N)], [(0, 0)], 1, 'A')
    en, _, _ = _add_face_assignment_map(st0, 'face', [('j', M), ('k', N)], [(0, K - 1)], 1, 'A')
    en.map.range = Range(reversed(en.map.range.ranges))
    en.map.params = list(reversed(en.map.params))

    return g


def test_does_not_permute_to_fuse():
    """ Negative test """
    # Construct SDFG with the maps on separate states.
    g = assign_bounary_3d_with_flip_sdfg()
    g.save(os.path.join('_dacegraphs', '3d-flip-0.sdfg'))
    g.validate()
    g.compile()

    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 0


def assign_mixed_dims_sdfg():
    g = SDFG('prog')
    g.add_array('A', (K, M, N), dace.float32)
    g.add_array('B', (K, M, N), dace.float32)

    st0 = g.add_state('st0')
    _add_face_assignment_map(st0, 'face', [('j', M), ('k', N)], [(0, 0)], 1, 'A')
    st1 = g.add_state('st1')
    _add_face_assignment_map(st1, 'edge', [('k', N)], [(0, 0), (1, 0)], 2, 'B')
    g.add_edge(st0, st1, InterstateEdge())

    return g


def test_does_not_extend_to_fuse():
    """ Negative test """
    # Construct SDFG with the maps on separate states.
    g = assign_mixed_dims_sdfg()
    g.save(os.path.join('_dacegraphs', '3d-mixed-0.sdfg'))
    g.validate()
    g.compile()

    # Has multiple states, but will not fuse them if the number of dimensions are different.
    assert g.apply_transformations_repeated(ConstAssignmentStateFusion) == 0
    # We can fuse them manually.
    assert g.apply_transformations_repeated(StateFusionExtended) == 1
    g.save(os.path.join('_dacegraphs', '3d-mixed-1.sdfg'))
    # But still won't fuse them maps.
    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 0


def assign_inconsistent_values_different_constants_sdfg():
    g = SDFG('prog')
    g.add_array('A', (K, M, N), dace.float32)

    st0 = g.add_state('st0')
    _add_face_assignment_map(st0, 'face', [('j', M), ('k', N)], [(0, 0)], 1, 'A')
    st1 = g.add_state('st1')
    _add_face_assignment_map(st1, 'face', [('j', M), ('k', N)], [(0, K - 1)], 42, 'A')
    g.add_edge(st0, st1, InterstateEdge())

    return g


def assign_inconsistent_values_non_constant_sdfg():
    g = SDFG('prog')
    g.add_array('A', (K, M, N), dace.float32)

    st0 = g.add_state('st0')
    _add_face_assignment_map(st0, 'face', [('j', M), ('k', N)], [(0, 0)], 1, 'A')
    st1 = g.add_state('st1')
    _, _, t = _add_face_assignment_map(st1, 'face', [('j', M), ('k', N)], [(0, K - 1)], 1, 'A')
    t.code = CodeBlock('__out = j + k')
    g.add_edge(st0, st1, InterstateEdge())

    return g


def test_does_not_fuse_with_inconsistent_assignments():
    """ Negative test """
    # Construct SDFG with the maps on separate states.
    g = assign_inconsistent_values_different_constants_sdfg()
    g.save(os.path.join('_dacegraphs', '3d-inconsistent-0a.sdfg'))
    g.validate()
    g.compile()

    # Has multiple states, but won't fuse them.
    assert g.apply_transformations_repeated(ConstAssignmentStateFusion) == 0
    # We can fuse them manually.
    assert g.apply_transformations_repeated(StateFusionExtended) == 1
    g.save(os.path.join('_dacegraphs', '3d-inconsistent-1a.sdfg'))
    # But still won't fuse them maps.
    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 0

    # Try another case.
    # Construct SDFG with the maps on separate states.
    g = assign_inconsistent_values_non_constant_sdfg()
    g.save(os.path.join('_dacegraphs', '3d-inconsistent-0b.sdfg'))
    g.validate()
    g.compile()

    # Has multiple states, but won't fuse them.
    assert g.apply_transformations_repeated(ConstAssignmentStateFusion) == 0
    # We can fuse them manually.
    assert g.apply_transformations_repeated(StateFusionExtended) == 1
    g.save(os.path.join('_dacegraphs', '3d-inconsistent-1b.sdfg'))
    # But still won't fuse them maps.
    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 0


def sdfg_with_tasklet_between_maps():
    g = SDFG('prog')
    g.add_array('A', (K, M, N), dace.float32)

    st0 = g.add_state('st0')
    _, ex1, _ = _add_face_assignment_map(st0, 'face', [('j', M), ('k', N)], [(0, 0)], 1, 'A')
    en2, _, _ = _add_face_assignment_map(st0, 'face', [('j', M), ('k', N)], [(0, K - 1)], 1, 'A')
    t = st0.add_tasklet('noop', {}, {}, '')
    st0.add_nedge(st0.out_edges(ex1)[0].dst, en2, Memlet())
    st0.add_nedge(st0.out_edges(ex1)[0].dst, t, Memlet())
    st0.add_nedge(t, en2, Memlet())

    return g


def test_does_not_fuse_with_unsuitable_dependencies():
    """ Negative test """
    A = np.random.uniform(size=(3, 4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = sdfg_with_tasklet_between_maps()
    g.save(os.path.join('_dacegraphs', '3d-baddeps-0.sdfg'))
    g.validate()
    g.compile()

    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 0


def sdfg_where_first_map_reads_data():
    g = SDFG('prog')
    g.add_array('A', (M, N), dace.float32)

    st0 = g.add_state('top')
    en1, _, t = _add_face_assignment_map(st0, 'top', [('j', N)], [(0, 0)], 1, 'A')
    en1.add_in_connector('IN_A')
    en1.add_out_connector('OUT_A')
    t.add_in_connector('__blank')
    A = st0.add_access('A')
    st0.add_edge(A, None, en1, 'IN_A', Memlet(expr='A[0, 0:N]'))
    for e in st0.out_edges(en1):
        st0.remove_edge(e)
    st0.add_edge(en1, 'OUT_A', t, '__blank', Memlet(expr='A[0, j]'))

    st1 = g.add_state('bottom')
    _add_face_assignment_map(st1, 'bottom', [('j', N)], [(0, M - 1)], 1, 'A')

    g.add_edge(st0, st1, InterstateEdge())

    return g


def test_does_not_fuse_when_the_first_map_reads_anything_at_all():
    """ Negative test """
    A = np.random.uniform(size=(3, 4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = sdfg_where_first_map_reads_data()
    g.save(os.path.join('_dacegraphs', '3d-map1-reads-0.sdfg'))
    g.validate()
    g.compile()

    # The state fusion won't work.
    assert g.apply_transformations_repeated(ConstAssignmentStateFusion) == 0

    # Fuse the states explicitly anyway.
    g.apply_transformations_repeated(StateFusionExtended, validate_all=True)
    g.save(os.path.join('_dacegraphs', '3d-map1-reads-1.sdfg'))
    g.validate()
    g.compile()

    # The map fusion won't work.
    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 0


def sdfg_where_first_state_has_multiple_toplevel_maps():
    g = SDFG('prog')
    g.add_array('A', (M, N), dace.float32)

    st0 = g.add_state('st0')
    _add_face_assignment_map(st0, 'top', [('j', N)], [(0, 0)], 1, 'A')
    _add_face_assignment_map(st0, 'bottom', [('j', N)], [(0, M - 1)], 1, 'A')

    st1 = g.add_state('st1')
    _add_face_assignment_map(st1, 'left', [('i', M)], [(1, 0)], 1, 'A')

    g.add_edge(st0, st1, InterstateEdge())

    return g


def test_does_not_fuse_when_the_first_state_has_multiple_toplevel_maps():
    """ Negative test """
    A = np.random.uniform(size=(3, 4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = sdfg_where_first_state_has_multiple_toplevel_maps()
    g.save(os.path.join('_dacegraphs', '3d-multimap-state-0.sdfg'))
    g.validate()
    g.compile()

    # Get the reference data.
    actual_A = deepcopy(A)
    g(A=actual_A, K=3, M=4, N=5)

    # The state fusion won't work.
    assert g.apply_transformations_repeated(ConstAssignmentStateFusion) == 0

    # Fuse the states explicitly anyway.
    g.apply_transformations_repeated(StateFusionExtended, validate_all=True)
    g.save(os.path.join('_dacegraphs', '3d-multimap-state-1.sdfg'))
    g.validate()
    g.compile()

    # But now, the fusion will work!
    assert g.apply_transformations_repeated(ConstAssignmentMapFusion) == 1
    g.save(os.path.join('_dacegraphs', '3d-multimap-state-2.sdfg'))
    g.validate()
    g.compile()

    # Get our data.
    our_A = deepcopy(A)
    g(A=our_A, K=3, M=4, N=5)

    # Verify numerically.
    assert np.allclose(our_A, actual_A)


if __name__ == '__main__':
    test_within_state_fusion()
    test_interstate_fusion()
    test_free_floating_fusion()
    test_fusion_with_branch()
    test_fusion_with_multiple_indices()
    test_does_not_extend_to_fuse()
    test_does_not_permute_to_fuse()
    test_does_not_fuse_with_inconsistent_assignments()
    test_does_not_fuse_with_unsuitable_dependencies()
    test_does_not_fuse_when_the_first_map_reads_anything_at_all()
    test_does_not_fuse_when_the_first_state_has_multiple_toplevel_maps()
