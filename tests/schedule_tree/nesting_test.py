# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
Nesting and dealiasing tests for schedule trees.
"""
import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree
from dace.transformation.dataflow import RemoveSliceView

import pytest

N = dace.symbol('N')
T = dace.symbol('T')


def test_stree_mpath_multiscope():

    @dace.program
    def tester(A: dace.float64[N, N]):
        for i in dace.map[0:N:T]:
            for j, k in dace.map[0:T, 0:N]:
                for l in dace.map[0:T]:
                    A[i + j, k + l] = 1

    # The test should generate different SDFGs for different simplify configurations,
    # but the same schedule tree
    stree = as_schedule_tree(tester.to_sdfg())
    assert [type(n) for n in stree.preorder_traversal()][1:] == [tn.MapScope, tn.MapScope, tn.MapScope, tn.TaskletNode]


def test_stree_mpath_multiscope_dependent():

    @dace.program
    def tester(A: dace.float64[N, N]):
        for i in dace.map[0:N:T]:
            for j, k in dace.map[0:T, 0:N]:
                for l in dace.map[0:k]:
                    A[i + j, l] = 1

    # The test should generate different SDFGs for different simplify configurations,
    # but the same schedule tree
    stree = as_schedule_tree(tester.to_sdfg())
    assert [type(n) for n in stree.preorder_traversal()][1:] == [tn.MapScope, tn.MapScope, tn.MapScope, tn.TaskletNode]


def test_stree_mpath_nested():

    @dace.program
    def nester(A, i, k, j):
        for l in range(k):
            A[i + j, l] = 1

    @dace.program
    def tester(A: dace.float64[N, N]):
        for i in dace.map[0:N:T]:
            for j, k in dace.map[0:T, 0:N]:
                nester(A, i, j, k)

    stree = as_schedule_tree(tester.to_sdfg())

    # Simplifying yields a different SDFG due to scalars and symbols, so testing is slightly different
    simplified = dace.Config.get_bool('optimizer', 'automatic_simplification')

    if simplified:
        assert [type(n)
                for n in stree.preorder_traversal()][1:] == [tn.MapScope, tn.MapScope, tn.ForScope, tn.TaskletNode]

    tasklet: tn.TaskletNode = list(stree.preorder_traversal())[-1]

    if simplified:
        assert str(next(iter(tasklet.out_memlets.values()))) == 'A[i + k, l]'
    else:
        assert str(next(iter(tasklet.out_memlets.values()))).endswith(', l]')


@pytest.mark.parametrize('dst_subset', (False, True))
def test_stree_copy_same_scope(dst_subset):
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [3 * N], dace.float64)
    sdfg.add_array('B', [3 * N], dace.float64)
    state = sdfg.add_state()

    r = state.add_read('A')
    w = state.add_write('B')
    if not dst_subset:
        state.add_nedge(r, w, dace.Memlet(data='A', subset='2*N:3*N', other_subset='N:2*N'))
    else:
        state.add_nedge(r, w, dace.Memlet(data='B', subset='N:2*N', other_subset='2*N:3*N'))

    stree = as_schedule_tree(sdfg)
    assert len(stree.children) == 1 and isinstance(stree.children[0], tn.CopyNode)
    assert stree.children[0].target == 'B'
    assert stree.children[0].as_string() == 'B[N:2*N] = copy A[2*N:3*N]'


@pytest.mark.parametrize('dst_subset', (False, True))
def test_stree_copy_different_scope(dst_subset):
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [3 * N], dace.float64)
    sdfg.add_array('B', [3 * N], dace.float64)
    state = sdfg.add_state()

    r = state.add_read('A')
    w = state.add_write('B')
    me, mx = state.add_map('something', dict(i='0:1'))
    if not dst_subset:
        state.add_memlet_path(r, me, w, memlet=dace.Memlet(data='A', subset='2*N:3*N', other_subset='N + i:2*N + i'))
    else:
        state.add_memlet_path(r, me, w, memlet=dace.Memlet(data='B', subset='N + i:2*N + i', other_subset='2*N:3*N'))
    state.add_nedge(w, mx, dace.Memlet())

    stree = as_schedule_tree(sdfg)
    stree_nodes = list(stree.preorder_traversal())[1:]
    assert [type(n) for n in stree_nodes] == [tn.MapScope, tn.CopyNode]
    assert stree_nodes[-1].target == 'B'
    assert stree_nodes[-1].as_string() == 'B[N + i:2*N + i] = copy A[2*N:3*N]'


def test_dealias_nested_call():

    @dace.program
    def nester(a, b):
        b[:] = a

    @dace.program
    def tester(a: dace.float64[40], b: dace.float64[40]):
        nester(b[1:21], a[10:30])

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(RemoveSliceView)

    stree = as_schedule_tree(sdfg)
    assert len(stree.children) == 1
    copy = stree.children[0]
    assert isinstance(copy, tn.CopyNode)
    assert copy.target == 'a'
    assert copy.memlet.data == 'b'
    assert str(copy.memlet.src_subset) == '1:21'
    assert str(copy.memlet.dst_subset) == '10:30'


def test_dealias_nested_call_samearray():

    @dace.program
    def nester(a, b):
        b[:] = a

    @dace.program
    def tester(a: dace.float64[40]):
        nester(a[1:21], a[10:30])

    sdfg = tester.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(RemoveSliceView)

    stree = as_schedule_tree(sdfg)
    assert len(stree.children) == 1
    copy = stree.children[0]
    assert isinstance(copy, tn.CopyNode)
    assert copy.target == 'a'
    assert copy.memlet.data == 'a'
    assert str(copy.memlet.src_subset) == '1:21'
    assert str(copy.memlet.dst_subset) == '10:30'


@pytest.mark.parametrize('simplify', (False, True))
def test_dealias_memlet_composition(simplify):

    def nester2(c):
        c[2] = 1

    def nester1(b):
        nester2(b[-5:])

    @dace.program
    def tester(a: dace.float64[N, N]):
        nester1(a[:, 1])

    sdfg = tester.to_sdfg(simplify=simplify)
    stree = as_schedule_tree(sdfg)

    # Simplifying yields a different SDFG due to views, so testing is slightly different
    if simplify:
        assert len(stree.children) == 1
        tasklet = stree.children[0]
        assert isinstance(tasklet, tn.TaskletNode)
        assert str(next(iter(tasklet.out_memlets.values()))) == 'a[N - 3, 1]'
    else:
        assert len(stree.children) == 3
        stree_nodes = list(stree.preorder_traversal())[1:]
        assert [type(n) for n in stree_nodes] == [tn.ViewNode, tn.ViewNode, tn.TaskletNode]


def test_dealias_interstate_edge():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)

    nsdfg = dace.SDFG('nester')
    nsdfg.add_array('A', [19], dace.float64)
    nsdfg.add_array('B', [15], dace.float64)
    nsdfg.add_symbol('m', dace.float64)
    nstate1 = nsdfg.add_state()
    nstate2 = nsdfg.add_state()
    nsdfg.add_edge(nstate1, nstate2, dace.InterstateEdge(condition='B[1] > 0', assignments=dict(m='A[2]')))

    # Connect to nested SDFG both with flipped definitions and offset memlets
    state = sdfg.add_state()
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'A', 'B'}, {})
    ra = state.add_read('A')
    rb = state.add_read('B')
    state.add_edge(ra, None, nsdfg_node, 'B', dace.Memlet('A[1:20]'))
    state.add_edge(rb, None, nsdfg_node, 'A', dace.Memlet('B[2:17]'))

    sdfg.validate()
    stree = as_schedule_tree(sdfg)
    nodes = list(stree.preorder_traversal())[1:]
    assert [type(n) for n in nodes] == [tn.StateIfScope, tn.GotoNode, tn.AssignNode]
    assert 'A[2]' in nodes[0].condition.as_string
    assert 'B[4]' in nodes[-1].value.as_string


if __name__ == '__main__':
    test_stree_mpath_multiscope()
    test_stree_mpath_multiscope_dependent()
    test_stree_mpath_nested()
    test_stree_copy_same_scope(False)
    test_stree_copy_same_scope(True)
    test_stree_copy_different_scope(False)
    test_stree_copy_different_scope(True)
    test_dealias_nested_call()
    test_dealias_nested_call_samearray()
    test_dealias_memlet_composition(False)
    test_dealias_memlet_composition(True)
    test_dealias_interstate_edge()
