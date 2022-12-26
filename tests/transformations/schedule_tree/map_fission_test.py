# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace import nodes
from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree, as_sdfg
from dace.sdfg.analysis.schedule_tree.transformations import map_fission
from dace.transformation.helpers import nest_state_subgraph
from tests.transformations.mapfission_test import mapfission_sdfg


def test_map_with_tasklet_and_library():

    N = dace.symbol('N')
    @dace.program
    def map_with_tasklet_and_library(A: dace.float32[N, 5, 5], B: dace.float32[N, 5, 5], cst: dace.int32):
        out = np.ndarray((N, 5, 5), dtype=dace.float32)
        for i in dace.map[0:N]:
            out[i] = cst * (A[i] @ B[i])
        return out
    
    rng = np.random.default_rng(42)
    A = rng.random((10, 5, 5), dtype=np.float32)
    B = rng.random((10, 5, 5), dtype=np.float32)
    cst = rng.integers(0, 100, dtype=np.int32)
    ref = cst * (A @ B)

    val0 = map_with_tasklet_and_library(A, B, cst)
    sdfg0 = map_with_tasklet_and_library.to_sdfg()
    tree = as_schedule_tree(sdfg0)
    result = map_fission(tree.children[0], tree)
    assert result
    sdfg1 = as_sdfg(tree)
    val1 = sdfg1(A=A, B=B, cst=cst, N=A.shape[0])
    
    assert np.allclose(val0, ref)
    assert np.allclose(val1, ref)


def test_subgraph():

    rng = np.random.default_rng(42)
    A = rng.random((4, ))
    ref = np.zeros([2], dtype=np.float64)
    ref[0] = (A[0] + A[1]) + (A[0] * 2 * A[1] * 2) + (A[0] * 3) + 5.0
    ref[1] = (A[2] + A[3]) + (A[2] * 2 * A[3] * 2) + (A[2] * 3) + 5.0
    val = np.empty((2, ))

    sdfg0 = mapfission_sdfg()
    tree = as_schedule_tree(sdfg0)
    pcode, _ = tree.as_python()
    print(pcode)
    result = map_fission(tree.children[0], tree)
    assert result
    pcode, _ = tree.as_python()
    print(pcode)
    sdfg1 = as_sdfg(tree)
    sdfg1(A=A, B=val)
    
    assert np.allclose(val, ref)


def test_nested_subgraph():

    rng = np.random.default_rng(42)
    A = rng.random((4, ))
    ref = np.zeros([2], dtype=np.float64)
    ref[0] = (A[0] + A[1]) + (A[0] * 2 * A[1] * 2) + (A[0] * 3) + 5.0
    ref[1] = (A[2] + A[3]) + (A[2] * 2 * A[3] * 2) + (A[2] * 3) + 5.0
    val = np.empty((2, ))

    sdfg0 = mapfission_sdfg()
    state = sdfg0.nodes()[0]
    topmap = next(node for node in state.nodes() if isinstance(node, nodes.MapEntry) and node.label == 'outer')
    subgraph = state.scope_subgraph(topmap, include_entry=False, include_exit=False)
    nest_state_subgraph(sdfg0, state, subgraph)
    tree = as_schedule_tree(sdfg0)
    result = map_fission(tree.children[0], tree)
    assert result
    pcode, _ = tree.as_python()
    print(pcode)
    sdfg1 = as_sdfg(tree)
    sdfg1(A=A, B=val)
    
    assert np.allclose(val, ref)


def test_nested_transient():
    """ Test nested SDFGs with transients. """

    # Inner SDFG
    nsdfg = dace.SDFG('nested')
    nsdfg.add_array('a', [1], dace.float64)
    nsdfg.add_array('b', [1], dace.float64)
    nsdfg.add_transient('t', [1], dace.float64)

    # a->t state
    nstate = nsdfg.add_state()
    irnode = nstate.add_read('a')
    task = nstate.add_tasklet('t1', {'inp'}, {'out'}, 'out = 2*inp')
    iwnode = nstate.add_write('t')
    nstate.add_edge(irnode, None, task, 'inp', dace.Memlet.simple('a', '0'))
    nstate.add_edge(task, 'out', iwnode, None, dace.Memlet.simple('t', '0'))

    # t->a state
    first_state = nstate
    nstate = nsdfg.add_state()
    irnode = nstate.add_read('t')
    task = nstate.add_tasklet('t2', {'inp'}, {'out'}, 'out = 3*inp')
    iwnode = nstate.add_write('b')
    nstate.add_edge(irnode, None, task, 'inp', dace.Memlet.simple('t', '0'))
    nstate.add_edge(task, 'out', iwnode, None, dace.Memlet.simple('b', '0'))

    nsdfg.add_edge(first_state, nstate, dace.InterstateEdge())

    # Outer SDFG
    sdfg = dace.SDFG('nested_transient_fission')
    sdfg.add_array('A', [2], dace.float64)
    state = sdfg.add_state()
    rnode = state.add_read('A')
    wnode = state.add_write('A')
    me, mx = state.add_map('outer', dict(i='0:2'))
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'a'}, {'b'})
    state.add_memlet_path(rnode, me, nsdfg_node, dst_conn='a', memlet=dace.Memlet.simple('A', 'i'))
    state.add_memlet_path(nsdfg_node, mx, wnode, src_conn='b', memlet=dace.Memlet.simple('A', 'i'))

    # self.assertGreater(sdfg.apply_transformations_repeated(MapFission), 0)
    tree = as_schedule_tree(sdfg)
    result = map_fission(tree.children[0], tree)
    assert result
    sdfg = as_sdfg(tree)

    # Test
    A = np.random.rand(2)
    expected = A * 6
    sdfg(A=A)
    # self.assertTrue(np.allclose(A, expected))
    assert np.allclose(A, expected)


def test_inputs_outputs():
    """
    Test subgraphs where the computation modules that are in the middle
    connect to the outside.
    """

    sdfg = dace.SDFG('inputs_outputs_fission')
    sdfg.add_array('in1', [2], dace.float64)
    sdfg.add_array('in2', [2], dace.float64)
    sdfg.add_scalar('tmp', dace.float64, transient=True)
    sdfg.add_array('out1', [2], dace.float64)
    sdfg.add_array('out2', [2], dace.float64)
    state = sdfg.add_state()
    in1 = state.add_read('in1')
    in2 = state.add_read('in2')
    out1 = state.add_write('out1')
    out2 = state.add_write('out2')
    me, mx = state.add_map('outer', dict(i='0:2'))
    t1 = state.add_tasklet('t1', {'i1'}, {'o1', 'o2'}, 'o1 = i1 * 2; o2 = i1 * 5')
    t2 = state.add_tasklet('t2', {'i1', 'i2'}, {'o1'}, 'o1 = i1 * i2')
    state.add_memlet_path(in1, me, t1, dst_conn='i1', memlet=dace.Memlet.simple('in1', 'i'))
    state.add_memlet_path(in2, me, t2, dst_conn='i2', memlet=dace.Memlet.simple('in2', 'i'))
    state.add_edge(t1, 'o1', t2, 'i1', dace.Memlet.simple('tmp', '0'))
    state.add_memlet_path(t2, mx, out1, src_conn='o1', memlet=dace.Memlet.simple('out1', 'i'))
    state.add_memlet_path(t1, mx, out2, src_conn='o2', memlet=dace.Memlet.simple('out2', 'i'))

    # self.assertGreater(sdfg.apply_transformations(MapFission), 0)
    tree = as_schedule_tree(sdfg)
    result = map_fission(tree.children[0], tree)
    assert result
    sdfg = as_sdfg(tree)

    # Test
    A, B, C, D = tuple(np.random.rand(2) for _ in range(4))
    expected_C = (A * 2) * B
    expected_D = A * 5
    sdfg(in1=A, in2=B, out1=C, out2=D)
    # self.assertTrue(np.allclose(C, expected_C))
    # self.assertTrue(np.allclose(D, expected_D))
    assert np.allclose(C, expected_C)
    assert np.allclose(D, expected_D)


def test_offsets():
    sdfg = dace.SDFG('mapfission_offsets')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_scalar('interim', dace.float64, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map('outer', dict(i='10:20'))

    t1 = state.add_tasklet('addone', {'a'}, {'b'}, 'b = a + 1')
    t2 = state.add_tasklet('addtwo', {'a'}, {'b'}, 'b = a + 2')

    aread = state.add_read('A')
    awrite = state.add_write('A')
    state.add_memlet_path(aread, me, t1, dst_conn='a', memlet=dace.Memlet.simple('A', 'i'))
    state.add_edge(t1, 'b', t2, 'a', dace.Memlet.simple('interim', '0'))
    state.add_memlet_path(t2, mx, awrite, src_conn='b', memlet=dace.Memlet.simple('A', 'i'))

    # self.assertGreater(sdfg.apply_transformations(MapFission), 0)
    tree = as_schedule_tree(sdfg)
    pcode, _ = tree.as_python()
    print(pcode)
    result = map_fission(tree.children[0], tree)
    assert result
    pcode, _ = tree.as_python()
    print(pcode)
    sdfg = as_sdfg(tree)

    # dace.propagate_memlets_sdfg(sdfg)
    # sdfg.validate()

    # Test
    A = np.random.rand(20)
    expected = A.copy()
    expected[10:] += 3
    sdfg(A=A)
    # self.assertTrue(np.allclose(A, expected))
    assert np.allclose(A, expected)


def test_offsets_array():
    sdfg = dace.SDFG('mapfission_offsets2')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('interim', [1], dace.float64, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map('outer', dict(i='10:20'))

    t1 = state.add_tasklet('addone', {'a'}, {'b'}, 'b = a + 1')
    interim = state.add_access('interim')
    t2 = state.add_tasklet('addtwo', {'a'}, {'b'}, 'b = a + 2')

    aread = state.add_read('A')
    awrite = state.add_write('A')
    state.add_memlet_path(aread, me, t1, dst_conn='a', memlet=dace.Memlet.simple('A', 'i'))
    state.add_edge(t1, 'b', interim, None, dace.Memlet.simple('interim', '0'))
    state.add_edge(interim, None, t2, 'a', dace.Memlet.simple('interim', '0'))
    state.add_memlet_path(t2, mx, awrite, src_conn='b', memlet=dace.Memlet.simple('A', 'i'))

    # self.assertGreater(sdfg.apply_transformations(MapFission), 0)
    tree = as_schedule_tree(sdfg)
    result = map_fission(tree.children[0], tree)
    assert result
    sdfg = as_sdfg(tree)

    # dace.propagate_memlets_sdfg(sdfg)
    # sdfg.validate()

    # Test
    A = np.random.rand(20)
    expected = A.copy()
    expected[10:] += 3
    sdfg(A=A)
    # self.assertTrue(np.allclose(A, expected))
    assert np.allclose(A, expected)


def test_mapfission_with_symbols():
    '''
    Tests MapFission in the case of a Map containing a NestedSDFG that is using some symbol from the top-level SDFG
    missing from the NestedSDFG's symbol mapping. Please note that this is an unusual case that is difficult to
    reproduce and ultimately unrelated to MapFission. Consider solving the underlying issue and then deleting this
    test and the corresponding (obsolete) code in MapFission.
    '''

    M, N = dace.symbol('M'), dace.symbol('N')

    sdfg = dace.SDFG('tasklet_code_with_symbols')
    sdfg.add_array('A', (M, N), dace.int32)
    sdfg.add_array('B', (M, N), dace.int32)

    state = sdfg.add_state('parent', is_start_state=True)
    me, mx = state.add_map('parent_map', {'i': '0:N'})

    nsdfg = dace.SDFG('nested_sdfg')
    nsdfg.add_scalar('inner_A', dace.int32)
    nsdfg.add_scalar('inner_B', dace.int32)

    nstate = nsdfg.add_state('child', is_start_state=True)
    na = nstate.add_access('inner_A')
    nb = nstate.add_access('inner_B')
    ta = nstate.add_tasklet('tasklet_A', {}, {'__out'}, '__out = M')
    tb = nstate.add_tasklet('tasklet_B', {}, {'__out'}, '__out = M')
    nstate.add_edge(ta, '__out', na, None, dace.Memlet.from_array('inner_A', nsdfg.arrays['inner_A']))
    nstate.add_edge(tb, '__out', nb, None, dace.Memlet.from_array('inner_B', nsdfg.arrays['inner_B']))

    a = state.add_access('A')
    b = state.add_access('B')
    t = nodes.NestedSDFG('child_sdfg', nsdfg, {}, {'inner_A', 'inner_B'}, {})
    nsdfg.parent = state
    nsdfg.parent_sdfg = sdfg
    nsdfg.parent_nsdfg_node = t
    state.add_node(t)
    state.add_nedge(me, t, dace.Memlet())
    state.add_memlet_path(t, mx, a, memlet=dace.Memlet('A[0, i]'), src_conn='inner_A')
    state.add_memlet_path(t, mx, b, memlet=dace.Memlet('B[0, i]'), src_conn='inner_B')

    # num = sdfg.apply_transformations_repeated(MapFission)
    tree = as_schedule_tree(sdfg)
    result = map_fission(tree.children[0], tree)
    assert result
    sdfg = as_sdfg(tree)

    A = np.ndarray((2, 10), dtype=np.int32)
    B = np.ndarray((2, 10), dtype=np.int32)
    sdfg(A=A, B=B, M=2, N=10)

    ref = np.full((10, ), fill_value=2, dtype=np.int32)

    assert np.array_equal(A[0], ref)
    assert np.array_equal(B[0], ref)


if __name__ == "__main__":
    test_map_with_tasklet_and_library()
    test_subgraph()
    test_nested_subgraph()
    test_nested_transient()
    test_inputs_outputs()
    test_offsets()
    test_offsets_array()
    test_mapfission_with_symbols()
