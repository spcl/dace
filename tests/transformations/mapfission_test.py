# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import MapFission
from dace.transformation.helpers import nest_state_subgraph
import numpy as np
import unittest


def mapfission_sdfg():
    sdfg = dace.SDFG('mapfission')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [2], dace.float64)
    sdfg.add_scalar('scal', dace.float64, transient=True)
    sdfg.add_scalar('s1', dace.float64, transient=True)
    sdfg.add_transient('s2', [2], dace.float64)
    sdfg.add_transient('s3out', [1], dace.float64)
    state = sdfg.add_state()

    # Nodes
    rnode = state.add_read('A')
    ome, omx = state.add_map('outer', dict(i='0:2'))
    t1 = state.add_tasklet('one', {'a'}, {'b'}, 'b = a[0] + a[1]')
    ime2, imx2 = state.add_map('inner', dict(j='0:2'))
    t2 = state.add_tasklet('two', {'a'}, {'b'}, 'b = a * 2')
    s24node = state.add_access('s2')
    s34node = state.add_access('s3out')
    ime3, imx3 = state.add_map('inner', dict(j='0:2'))
    t3 = state.add_tasklet('three', {'a'}, {'b'}, 'b = a[0] * 3')
    scalar = state.add_tasklet('scalar', {}, {'out'}, 'out = 5.0')
    t4 = state.add_tasklet('four', {'ione', 'itwo', 'ithree', 'sc'}, {'out'},
                           'out = ione + itwo[0] * itwo[1] + ithree + sc')
    wnode = state.add_write('B')

    # Edges
    state.add_nedge(ome, scalar, dace.Memlet())
    state.add_memlet_path(rnode, ome, t1, memlet=dace.Memlet.simple('A', '2*i:2*i+2'), dst_conn='a')
    state.add_memlet_path(rnode, ome, ime2, t2, memlet=dace.Memlet.simple('A', '2*i+j'), dst_conn='a')
    state.add_memlet_path(t2, imx2, s24node, memlet=dace.Memlet.simple('s2', 'j'), src_conn='b')
    state.add_memlet_path(rnode, ome, ime3, t3, memlet=dace.Memlet.simple('A', '2*i:2*i+2'), dst_conn='a')
    state.add_memlet_path(t3, imx3, s34node, memlet=dace.Memlet.simple('s3out', '0'), src_conn='b')

    state.add_edge(t1, 'b', t4, 'ione', dace.Memlet.simple('s1', '0'))
    state.add_edge(s24node, None, t4, 'itwo', dace.Memlet.simple('s2', '0:2'))
    state.add_edge(s34node, None, t4, 'ithree', dace.Memlet.simple('s3out', '0'))
    state.add_edge(scalar, 'out', t4, 'sc', dace.Memlet.simple('scal', '0'))
    state.add_memlet_path(t4, omx, wnode, memlet=dace.Memlet.simple('B', 'i'), src_conn='out')

    sdfg.validate()
    return sdfg


def config():
    A = np.random.rand(4)
    expected = np.zeros([2], dtype=np.float64)
    expected[0] = (A[0] + A[1]) + (A[0] * 2 * A[1] * 2) + (A[0] * 3) + 5.0
    expected[1] = (A[2] + A[3]) + (A[2] * 2 * A[3] * 2) + (A[2] * 3) + 5.0
    return A, expected


class MapFissionTest(unittest.TestCase):
    def test_subgraph(self):
        A, expected = config()
        B = np.random.rand(2)

        graph = mapfission_sdfg()
        self.assertGreater(graph.apply_transformations(MapFission), 0)
        graph(A=A, B=B)

        self.assertTrue(np.allclose(B, expected))

    def test_nested_sdfg(self):
        A, expected = config()
        B = np.random.rand(2)

        # Nest the subgraph within the outer map, then apply transformation
        graph = mapfission_sdfg()
        state = graph.nodes()[0]
        topmap = next(node for node in state.nodes() if isinstance(node, nodes.MapEntry) and node.label == 'outer')
        subgraph = state.scope_subgraph(topmap, include_entry=False, include_exit=False)
        nest_state_subgraph(graph, state, subgraph)
        self.assertGreater(graph.apply_transformations(MapFission), 0)
        graph(A=A, B=B)
        self.assertTrue(np.allclose(B, expected))

    def test_nested_transient(self):
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

        self.assertGreater(sdfg.apply_transformations(MapFission), 0)

        # Test
        A = np.random.rand(2)
        expected = A * 6
        sdfg(A=A)
        self.assertTrue(np.allclose(A, expected))

    def test_inputs_outputs(self):
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

        self.assertGreater(sdfg.apply_transformations(MapFission), 0)

        # Test
        A, B, C, D = tuple(np.random.rand(2) for _ in range(4))
        expected_C = (A * 2) * B
        expected_D = A * 5
        sdfg(in1=A, in2=B, out1=C, out2=D)
        self.assertTrue(np.allclose(C, expected_C))
        self.assertTrue(np.allclose(D, expected_D))

    def test_multidim(self):
        sdfg = dace.SDFG('mapfission_multidim')
        sdfg.add_array('A', [2, 3], dace.float64)
        state = sdfg.add_state()
        me, mx = state.add_map('outer', dict(i='0:2', j='0:3'))

        nsdfg = dace.SDFG('nested')
        nsdfg.add_array('a', [1], dace.float64)
        nstate = nsdfg.add_state()
        t = nstate.add_tasklet('reset', {}, {'out'}, 'out = 0')
        a = nstate.add_write('a')
        nstate.add_edge(t, 'out', a, None, dace.Memlet.simple('a', '0'))
        nsdfg_node = state.add_nested_sdfg(nsdfg, None, {}, {'a'})

        state.add_edge(me, None, nsdfg_node, None, dace.Memlet())
        anode = state.add_write('A')
        state.add_memlet_path(nsdfg_node, mx, anode, src_conn='a', memlet=dace.Memlet.simple('A', 'i,j'))

        self.assertGreater(sdfg.apply_transformations(MapFission), 0)

        # Test
        A = np.random.rand(2, 3)
        sdfg(A=A)
        self.assertTrue(np.allclose(A, np.zeros_like(A)))

    def test_offsets(self):
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

        self.assertGreater(sdfg.apply_transformations(MapFission), 0)

        dace.propagate_memlets_sdfg(sdfg)
        sdfg.validate()

        # Test
        A = np.random.rand(20)
        expected = A.copy()
        expected[10:] += 3
        sdfg(A=A)
        self.assertTrue(np.allclose(A, expected))

    def test_offsets_array(self):
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

        self.assertGreater(sdfg.apply_transformations(MapFission), 0)

        dace.propagate_memlets_sdfg(sdfg)
        sdfg.validate()

        # Test
        A = np.random.rand(20)
        expected = A.copy()
        expected[10:] += 3
        sdfg(A=A)
        self.assertTrue(np.allclose(A, expected))


    def test_mapfission_with_symbols(self):
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
        nsdfg.parent = nstate
        nsdfg.parent_sdfg = sdfg
        nsdfg.parent_nsdfg_node = t
        state.add_node(t)
        state.add_nedge(me, t, dace.Memlet())
        state.add_memlet_path(t, mx, a, memlet=dace.Memlet('A[0, i]'), src_conn='inner_A')
        state.add_memlet_path(t, mx, b, memlet=dace.Memlet('B[0, i]'), src_conn='inner_B')


        num = sdfg.apply_transformations(MapFission)
        self.assertTrue(num == 1)

        A = np.ndarray((2, 10), dtype=np.int32)
        B = np.ndarray((2, 10), dtype=np.int32)
        sdfg(A=A, B=B, M=2, N=10)

        ref = np.full((10,), fill_value=2, dtype=np.int32)

        self.assertTrue(np.array_equal(A[0], ref))
        self.assertTrue(np.array_equal(B[0], ref))


    def test_two_edges_through_map(self):
        '''
        Tests MapFission in the case of a Map with a component that has two inputs from a single data container. In such
        cases, using `fill_scope_connectors` will lead to broken Map connectors. The tests confirms that new code in the
        transformation manually adding the appropriate Map connectors works properly.
        '''

        N = dace.symbol('N')

        sdfg = dace.SDFG('two_edges_through_map')
        sdfg.add_array('A', (N,), dace.int32)
        sdfg.add_array('B', (N,), dace.int32)

        state = sdfg.add_state('parent', is_start_state=True)
        me, mx = state.add_map('parent_map', {'i': '0:N'})

        nsdfg = dace.SDFG('nested_sdfg')
        nsdfg.add_array('inner_A', (N,), dace.int32)
        nsdfg.add_scalar('inner_B', dace.int32)

        nstate = nsdfg.add_state('child', is_start_state=True)
        na = nstate.add_access('inner_A')
        nb = nstate.add_access('inner_B')
        t = nstate.add_tasklet('tasklet', {'__in1', '__in2'}, {'__out'}, '__out = __in1 + __in2')
        nstate.add_edge(na, None, t, '__in1', dace.Memlet('inner_A[i]'))
        nstate.add_edge(na, None, t, '__in2', dace.Memlet('inner_A[N-i-1]'))
        nstate.add_edge(t, '__out', nb, None, dace.Memlet.from_array('inner_B', nsdfg.arrays['inner_B']))

        a = state.add_access('A')
        b = state.add_access('B')
        t = state.add_nested_sdfg(nsdfg, None, {'inner_A'}, {'inner_B'}, {'N': 'N', 'i': 'i'})
        state.add_memlet_path(a, me, t, memlet=dace.Memlet.from_array('A', sdfg.arrays['A']), dst_conn='inner_A')
        state.add_memlet_path(t, mx, b, memlet=dace.Memlet('B[i]'), src_conn='inner_B')
        
        num = sdfg.apply_transformations(MapFission)
        self.assertTrue(num == 1)

        A = np.arange(10, dtype=np.int32)
        B = np.ndarray((10,), dtype=np.int32)
        sdfg(A=A, B=B, N=10)

        ref = np.full((10,), fill_value=9, dtype=np.int32)

        self.assertTrue(np.array_equal(B, ref))

    def test_if_scope(self):

        @dace.program
        def map_with_if(A: dace.int32[10]):
            for i in dace.map[0:10]:
                j = i < 5
                if j:
                    A[i] = 0
                else:
                    A[i] = 1
        
        ref = np.array([0] * 5 + [1] * 5, dtype=np.int32)

        sdfg = map_with_if.to_sdfg(simplify=False)
        val0 = np.ndarray((10,), dtype=np.int32)
        sdfg(A=val0)
        self.assertTrue(np.array_equal(val0, ref))


        sdfg.apply_transformations(MapFission)
        val1 = np.ndarray((10,), dtype=np.int32)
        sdfg(A=val1)
        self.assertTrue(np.array_equal(val1, ref))


if __name__ == '__main__':
    unittest.main()
