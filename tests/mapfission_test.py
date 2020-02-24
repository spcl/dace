import copy
import dace
from dace.graph import nodes
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
    state.add_nedge(ome, scalar, dace.EmptyMemlet())
    state.add_memlet_path(
        rnode,
        ome,
        t1,
        memlet=dace.Memlet.simple('A', '2*i:2*i+2'),
        dst_conn='a')
    state.add_memlet_path(
        rnode,
        ome,
        ime2,
        t2,
        memlet=dace.Memlet.simple('A', '2*i+j'),
        dst_conn='a')
    state.add_memlet_path(
        t2, imx2, s24node, memlet=dace.Memlet.simple('s2', 'j'), src_conn='b')
    state.add_memlet_path(
        rnode,
        ome,
        ime3,
        t3,
        memlet=dace.Memlet.simple('A', '2*i:2*i+2'),
        dst_conn='a')
    state.add_memlet_path(
        t3,
        imx3,
        s34node,
        memlet=dace.Memlet.simple('s3out', '0'),
        src_conn='b')

    state.add_edge(t1, 'b', t4, 'ione', dace.Memlet.simple('s1', '0'))
    state.add_edge(s24node, None, t4, 'itwo', dace.Memlet.simple('s2', '0:2'))
    state.add_edge(s34node, None, t4, 'ithree', dace.Memlet.simple(
        's3out', '0'))
    state.add_edge(scalar, 'out', t4, 'sc', dace.Memlet.simple('scal', '0'))
    state.add_memlet_path(
        t4, omx, wnode, memlet=dace.Memlet.simple('B', 'i'), src_conn='out')

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
        topmap = next(
            node for node in state.nodes()
            if isinstance(node, nodes.MapEntry) and node.label == 'outer')
        subgraph = state.scope_subgraph(
            topmap, include_entry=False, include_exit=False)
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
        nstate.add_edge(irnode, None, task, 'inp', dace.Memlet.simple(
            'a', '0'))
        nstate.add_edge(task, 'out', iwnode, None, dace.Memlet.simple(
            't', '0'))

        # t->a state
        first_state = nstate
        nstate = nsdfg.add_state()
        irnode = nstate.add_read('t')
        task = nstate.add_tasklet('t2', {'inp'}, {'out'}, 'out = 3*inp')
        iwnode = nstate.add_write('b')
        nstate.add_edge(irnode, None, task, 'inp', dace.Memlet.simple(
            't', '0'))
        nstate.add_edge(task, 'out', iwnode, None, dace.Memlet.simple(
            'b', '0'))

        nsdfg.add_edge(first_state, nstate, dace.InterstateEdge())

        # Outer SDFG
        sdfg = dace.SDFG('nested_transient_fission')
        sdfg.add_array('A', [2], dace.float64)
        state = sdfg.add_state()
        rnode = state.add_read('A')
        wnode = state.add_write('A')
        me, mx = state.add_map('outer', dict(i='0:2'))
        nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'a'}, {'b'})
        state.add_memlet_path(
            rnode,
            me,
            nsdfg_node,
            dst_conn='a',
            memlet=dace.Memlet.simple('A', 'i'))
        state.add_memlet_path(
            nsdfg_node,
            mx,
            wnode,
            src_conn='b',
            memlet=dace.Memlet.simple('A', 'i'))

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
        t1 = state.add_tasklet('t1', {'i1'}, {'o1', 'o2'},
                               'o1 = i1 * 2; o2 = i1 * 5')
        t2 = state.add_tasklet('t2', {'i1', 'i2'}, {'o1'}, 'o1 = i1 * i2')
        state.add_memlet_path(
            in1, me, t1, dst_conn='i1', memlet=dace.Memlet.simple('in1', 'i'))
        state.add_memlet_path(
            in2, me, t2, dst_conn='i2', memlet=dace.Memlet.simple('in2', 'i'))
        state.add_edge(t1, 'o1', t2, 'i1', dace.Memlet.simple('tmp', '0'))
        state.add_memlet_path(
            t2,
            mx,
            out1,
            src_conn='o1',
            memlet=dace.Memlet.simple('out1', 'i'))
        state.add_memlet_path(
            t1,
            mx,
            out2,
            src_conn='o2',
            memlet=dace.Memlet.simple('out2', 'i'))

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

        state.add_edge(me, None, nsdfg_node, None, dace.EmptyMemlet())
        anode = state.add_write('A')
        state.add_memlet_path(
            nsdfg_node,
            mx,
            anode,
            src_conn='a',
            memlet=dace.Memlet.simple('A', 'i,j'))

        self.assertGreater(sdfg.apply_transformations(MapFission), 0)

        # Test
        A = np.random.rand(2, 3)
        sdfg(A=A)
        self.assertTrue(np.allclose(A, np.zeros_like(A)))


if __name__ == '__main__':
    unittest.main()
