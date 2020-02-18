import copy
import dace
from dace.graph import nodes
from dace.transformation.dataflow import MapFission
from dace.transformation.helpers import nest_state_subgraph
import numpy as np
import unittest

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
    rnode, ome, t1, memlet=dace.Memlet.simple('A', '2*i:2*i+2'), dst_conn='a')
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
    t3, imx3, s34node, memlet=dace.Memlet.simple('s3out', '0'), src_conn='b')

state.add_edge(t1, 'b', t4, 'ione', dace.Memlet.simple('s1', '0'))
state.add_edge(s24node, None, t4, 'itwo', dace.Memlet.simple('s2', '0:2'))
state.add_edge(s34node, None, t4, 'ithree', dace.Memlet.simple('s3out', '0'))
state.add_edge(scalar, 'out', t4, 'sc', dace.Memlet.simple('scal', '0'))
state.add_memlet_path(
    t4, omx, wnode, memlet=dace.Memlet.simple('B', 'i'), src_conn='out')

sdfg.validate()


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

        graph = copy.deepcopy(sdfg)
        graph.apply_transformations(MapFission, apply_once=True)
        graph(A=A, B=B)

        self.assertTrue(np.allclose(B, expected))

    def test_nested_sdfg(self):
        A, expected = config()
        B = np.random.rand(2)

        # Nest the subgraph within the outer map, then apply transformation
        graph = copy.deepcopy(sdfg)
        state = graph.nodes()[0]
        topmap = next(
            node for node in state.nodes()
            if isinstance(node, nodes.MapEntry) and node.label == 'outer')
        subgraph = state.scope_subgraph(
            topmap, include_entry=False, include_exit=False)
        nest_state_subgraph(graph, state, subgraph)
        graph.apply_transformations(MapFission, apply_once=True)
        graph(A=A, B=B)
        self.assertTrue(np.allclose(B, expected))


if __name__ == '__main__':
    unittest.main()
