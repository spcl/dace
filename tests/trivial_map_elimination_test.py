# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import TrivialMapElimination
import unittest


def trivial_map_sdfg():
    sdfg = dace.SDFG('trivial_map')
    sdfg.add_array('A', [5], dace.float64)
    sdfg.add_array('B', [5], dace.float64)
    state = sdfg.add_state()

    # Nodes
    read = state.add_read('A')
    map_entry, map_exit = state.add_map('map', dict(i='0:1'))
    tasklet = state.add_tasklet('tasklet', {'a'}, {'b'}, 'b = a')
    write = state.add_write('B')

    # Edges
    state.add_memlet_path(read, map_entry, tasklet, memlet=dace.Memlet.simple('A', '0'), dst_conn='a')
    state.add_memlet_path(tasklet, map_exit, write, memlet=dace.Memlet.simple('B', 'i'), src_conn='b')

    sdfg.validate()
    return sdfg


class TrivialMapEliminationTest(unittest.TestCase):
    def test_can_be_applied(self):
        graph = trivial_map_sdfg()

        count = graph.apply_transformations(TrivialMapElimination)

        self.assertGreater(count, 0)

    def test_removes_map(self):
        graph = trivial_map_sdfg()

        graph.apply_transformations(TrivialMapElimination)

        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 0)

    def test_raplaces_map_params_in_scope(self):
        # Tests if the 'i' in the range of the memlet to B gets replaced
        # with the value 'i' obtains in the map, namely '0'.

        graph = trivial_map_sdfg()

        graph.apply_transformations(TrivialMapElimination)

        state = graph.nodes()[0]
        B = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.AccessNode) and n.data == 'B'][0]
        out_memlet = state.in_edges(B)[0]
        self.assertEqual(out_memlet.data.subset, dace.subsets.Range([(0, 0, 1)]))


if __name__ == '__main__':
    unittest.main()
