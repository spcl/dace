# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import TrivialMapRangeElimination
import unittest


def trivial_map_range_sdfg():
    sdfg = dace.SDFG('trivial_map_range')
    sdfg.add_array('A', [5], dace.float64)
    sdfg.add_array('B', [5], dace.float64)
    state = sdfg.add_state()

    # Nodes
    read = state.add_read('A')
    map_entry, map_exit = state.add_map('map', dict(i='0:1', j='0:5'))
    tasklet = state.add_tasklet('tasklet', {'a'}, {'b'}, 'b = a')
    write = state.add_write('B')

    # Edges
    state.add_memlet_path(read, map_entry, tasklet, memlet=dace.Memlet.simple('A', '0'), dst_conn='a')
    state.add_memlet_path(tasklet, map_exit, write, memlet=dace.Memlet.simple('B', 'i'), src_conn='b')

    sdfg.validate()
    return sdfg


class TrivialMapRangeEliminationTest(unittest.TestCase):
    def test_can_be_applied(self):
        graph = trivial_map_range_sdfg()

        count = graph.apply_transformations(TrivialMapRangeElimination)

        self.assertGreater(count, 0)

    def test_transforms_map(self):
        graph = trivial_map_range_sdfg()

        graph.apply_transformations(TrivialMapRangeElimination)

        state = graph.nodes()[0]
        map_entry = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)][0]
        self.assertEqual(map_entry.map.params, ['j'])
        self.assertEqual(map_entry.map.range, dace.subsets.Range([(0, 4, 1)]))

    def test_raplaces_map_params_in_scope(self):
        graph = trivial_map_range_sdfg()

        graph.apply_transformations(TrivialMapRangeElimination)

        state = graph.nodes()[0]
        map_exit = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapExit)][0]
        out_memlet = state.in_edges(map_exit)[0]
        self.assertEqual(out_memlet.data.subset, dace.subsets.Range([(0, 0, 1)]))


if __name__ == '__main__':
    unittest.main()
