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


def trivial_map_init_sdfg():
    sdfg = dace.SDFG('trivial_map_range_expanded')
    sdfg.add_array('B', [5, 1], dace.float64)
    state = sdfg.add_state()

    # Nodes
    map_entry_outer, map_exit_outer = state.add_map('map_outer', dict(j='0:5'))
    map_entry_inner, map_exit_inner = state.add_map('map_inner', dict(i='0:1'))

    tasklet = state.add_tasklet('tasklet', {}, {'b'}, 'b = 1')
    write = state.add_write('B')

    # Edges
    state.add_memlet_path(map_entry_outer, map_entry_inner, memlet=dace.Memlet())
    state.add_memlet_path(map_entry_inner, tasklet, memlet=dace.Memlet())

    state.add_memlet_path(tasklet,
                          map_exit_inner,
                          memlet=dace.Memlet.simple('B', 'j, i'),
                          src_conn='b',
                          dst_conn='IN_B')
    state.add_memlet_path(map_exit_inner,
                          map_exit_outer,
                          memlet=dace.Memlet.simple('B', 'j, 0'),
                          src_conn='OUT_B',
                          dst_conn='IN_B')
    state.add_memlet_path(map_exit_outer, write, memlet=dace.Memlet.simple('B', '0:5, 0'), src_conn='OUT_B')

    sdfg.validate()
    return sdfg


def trivial_map_with_dynamic_map_range_sdfg():
    sdfg = dace.SDFG("trivial_map_with_dynamic_map_range")
    state = sdfg.add_state("state1", is_start_block=True)

    for name in "ABC":
        sdfg.add_scalar(name, dtype=dace.float32, transient=False)
    A, B, C = (state.add_access(name) for name in "ABC")

    _, me, _ = state.add_mapped_tasklet(
        name="MAP",
        map_ranges=[("__i", "0:1"), ("__j", "10:11")],
        inputs={"__in": dace.Memlet("A[0]")},
        input_nodes={"A": A},
        code="__out = __in + 1",
        outputs={"__out": dace.Memlet("B[0]")},
        output_nodes={"B": B},
        external_edges=True,
    )
    state.add_edge(
        C,
        None,
        me,
        "dynamic_variable",
        dace.Memlet("C[0]"),
    )
    me.add_in_connector("dynamic_variable")
    sdfg.validate()

    return sdfg


def trivial_map_pseudo_init_sdfg():
    sdfg = dace.SDFG('trivial_map_range_expanded')
    sdfg.add_array('A', [5, 1], dace.float64)
    sdfg.add_array('B', [5, 1], dace.float64)
    state = sdfg.add_state()

    # Nodes
    map_entry_outer, map_exit_outer = state.add_map('map_outer', dict(j='0:5'))
    map_entry_inner, map_exit_inner = state.add_map('map_inner', dict(i='0:1'))

    read = state.add_read('A')
    tasklet = state.add_tasklet('tasklet', {'a'}, {'b'}, 'b = a')
    write = state.add_write('B')

    # Edges
    state.add_memlet_path(map_entry_outer, map_entry_inner, memlet=dace.Memlet())
    state.add_memlet_path(read,
                          map_entry_outer,
                          map_entry_inner,
                          memlet=dace.Memlet.simple('A', '0:5, 0'),
                          dst_conn='IN_A')
    state.add_memlet_path(map_entry_inner, tasklet, memlet=dace.Memlet())
    state.add_memlet_path(map_entry_inner,
                          tasklet,
                          memlet=dace.Memlet.simple('A', 'j, 0'),
                          src_conn='OUT_A',
                          dst_conn='a')

    state.add_memlet_path(tasklet,
                          map_exit_inner,
                          memlet=dace.Memlet.simple('B', 'j, i'),
                          src_conn='b',
                          dst_conn='IN_B')
    state.add_memlet_path(map_exit_inner,
                          map_exit_outer,
                          memlet=dace.Memlet.simple('B', 'j, 0'),
                          src_conn='OUT_B',
                          dst_conn='IN_B')
    state.add_memlet_path(map_exit_outer, write, memlet=dace.Memlet.simple('B', '0:5, 0'), src_conn='OUT_B')

    sdfg.validate()
    return sdfg


class TrivialMapEliminationTest(unittest.TestCase):
    """
    Tests the case where the map has an empty input edge
    """

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


class TrivialMapInitEliminationTest(unittest.TestCase):

    def test_can_be_applied(self):
        graph = trivial_map_init_sdfg()

        count = graph.apply_transformations(TrivialMapElimination, validate=False, validate_all=False)
        graph.validate()

        self.assertGreater(count, 0)

    def test_removes_map(self):
        graph = trivial_map_init_sdfg()

        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 2)

        graph.apply_transformations(TrivialMapElimination)

        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 1)

    def test_reconnects_edges(self):
        graph = trivial_map_init_sdfg()

        graph.apply_transformations(TrivialMapElimination)
        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 1)
        # Check that there is an outgoing edge from the map entry
        self.assertEqual(len(state.out_edges(map_entries[0])), 1)


class TrivialMapPseudoInitEliminationTest(unittest.TestCase):
    """
    Test cases where the map has an empty input and a non empty input
    """

    def test_can_be_applied(self):
        graph = trivial_map_pseudo_init_sdfg()

        count = graph.apply_transformations(TrivialMapElimination, validate=False, validate_all=False)
        graph.validate()

        self.assertGreater(count, 0)

    def test_removes_map(self):
        graph = trivial_map_pseudo_init_sdfg()

        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 2)

        graph.apply_transformations(TrivialMapElimination)

        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 1)

    def test_reconnects_edges(self):
        graph = trivial_map_pseudo_init_sdfg()

        graph.apply_transformations(TrivialMapElimination)
        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 1)
        # Check that there is an outgoing edge from the map entry
        self.assertEqual(len(state.out_edges(map_entries[0])), 1)


class TrivialMapEliminationWithDynamicMapRangesTest(unittest.TestCase):
    """
    Tests the case where the map has trivial ranges and dynamic map ranges.
    """

    def test_can_be_applied(self):
        graph = trivial_map_with_dynamic_map_range_sdfg()

        count = graph.apply_transformations(TrivialMapElimination)
        graph.validate()

        self.assertEqual(count, 1)

    def test_removes_map(self):
        graph = trivial_map_with_dynamic_map_range_sdfg()

        graph.apply_transformations(TrivialMapElimination)

        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 1)
        self.assertEqual(state.in_degree(map_entries[0]), 2)
        self.assertTrue(any(e.dst_conn.startswith("IN_") for e in state.in_edges(map_entries[0])))
        self.assertTrue(any(not e.dst_conn.startswith("IN_") for e in state.in_edges(map_entries[0])))

    def test_not_remove_dynamic_map_range(self):
        graph = trivial_map_with_dynamic_map_range_sdfg()

        count1 = graph.apply_transformations(TrivialMapElimination)
        self.assertEqual(count1, 1)

        count2 = graph.apply_transformations(TrivialMapElimination)
        self.assertEqual(count2, 0)


if __name__ == '__main__':
    unittest.main()
