import unittest
from dace.graph.nxutil import *


class TestRangeConversion(unittest.TestCase):
    def test_str_to_range(self):
        self.assertEqual(str_to_range("[1:3, 5:50:5, :7, 3:, :]"),
                         [("1", "3", None), ("5", "50", "5"),
                          (None, "7", None), ("3", None, None),
                          (None, None, None)])
        with self.assertRaises(ValueError):
            str_to_range("[::]")
        with self.assertRaises(ValueError):
            str_to_range("[1:2:1:1]")
        with self.assertRaises(ValueError):
            str_to_range("[1:2, ]")
        with self.assertRaises(ValueError):
            str_to_range("1:2]")
        with self.assertRaises(ValueError):
            str_to_range("[1:2")
        with self.assertRaises(TypeError):
            str_to_range(["(", "1", ":", "2", ")"])

    def test_range_to_str(self):
        self.assertEqual(
            range_to_str([(None, None, None), (1, None, None), (None, 3, None),
                          (None, None, 7), (3, 8, None), (None, 8, 1),
                          (4, None, 1), (1, 2, 3)],
                         limit_length=None), ("[(None):(None):(None), "
                                              "1:(None):(None), "
                                              "(None):3:(None), "
                                              "(None):(None):7, "
                                              "3:8:(None), "
                                              "(None):8, "
                                              "4:(None), "
                                              "1:2:3]"))
        self.assertEqual(range_to_str((1, 8, 2)), "[1:8:2]")
        with self.assertRaises(ValueError):
            range_to_str([(1, 2)])
        with self.assertRaises(TypeError):
            range_to_str(3)


class GraphSearchSpace(object):
    def __init__(self, graph, graph_node):
        self.graph = graph
        self.node = graph_node

    def evaluate(self):
        return self.node

    def children_iter(self):
        for _, child in self.graph.out_edges(self.node):
            yield GraphSearchSpace(self.graph, child)


class TestDLS(unittest.TestCase):
    def test_simple(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        graph.add_edges_from([(-5, -1), (-1, 1), (-2, 2), (-1, 5), (1, 6),
                              (2, 6), (-3, 3), (-4, 4), (3, 7), (4, 8),
                              (5, -4)])
        sspace = GraphSearchSpace(graph, -5)

        self.assertEqual(depth_limited_search(sspace, 0)[1], -5)
        self.assertEqual(depth_limited_search(sspace, 1)[1], -1)
        self.assertEqual(depth_limited_search(sspace, 2)[1], 5)
        self.assertEqual(depth_limited_search(sspace, 3)[1], 6)
        self.assertEqual(depth_limited_search(sspace, 4)[1], 6)
        self.assertEqual(depth_limited_search(sspace, 5)[1], 8)
        self.assertEqual(depth_limited_search(sspace, 1000)[1], 8)

    def test_iter(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        graph.add_edges_from([(-5, -1), (-1, 1), (-2, 2), (-1, 5), (1, 6),
                              (2, 6), (-3, 3), (-4, 4), (3, 7), (4, 8),
                              (5, -4)])
        sspace = GraphSearchSpace(graph, -5)

        def winner(sspace, depth):
            return max(((t, t.evaluate())
                        for t in depth_limited_dfs_iter(sspace, depth)),
                       key=lambda t: t[1])

        self.assertEqual(winner(sspace, 0)[1], -5)
        self.assertEqual(winner(sspace, 1)[1], -1)
        self.assertEqual(winner(sspace, 2)[1], 5)
        self.assertEqual(winner(sspace, 3)[1], 6)
        self.assertEqual(winner(sspace, 4)[1], 6)
        self.assertEqual(winner(sspace, 5)[1], 8)
        self.assertEqual(winner(sspace, 1000)[1], 8)


if __name__ == "__main__":
    unittest.main()
