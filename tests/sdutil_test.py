# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import unittest
import networkx as nx
from dace.sdfg.utils import *


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
        graph.add_edges_from([(-5, -1), (-1, 1), (-2, 2), (-1, 5), (1, 6), (2, 6), (-3, 3), (-4, 4), (3, 7), (4, 8),
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
        graph.add_edges_from([(-5, -1), (-1, 1), (-2, 2), (-1, 5), (1, 6), (2, 6), (-3, 3), (-4, 4), (3, 7), (4, 8),
                              (5, -4)])
        sspace = GraphSearchSpace(graph, -5)

        def winner(sspace, depth):
            return max(((t, t.evaluate()) for t in depth_limited_dfs_iter(sspace, depth)), key=lambda t: t[1])

        self.assertEqual(winner(sspace, 0)[1], -5)
        self.assertEqual(winner(sspace, 1)[1], -1)
        self.assertEqual(winner(sspace, 2)[1], 5)
        self.assertEqual(winner(sspace, 3)[1], 6)
        self.assertEqual(winner(sspace, 4)[1], 6)
        self.assertEqual(winner(sspace, 5)[1], 8)
        self.assertEqual(winner(sspace, 1000)[1], 8)


if __name__ == "__main__":
    unittest.main()
