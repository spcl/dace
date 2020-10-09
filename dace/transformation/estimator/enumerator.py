""" This file implements the Enuerator class """

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from collections import deque

class TreeNode:
    def __init__(self, map, children):
        self.map = map
        if children = None:
            self.children = {}
        else:
            self.children = children
    def add_child(self, map, node = None):
        if node:
            self.children[map] = node
        else:
            self.children[map] = TreeNode(map)


@make_properties
class Enumerator:
    mode = Property(desc = "What the Iterator should return",
                    default = "subgraph",
                    choices = ["subgraph", "map_entries"])

    def __init__(self, sdfg, graph, subgraph):
        self.sdfg = sdfg
        self.graph = graph
        self.subgraph = subgraph

        # get hightest scope maps
        self.map_entries = helpers.get_highest_scope_maps(sdfg, graph, subgraph)

        # label all the map entries
        label = 0
        self.labels = {}
        for map_entry in map_entries:
            self.labels[map_entry] = label
            label += 1

        # create adjacency list
        # directionality of the DAG implies partial order
        self.adjacency_list = defaultdict(set)
        for map_entry in map_entries:
            map_exit = graph.exit_node(map_entry)
            for edge in graph.out_edges(map_exit):
                current_node = edge.dst
                if not isinstance(current, nodes.AccessNode):
                    continue
                for dst_edge in graph.out_edges(current_node):
                    if dst_edge.dst in map_entries:
                        self.adjacency_list[map_entry].add(current_node)

        # next up, create topologial labelling
        label = len(map_entries)
        self.labels = {}
        entries = map_entries.copy()
        while len(entries) > 0:

        # add Null as root with all the nodes as children
        self.adjacency_list[None] = set(map_entries)
        self.tree = []
        self.build_tree()
        print(adjacency_list)


    def build_tree(self, adjacency_list, prune = False):
        queue = deque()
        queue.append(None)
        # This is O(n^2)
        while len(queue) > 0:
            current = queue.l #TODO



    def __iter__():
        self
