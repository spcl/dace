""" This file implements the Enuerator class """

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from collections import deque, defaultdict, ChainMap

import dace.sdfg.nodes as nodes

from typing import Set, Union, List

class TreeNode:
    def __init__(self, map, children = None):
        self.map = map
        if children is None:
            self.children = {}
        else:
            self.children = children
    def add_child(self, map, node = None):
        if node:
            self.children[map] = node
        else:
            self.children[map] = TreeNode(map)


class Enumerator:
    mode = Property(desc = "What the Iterator should return",
                    default = "subgraph",
                    choices = ["subgraph", "map_entries"])

    def __init__(self, sdfg, graph, subgraph):
        self.sdfg = sdfg
        self.graph = graph
        self.subgraph = subgraph

        # get hightest scope maps
        map_entries = helpers.get_highest_scope_maps(sdfg, graph, subgraph)

        # create adjacency list
        self.adjacency_list = {m: set() for m in map_entries}
        for map_entry in map_entries:
            map_exit = graph.exit_node(map_entry)
            for edge in graph.out_edges(map_exit):
                current_node = edge.dst
                if not isinstance(current_node, nodes.AccessNode):
                    continue
                for dst_edge in graph.out_edges(current_node):
                    if dst_edge.dst in map_entries:
                        self.adjacency_list[map_entry].add(dst_edge.dst)
                        self.adjacency_list[dst_edge.dst].add(map_entry)


    def traverse(self, current: List, forbidden: Set, prune = False):
        if len(current) > 0:
            yield current.copy()
            go_next = set(m for c in current for m in self.adjacency_list[c] if m not in current and m not in forbidden)
        else:
            go_next = set(m for m in self.adjacency_list.keys())
        if len(go_next) > 0:
            # we can explore
            forbidden_current = set()
            for child in go_next:
                current.append(child)
                yield from self.traverse(current, forbidden | forbidden_current, prune)
                pp = current.pop()
                forbidden_current.add(child)

    def iterator(self):
        yield from self.traverse([], set(), False)

    def list(self):
        return list(self.iterator())

    def __iter__(self):
        yield from self.iterator()
