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

@make_properties
class Enumerator:
    mode = Property(desc = "What the Iterator should return",
                    default = "subgraph",
                    choices = ["subgraph", "map_entries"])

    def __init__(self, sdfg, graph, subgraph):
        self._sdfg = sdfg
        self._graph = graph
        self._scope_dict = graph.scope_dict(node_to_children = True)

        # get hightest scope maps
        map_entries = helpers.get_highest_scope_maps(sdfg, graph, subgraph)

        # create adjacency list
        self._adjacency_list = {m: set() for m in map_entries}
        for map_entry in map_entries:
            map_exit = graph.exit_node(map_entry)
            for edge in graph.out_edges(map_exit):
                current_node = edge.dst
                if not isinstance(current_node, nodes.AccessNode):
                    continue
                for dst_edge in graph.out_edges(current_node):
                    if dst_edge.dst in map_entries:
                        self._adjacency_list[map_entry].add(dst_edge.dst)
                        self._adjacency_list[dst_edge.dst].add(map_entry)


    def traverse(self, current: List, forbidden: Set, prune = False):
        if len(current) > 0:
            if self.mode == 'map_entries':
                yield current.copy()
            else:
                yield helpers.subgraph_from_maps(self._sdfg, self._graph, current, self._scope_dict)

            go_next = set(m for c in current for m in self._adjacency_list[c] if m not in current and m not in forbidden)
        else:
            go_next = set(m for m in self._adjacency_list.keys())
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

    def histogram(self, visual = True):
        old_mode = self.mode
        self.mode = 'map_entries'
        lst = self.list()
        print("Subgraph Statistics")
        print("-------------------")
        for i in range(1, 1 + len(self._adjacency_list)):
            no_elements = sum([len(sg) == i for sg in lst])
            if visual:
                print(i, no_elements, "*" * no_elements)
            else:
                print("Subgraphs with", i, "elements:", no_elements)
        self.mode = old_mode
