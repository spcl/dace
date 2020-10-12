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



'''
    def build_tree(self, prune = False):

        self.root = TreeNode(None)
        processed = set()
        for _ in range(len(self.adjacency_list)):
            current = next(m for (m,s) in self.adjacency_list.items() if (len(s) == 0 or all([ms in processed for ms in s])) and m not in processed)
            # append current as a node to our tree
            # prune if necessary
            # *copy* all other treenodes from earlier
            # root branches, copy will do fine as this is read only

            # search for all nodes whose children contain current and
            # whose other children are already processed
            # this has suboptimal complexity but we don't really
            # care since what we are doing is NP

            # FORNOW: shallow copy from root children suffices
            print("***")
            print(current)
            children = {cm:n for (cm,n) in self.root.children.items() if cm in self.adjacency_list[current]}
            print(children)
            self.root.children[current] = TreeNode(current.map, children)
            print("***")
            processed.add(current)

    def traverse(self, current, node):
        current.append(node.map)
        yield current
        for child in node.children.values():
            yield from self.traverse(current, child)
        current.pop()

    def __iter__(self):
        print("ITER")
        print(self.root.children)
        for node in self.root.children.values():
            yield from self.traverse([], node)
'''
