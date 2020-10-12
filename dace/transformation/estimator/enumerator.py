""" This file implements the Enuerator class """

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from collections import deque, defaultdict

import dace.sdfg.nodes as nodes

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
                if not isinstance(current_node, nodes.AccessNode):
                    continue
                for dst_edge in graph.out_edges(current_node):
                    if dst_edge.dst in map_entries:
                        self.adjacency_list[map_entry].add(current_node)

        self.adjacency_list[None] = set(map_entries)
        print(self.adjacency_list)

        # build tree
        self.build_tree()


    def build_tree(self, prune = False):

        self.root = TreeNode(None)
        queue = deque([self.root])
        processed = set()
        while len(queue) > 0:
            current = queue.popleft()
            # append current as a node to our tree
            # prune if necessary
            # *copy* all other treenodes from earlier
            # root branches, copy will do fine as this is read only

            # search for all nodes whose children contain current and
            # whose other children are already processed
            # this has suboptimal complexity but we don't really
            # care since what we are doing is NP
            for map, children in self.adjacency_list.items():
                if map not in processed \
                   and map not in queue \
                   and (len(children) == 0 \
                   or current in children and all([(c in processed or c in queue) for c in children])):

                    queue.append(map)

            # FORNOW: shallow copy from root children suffices
            children = {c:n for (c,n) in self.root.children.items() if c in self.adjacency_list[current]}
            self.root.children[current] = TreeNode(current.map, children)
            processed.add(current)


    def traverse(self, current, node):
        print("TRAVERSE")
        current.append(node.map)
        yield current
        for child in node.children.values():
            self.traverse(current, child)
        current.pop()

    def __iter__(self):
        print("ITER")
        print(self.root.map)
        print(self.root.children)
        for node in self.root.children.values():
            yield from self.traverse([], node)
